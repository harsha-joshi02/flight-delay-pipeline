from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

from src.config import (
    API_HOST,
    API_PORT,
    FEATURE_COLS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)
from src.features.engineer import FlightFeatureTransformer
from src.logger import get_logger
from src.monitoring.logger import PredictionLogger
from src.serving.schemas import (
    BatchFlightInput,
    BatchPredictionResponse,
    FlightInput,
    HealthResponse,
    PredictionResponse,
    PredictionResult,
)

log = get_logger(__name__)

_state: dict[str, Any] = {
    "model": None,
    "transformer": None,
    "model_version": None,
    "start_time": time.time(),
}


def _load_model_and_transformer() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}/Production"

    log.info("Loading production model", uri=model_uri)
    try:
        _state["model"] = mlflow.xgboost.load_model(model_uri)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        _state["model_version"] = versions[0].version if versions else "unknown"
        log.info("Model loaded", version=_state["model_version"])
    except Exception as exc:
        log.error("Failed to load production model", error=str(exc))
        _state["model"] = None

    try:
        _state["transformer"] = FlightFeatureTransformer.load()
        log.info("Feature transformer loaded")
    except Exception as exc:
        log.error("Failed to load feature transformer", error=str(exc))
        _state["transformer"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_and_transformer()
    _state["prediction_logger"] = PredictionLogger()
    yield
    log.info("Shutting down prediction API")


app = FastAPI(
    title="Flight Delay Prediction API",
    description="Predicts the probability of a 15+ minute arrival delay for US domestic flights.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _input_to_features(flight: FlightInput) -> pd.DataFrame:
    transformer: Optional[FlightFeatureTransformer] = _state.get("transformer")
    if transformer is None:
        raise HTTPException(status_code=503, detail="Feature transformer not loaded")

    raw = pd.DataFrame([{
        "Month": flight.month,
        "DayOfWeek": flight.day_of_week,
        "CRSDepTime": flight.dep_hour * 100,
        "Reporting_Airline": flight.carrier,
        "Origin": flight.origin,
        "Dest": flight.dest,
        "Distance": flight.distance,
        "CRSElapsedTime": flight.crs_elapsed_time,
    }])

    features = transformer.transform(raw)
    for col in FEATURE_COLS:
        if col not in features.columns:
            features[col] = 0
    return features[FEATURE_COLS]


def _confidence_tier(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "high"
    elif prob >= 0.6 or prob <= 0.4:
        return "medium"
    return "low"


def _predict_one(flight: FlightInput) -> PredictionResult:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = _input_to_features(flight)
    prob = float(model.predict_proba(features)[:, 1][0])

    return PredictionResult(
        delay_probability=round(prob, 4),
        is_delayed=prob >= 0.5,
        confidence=_confidence_tier(prob),
        threshold_used=0.5,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if _state["model"] is not None else "degraded",
        model_loaded=_state["model"] is not None,
        model_version=_state["model_version"],
        uptime_seconds=round(time.time() - _state["start_time"], 2),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(flight: FlightInput):
    result = _predict_one(flight)

    pred_logger: Optional[PredictionLogger] = _state.get("prediction_logger")
    if pred_logger:
        pred_logger.log(
            features=flight.model_dump(),
            prediction=result.delay_probability,
            model_version=_state["model_version"],
        )

    return PredictionResponse(
        input=flight,
        prediction=result,
        model_version=_state["model_version"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchFlightInput):
    responses = []
    pred_logger: Optional[PredictionLogger] = _state.get("prediction_logger")

    for flight in batch.flights:
        result = _predict_one(flight)
        if pred_logger:
            pred_logger.log(
                features=flight.model_dump(),
                prediction=result.delay_probability,
                model_version=_state["model_version"],
            )
        responses.append(
            PredictionResponse(
                input=flight,
                prediction=result,
                model_version=_state["model_version"],
            )
        )

    return BatchPredictionResponse(
        predictions=responses,
        total=len(responses),
        model_version=_state["model_version"],
    )


@app.get("/model/info")
async def model_info():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
    if not versions:
        return {"status": "no production model"}
    v = versions[0]
    run = client.get_run(v.run_id)
    return {
        "model_name": MLFLOW_MODEL_NAME,
        "version": v.version,
        "run_id": v.run_id,
        "stage": v.current_stage,
        "created_at": v.creation_timestamp,
        "metrics": run.data.metrics,
        "params": {k: val for k, val in run.data.params.items()
                   if k in ("n_estimators", "max_depth", "learning_rate")},
    }


if __name__ == "__main__":
    uvicorn.run("src.serving.app:app", host=API_HOST, port=API_PORT, reload=False)
