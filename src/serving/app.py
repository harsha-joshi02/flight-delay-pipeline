from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

import mlflow
import mlflow.artifacts
import mlflow.pyfunc
from mlflow import MlflowClient

from src.config import (
    API_HOST,
    API_PORT,
    FEATURE_COLS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)
from src.features.engineer import FlightFeatureTransformer, PIPELINE_FILENAME
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
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if not versions:
            raise RuntimeError("No production model version found")

        version = versions[0]
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{version.version}"
        transformer_uri = f"{version.source.rstrip('/')}/{PIPELINE_FILENAME}"

        log.info("Loading production model", uri=model_uri)
        model = mlflow.xgboost.load_model(model_uri)
        transformer_path = mlflow.artifacts.download_artifacts(
            artifact_uri=transformer_uri,
            tracking_uri=MLFLOW_TRACKING_URI,
        )
        transformer = FlightFeatureTransformer.load(Path(transformer_path))

        _state["model"] = model
        _state["transformer"] = transformer
        _state["model_version"] = version.version
        log.info("Model and feature transformer loaded", version=version.version)
    except Exception as exc:
        log.error("Failed to load production model and transformer", error=str(exc))
        _state["model"] = None
        _state["transformer"] = None
        _state["model_version"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _state.get("model") is None or _state.get("transformer") is None:
        _load_model_and_transformer()
    if "prediction_logger" not in _state:
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


def _predict_one(flight: FlightInput) -> tuple[PredictionResult, dict[str, float]]:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = _input_to_features(flight)
    prob = float(model.predict_proba(features)[:, 1][0])

    result = PredictionResult(
        delay_probability=round(prob, 4),
        is_delayed=prob >= 0.5,
        confidence=_confidence_tier(prob),
        threshold_used=0.5,
    )
    return result, features.iloc[0].to_dict()


@app.get("/health", response_model=HealthResponse)
async def health(response: Response):
    ready = _state.get("model") is not None and _state.get("transformer") is not None
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=ready,
        model_version=_state.get("model_version"),
        uptime_seconds=round(time.time() - _state.get("start_time", time.time()), 2),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(flight: FlightInput):
    result, features = _predict_one(flight)

    pred_logger: Optional[PredictionLogger] = _state.get("prediction_logger")
    if pred_logger:
        pred_logger.log(
            features=features,
            prediction=result.delay_probability,
            model_version=_state["model_version"],
            raw_input=flight.model_dump(),
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
        result, features = _predict_one(flight)
        if pred_logger:
            pred_logger.log(
                features=features,
                prediction=result.delay_probability,
                model_version=_state["model_version"],
                raw_input=flight.model_dump(),
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
