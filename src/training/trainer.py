"""MLflow training run for flight delay prediction."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.config import (
    FEATURE_COLS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    TARGET_COL,
)
from src.logger import get_logger

log = get_logger(__name__)

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 3,  # ~3x more on-time flights than delayed
    "eval_metric": "auc",
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
}


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train(
    df: pd.DataFrame,
    params: Optional[dict[str, Any]] = None,
    test_size: float = 0.2,
    run_name: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    final_params = {**_DEFAULT_PARAMS, **(params or {})}

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    log.info(
        "Starting training",
        rows=len(X),
        positive_rate=round(float(y.mean()), 4),
        features=len(FEATURE_COLS),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    run_name = run_name or f"xgboost-{int(time.time())}"

    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        run_id = run.info.run_id

        mlflow.log_params(final_params)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("positive_rate_train", round(float(y_train.mean()), 4))
        mlflow.log_param("feature_count", len(FEATURE_COLS))

        model = XGBClassifier(**final_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = _compute_metrics(y_test.values, y_pred, y_prob)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            XGBClassifier(**final_params), X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        metrics["cv_auc_mean"] = float(cv_scores.mean())
        metrics["cv_auc_std"] = float(cv_scores.std())

        mlflow.log_metrics(metrics)
        log.info("Training complete", **{k: round(v, 4) for k, v in metrics.items()})

        fi = {col: float(imp) for col, imp in zip(FEATURE_COLS, model.feature_importances_)}
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(fi_sorted, tmp, indent=2)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="feature_importance")
        os.unlink(tmp_path)

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        model_uri = f"runs:/{run_id}/model"
        log.info("Model logged", run_id=run_id, model_uri=model_uri)

    return {
        "run_id": run_id,
        "model_uri": model_uri,
        "metrics": metrics,
        "model": model,
        "feature_importances": fi_sorted,
    }
