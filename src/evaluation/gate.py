"""Model promotion gate: compares new model AUC against production and promotes if improvement exceeds threshold."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from src.config import (
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PROMOTION_THRESHOLD,
    REPORTS_DIR,
)
from src.logger import get_logger

log = get_logger(__name__)


class GateDecision(str, Enum):
    PROMOTED = "promoted"
    REJECTED = "rejected"
    FIRST_MODEL = "first_model"


@dataclass
class GateReport:
    decision: GateDecision
    new_run_id: str
    new_auc: float
    production_run_id: Optional[str]
    production_auc: Optional[float]
    auc_delta: Optional[float]
    threshold: float
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


def _get_production_model() -> Optional[ModelVersion]:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        return versions[0] if versions else None
    except Exception as exc:
        log.warning("Could not fetch production model", error=str(exc))
        return None


def _auc_from_run(run_id: str) -> Optional[float]:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        run = client.get_run(run_id)
        return run.data.metrics.get("auc")
    except Exception as exc:
        log.warning("Could not fetch AUC for run", run_id=run_id, error=str(exc))
        return None


def _transition_model(version: str, stage: str) -> None:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    log.info("Model version transitioned", version=version, stage=stage)


def _get_new_model_version(run_id: str) -> Optional[str]:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if versions:
            return versions[0].version
    except Exception as exc:
        log.warning("Could not find model version for run", run_id=run_id, error=str(exc))
    return None


def evaluate_and_gate(
    new_run_id: str,
    threshold: float = MODEL_PROMOTION_THRESHOLD,
    save_report: bool = True,
) -> GateReport:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    new_auc = _auc_from_run(new_run_id)
    if new_auc is None:
        raise ValueError(f"Could not retrieve AUC for run {new_run_id}")

    prod_model = _get_production_model()

    if prod_model is None:
        log.info("No production model found — promoting new model unconditionally", new_auc=new_auc)
        new_version = _get_new_model_version(new_run_id)
        if new_version:
            _transition_model(new_version, "Production")

        report = GateReport(
            decision=GateDecision.FIRST_MODEL,
            new_run_id=new_run_id,
            new_auc=new_auc,
            production_run_id=None,
            production_auc=None,
            auc_delta=None,
            threshold=threshold,
            reason="No existing production model; promoted as first model.",
        )
    else:
        prod_auc = _auc_from_run(prod_model.run_id)
        if prod_auc is None:
            log.warning("Could not retrieve production model AUC — defaulting to 0.5", prod_run_id=prod_model.run_id)
            prod_auc = 0.5

        delta = new_auc - prod_auc
        log.info("Comparing models", new_auc=new_auc, prod_auc=prod_auc, delta=delta, threshold=threshold)

        if delta >= threshold:
            new_version = _get_new_model_version(new_run_id)
            if new_version:
                _transition_model(new_version, "Production")
            decision = GateDecision.PROMOTED
            reason = (
                f"New model AUC ({new_auc:.4f}) exceeds production AUC ({prod_auc:.4f}) "
                f"by {delta:.4f}, above threshold {threshold}."
            )
        else:
            new_version = _get_new_model_version(new_run_id)
            if new_version:
                _transition_model(new_version, "Staging")
            decision = GateDecision.REJECTED
            reason = (
                f"New model AUC ({new_auc:.4f}) does not improve production AUC ({prod_auc:.4f}) "
                f"by required threshold {threshold} (delta={delta:.4f}). Moved to Staging for review."
            )

        report = GateReport(
            decision=decision,
            new_run_id=new_run_id,
            new_auc=new_auc,
            production_run_id=prod_model.run_id,
            production_auc=prod_auc,
            auc_delta=delta,
            threshold=threshold,
            reason=reason,
        )

    log.info("Gate decision", decision=report.decision.value, reason=report.reason)

    if save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / "gate_report.json"
        with open(report_path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        log.info("Gate report saved", path=str(report_path))

    return report


def load_production_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}/Production"
    log.info("Loading production model", uri=model_uri)
    return mlflow.pyfunc.load_model(model_uri)
