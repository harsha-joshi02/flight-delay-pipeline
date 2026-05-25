"""Training flow: load features → train → promotion gate."""

from __future__ import annotations

try:
    from prefect import flow, task, get_run_logger
except ImportError:
    def flow(fn=None, **_):
        return fn if fn else lambda f: f
    def task(fn=None, **_):
        return fn if fn else lambda f: f
    def get_run_logger():
        import logging
        return logging.getLogger(__name__)

from src.features.engineer import load_processed
from src.training.trainer import train
from src.evaluation.gate import evaluate_and_gate


@task(name="load-features")
def load_features_task():
    logger = get_run_logger()
    df = load_processed()
    logger.info(f"Loaded {len(df):,} rows for training")
    return df


@task(name="train-model")
def train_task(df):
    logger = get_run_logger()
    result = train(df, run_name="prefect-run")
    logger.info(f"Training complete — run_id={result['run_id']}, AUC={result['metrics']['auc']:.4f}")
    return result


@task(name="promotion-gate")
def gate_task(train_result: dict):
    logger = get_run_logger()
    report = evaluate_and_gate(train_result["run_id"])
    logger.info(f"Gate decision: {report.decision.value} — {report.reason}")
    return report


@flow(name="training-flow", log_prints=True)
def training_flow():
    df = load_features_task()
    train_result = train_task(df)
    report = gate_task(train_result)
    return {
        "decision": report.decision.value,
        "new_auc": report.new_auc,
        "run_id": report.new_run_id,
    }


if __name__ == "__main__":
    training_flow()
