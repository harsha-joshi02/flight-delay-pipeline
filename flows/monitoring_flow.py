"""Monitoring flow: load reference + predictions → drift report → optional retrain."""

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

from src.monitoring.drift import load_reference, load_predictions, generate_drift_report


@task(name="load-drift-data")
def load_data_task():
    logger = get_run_logger()
    reference = load_reference()
    predictions = load_predictions(limit=5000)
    if reference is None:
        raise RuntimeError("No reference snapshot found. Run ingestion_flow first.")
    if predictions is None or len(predictions) == 0:
        raise RuntimeError("No prediction logs found. Make some predictions first.")
    logger.info(f"Reference rows: {len(reference)}, prediction rows: {len(predictions)}")
    return reference, predictions


@task(name="generate-drift-report")
def drift_report_task(reference, predictions):
    logger = get_run_logger()
    summary = generate_drift_report(reference, predictions, report_name="weekly_drift")
    pred_drift = summary["prediction_drift_score"]
    pred_drift_display = f"{pred_drift:.4f}" if pred_drift is not None else "N/A"
    logger.info(
        f"Drift report done — dataset_drift={summary['dataset_drift_detected']}, "
        f"pred_drift={pred_drift_display}, "
        f"retrain_recommended={summary['retrain_recommended']}"
    )
    return summary


@task(name="trigger-retraining")
def maybe_retrain_task(summary: dict):
    logger = get_run_logger()
    if summary.get("retrain_recommended"):
        logger.warning("Drift detected — triggering retraining")
        from flows.training_flow import training_flow
        result = training_flow()
        logger.info(f"Retraining complete: {result}")
        return result
    logger.info("No significant drift detected — skipping retraining")
    return None


@flow(name="monitoring-flow", log_prints=True)
def monitoring_flow(auto_retrain: bool = True):
    reference, predictions = load_data_task()
    summary = drift_report_task(reference, predictions)
    if auto_retrain:
        maybe_retrain_task(summary)
    return summary


if __name__ == "__main__":
    monitoring_flow()
