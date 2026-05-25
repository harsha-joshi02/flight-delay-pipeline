"""Train the flight delay model and optionally promote it to production.

Usage:
    python scripts/train.py
    python scripts/train.py --no-gate
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.gate import evaluate_and_gate, GateDecision
from src.features.engineer import load_processed
from src.training.trainer import train

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main(skip_gate: bool = False) -> None:
    log.info("Loading processed features...")
    df = load_processed()
    log.info(f"Loaded {len(df):,} rows")

    log.info("Training XGBoost model...")
    result = train(df)
    log.info(
        f"Training done — AUC={result['metrics']['auc']:.4f}, "
        f"F1={result['metrics']['f1']:.4f}, "
        f"run_id={result['run_id']}"
    )

    if skip_gate:
        log.info("Skipping gate check — model logged to MLflow")
        return

    log.info("Running model promotion gate...")
    report = evaluate_and_gate(result["run_id"], save_report=True)

    log.info("=" * 50)
    log.info(f"Gate decision : {report.decision.value.upper()}")
    log.info(f"New AUC       : {report.new_auc:.4f}")
    if report.production_auc is not None:
        log.info(f"Prod AUC      : {report.production_auc:.4f}")
        log.info(f"Delta         : {report.auc_delta:+.4f}")
    log.info(f"Reason        : {report.reason}")
    log.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flight delay model")
    parser.add_argument("--no-gate", action="store_true", help="Skip promotion gate")
    args = parser.parse_args()
    main(skip_gate=args.no_gate)
