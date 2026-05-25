"""Download and process BTS flight data.

Usage:
    python scripts/ingest.py --months 3
    python scripts/ingest.py --months 6 --no-refit
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR
from src.features.engineer import FlightFeatureTransformer, build_features, save_processed
from src.ingestion.downloader import download_range, load_raw
from src.ingestion.validator import clean, validate

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main(n_months: int = 3, refit: bool = True) -> None:
    log.info(f"Downloading last {n_months} month(s) of BTS data...")
    paths = download_range(n_months=n_months)
    log.info(f"Downloaded {len(paths)} file(s)")

    log.info("Loading and validating raw data...")
    df = load_raw()
    report = validate(df, raise_on_error=True)
    log.info(f"Validation passed — {report.row_count_raw:,} rows, {len(report.warnings)} warnings")

    df = clean(df)
    log.info(f"After cleaning: {len(df):,} rows")

    transformer_path = MODELS_DIR / "feature_pipeline.pkl"
    transformer = None
    if not refit and transformer_path.exists():
        log.info("Loading existing feature transformer")
        transformer = FlightFeatureTransformer.load(transformer_path)

    feature_df, transformer = build_features(df, transformer=transformer, fit=(refit or transformer is None))
    transformer.save(transformer_path)
    save_processed(feature_df)
    log.info(
        f"Features saved — {len(feature_df):,} rows, "
        f"positive rate {feature_df['is_delayed'].mean():.2%}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest BTS flight data")
    parser.add_argument("--months", type=int, default=3, help="Number of months to download")
    parser.add_argument("--no-refit", action="store_true", help="Reuse existing feature transformer")
    args = parser.parse_args()
    main(n_months=args.months, refit=not args.no_refit)
