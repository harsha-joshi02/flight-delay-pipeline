"""Ingestion flow: download → validate → clean → engineer features."""

from __future__ import annotations

from pathlib import Path

try:
    from prefect import flow, task, get_run_logger
    _PREFECT = True
except ImportError:
    _PREFECT = False
    def flow(fn=None, **_):
        return fn if fn else lambda f: f
    def task(fn=None, **_):
        return fn if fn else lambda f: f
    def get_run_logger():
        import logging
        return logging.getLogger(__name__)

from src.ingestion.downloader import download_range, load_raw
from src.ingestion.validator import validate, clean
from src.features.engineer import build_features, save_processed
from src.monitoring.drift import save_reference_snapshot
from src.config import INITIAL_MONTHS


@task(name="download-bts-data", retries=2, retry_delay_seconds=30)
def download_task(n_months: int = INITIAL_MONTHS) -> list[Path]:
    logger = get_run_logger()
    paths = download_range(n_months=n_months)
    logger.info(f"Downloaded {len(paths)} month(s) of BTS data")
    return paths


@task(name="validate-and-clean")
def validate_clean_task(paths: list[Path]):
    logger = get_run_logger()
    if not paths:
        raise ValueError("No raw data files to process")
    df = load_raw()
    logger.info(f"Loaded {len(df):,} raw rows")
    validate(df, raise_on_error=True)
    df = clean(df)
    logger.info(f"After cleaning: {len(df):,} rows")
    return df


@task(name="engineer-features")
def feature_task(df):
    logger = get_run_logger()
    feature_df, transformer = build_features(df, fit=True)
    transformer.save()
    path = save_processed(feature_df)
    save_reference_snapshot(feature_df)
    logger.info(f"Features saved to {path}, {len(feature_df):,} rows")
    return path


@flow(name="ingestion-flow", log_prints=True)
def ingestion_flow(n_months: int = INITIAL_MONTHS):
    paths = download_task(n_months)
    df = validate_clean_task(paths)
    feature_path = feature_task(df)
    return str(feature_path)


if __name__ == "__main__":
    ingestion_flow()
