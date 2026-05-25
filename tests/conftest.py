"""Shared pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_data_dirs(tmp_path, monkeypatch):
    """Redirect all data directory config to a temp dir for each test."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    for d in (raw, processed, models, reports):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_RAW_DIR", str(raw))
    monkeypatch.setenv("DATA_PROCESSED_DIR", str(processed))
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("REPORTS_DIR", str(reports))

    # Patch config module paths directly
    import src.config as cfg
    monkeypatch.setattr(cfg, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(cfg, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(cfg, "MODELS_DIR", models)
    monkeypatch.setattr(cfg, "REPORTS_DIR", reports)

    yield tmp_path
