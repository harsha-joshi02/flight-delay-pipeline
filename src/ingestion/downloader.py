"""BTS On-Time Performance data downloader."""

import io
import zipfile
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import BTS_REQUIRED_COLS, DATA_RAW_DIR, INITIAL_MONTHS
from src.logger import get_logger

log = get_logger(__name__)

BTS_BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

_HTTP_TIMEOUT = 120


def _prev_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def _month_range(start_year: int, start_month: int, n_months: int) -> list[tuple[int, int]]:
    months = []
    y, m = start_year, start_month
    for _ in range(n_months):
        months.append((y, m))
        y, m = _prev_month(y, m)
    return list(reversed(months))


def download_month(year: int, month: int, out_dir: Path = DATA_RAW_DIR) -> Optional[Path]:
    """Download one month of BTS data and save as parquet. Returns path or None on failure."""
    out_path = out_dir / f"flights_{year}_{month:02d}.parquet"
    if out_path.exists():
        log.info("Already downloaded, skipping", year=year, month=month, path=str(out_path))
        return out_path

    url = BTS_BASE_URL.format(year=year, month=month)
    log.info("Downloading BTS data", year=year, month=month, url=url)

    try:
        resp = requests.get(url, timeout=_HTTP_TIMEOUT, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Failed to download BTS data", year=year, month=month, error=str(exc))
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv") and "readme" not in n.lower()]
            if not csv_names:
                log.error("No CSV found in ZIP", year=year, month=month, contents=zf.namelist())
                return None
            with zf.open(csv_names[0]) as fh:
                df = pd.read_csv(fh, low_memory=False)
    except Exception as exc:
        log.error("Failed to parse ZIP", year=year, month=month, error=str(exc))
        return None

    available = [c for c in BTS_REQUIRED_COLS if c in df.columns]
    missing = set(BTS_REQUIRED_COLS) - set(available)
    if missing:
        log.warning("Some expected columns missing in raw data", missing=list(missing))
    df = df[available].copy()

    df.to_parquet(out_path, index=False, compression="snappy")
    log.info("Saved raw data", year=year, month=month, rows=len(df), path=str(out_path))
    return out_path


def download_range(
    start_year: Optional[int] = None,
    start_month: Optional[int] = None,
    n_months: int = INITIAL_MONTHS,
) -> list[Path]:
    """Download a range of months. Defaults to the last n_months before today."""
    today = date.today()
    # BTS data lags ~2 months behind
    ref = date(today.year, today.month, 1)
    if ref.month <= 2:
        ref_year, ref_month = ref.year - 1, ref.month + 10
    else:
        ref_year, ref_month = ref.year, ref.month - 2

    sy = start_year or ref_year
    sm = start_month or ref_month

    months = _month_range(sy, sm, n_months)
    log.info("Downloading BTS data range", months=[(y, m) for y, m in months])

    paths: list[Path] = []
    for year, month in months:
        p = download_month(year, month)
        if p:
            paths.append(p)
    return paths


def load_raw(out_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load all downloaded parquet files into a single DataFrame."""
    files = sorted(out_dir.glob("flights_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw parquet files found in {out_dir}")
    log.info("Loading raw files", n_files=len(files))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
