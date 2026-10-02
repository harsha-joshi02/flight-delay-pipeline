"""Central configuration loaded from environment / .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "flight-delay-prediction")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "flight-delay-model")

MODEL_PROMOTION_THRESHOLD: float = float(os.getenv("MODEL_PROMOTION_THRESHOLD", "0.005"))
DELAY_THRESHOLD_MINUTES: int = int(os.getenv("DELAY_THRESHOLD_MINUTES", "15"))

DATA_RAW_DIR: Path = ROOT_DIR / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR: Path = ROOT_DIR / os.getenv("DATA_PROCESSED_DIR", "data/processed")
DATA_REFERENCE_DIR: Path = ROOT_DIR / os.getenv("DATA_REFERENCE_DIR", "data/reference")
MODELS_DIR: Path = ROOT_DIR / os.getenv("MODELS_DIR", "data/models")
REPORTS_DIR: Path = ROOT_DIR / os.getenv("REPORTS_DIR", "data/reports")

for _p in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_REFERENCE_DIR, MODELS_DIR, REPORTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

INITIAL_MONTHS: int = int(os.getenv("INITIAL_MONTHS", "12"))

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

BTS_REQUIRED_COLS = [
    "Year", "Month", "DayofMonth", "DayOfWeek",
    "Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Origin", "Dest", "CRSDepTime", "DepDelay",
    "CRSArrTime", "ArrDelay", "Cancelled", "Diverted",
    "CRSElapsedTime", "Distance",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]

FEATURE_COLS = [
    "month", "day_of_week", "dep_hour", "is_weekend", "season",
    "carrier_encoded", "origin_encoded", "dest_encoded", "route_encoded",
    "distance", "crs_elapsed_time", "dep_time_bucket",
]

TARGET_COL = "is_delayed"
