# Flight Delay Prediction — ML Pipeline

Predicts the probability of a 15+ minute US domestic flight delay. Ingests BTS On-Time data, trains XGBoost with MLflow tracking, serves predictions via FastAPI, monitors for data drift with Evidently, and displays everything in a Streamlit dashboard — orchestrated with Prefect.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Prefect Orchestration                        │
│                                                                     │
│  ┌────────────────┐    ┌────────────────┐    ┌──────────────────┐  │
│  │  Ingestion     │───▶│   Training     │───▶│   Monitoring     │  │
│  │  Flow          │    │   Flow         │    │   Flow           │  │
│  │                │    │                │    │                  │  │
│  │ BTS Download   │    │ MLflow Run     │    │ Evidently Drift  │  │
│  │ Validate       │    │ XGBoost        │    │ Retrain Trigger  │  │
│  │ Clean          │    │ Promotion Gate │    │                  │  │
│  │ Feature Eng.   │    │                │    │                  │  │
│  └────────────────┘    └────────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                       │
          ▼                       ▼
  ┌──────────────┐       ┌──────────────────┐
  │  PostgreSQL  │       │  MLflow Tracking │
  │  Predictions │       │  Server          │
  │  Log         │       └──────────────────┘
  └──────────────┘                │
                         ┌────────▼─────────┐
                         │  FastAPI Serving  │
                         │  /predict         │
                         │  /predict/batch   │
                         │  /health          │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Streamlit        │
                         │  Dashboard        │
                         └──────────────────┘
```

**GitHub Actions:**
- `ci_cd.yml` — lint + test on every push to `main`
- `weekly_retrain.yml` — every Sunday 01:00 UTC: ingest → train → gate → monitor

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | BTS On-Time Performance (public) |
| Orchestration | Prefect 2.x |
| Experiment Tracking | MLflow |
| Model | XGBoost |
| Serving | FastAPI + Uvicorn |
| Monitoring | Evidently AI |
| Dashboard | Streamlit + Plotly |
| Storage | PostgreSQL + Parquet |
| Containers | Docker + docker-compose |
| CI/CD | GitHub Actions |

---

## Quick Start

```bash
# Start all services
docker compose up -d --build

# Bootstrap the model (run once after services are up)
python scripts/ingest.py --months 10
python scripts/train.py
docker compose restart api
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |
| Prefect | http://localhost:4200 |
| Streamlit | http://localhost:8501 |

---

## Running Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

mlflow server --host 127.0.0.1 --port 5001
python scripts/ingest.py --months 10
python scripts/train.py
uvicorn src.serving.app:app --reload --port 8000
streamlit run dashboard/app.py
```

---

## Register Prefect Schedules

```bash
python flows/deploy_flows.py
```

Registers:
- **ingestion** — 1st of every month at 02:00 UTC
- **training** — on-demand
- **monitoring** — every Monday at 06:00 UTC

---

## Tests

```bash
pytest --cov=src --cov-report=term-missing
```

---

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "month": 6, "day_of_week": 5, "dep_hour": 7,
    "carrier": "AA", "origin": "JFK", "dest": "LAX",
    "distance": 2475, "crs_elapsed_time": 330
  }'
```

Response:
```json
{
  "prediction": {
    "delay_probability": 0.3142,
    "is_delayed": false,
    "confidence": "medium",
    "threshold_used": 0.5
  },
  "model_version": "3"
}
```

---

## GitHub Secrets

Required for `weekly_retrain.yml`:

| Secret | Description |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `DATABASE_URL` | Postgres connection string |

---

## Data Source

**Bureau of Transportation Statistics — On-Time Performance**
https://transtats.bts.gov/DL_SelectFields.aspx
