"""Streamlit dashboard: live prediction, model metrics, drift reports, MLflow history."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "data/reports"))

st.set_page_config(
    page_title="Flight Delay ML Pipeline",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)



@st.cache_data(ttl=60)
def fetch_mlflow_runs() -> pd.DataFrame:
    """Pull all runs from the MLflow experiment via REST API."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name("flight-delay-prediction")
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=50,
        )
        rows = []
        for r in runs:
            rows.append({
                "run_id": r.info.run_id[:8],
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": datetime.fromtimestamp(r.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
                "auc": r.data.metrics.get("auc"),
                "f1": r.data.metrics.get("f1"),
                "precision": r.data.metrics.get("precision"),
                "recall": r.data.metrics.get("recall"),
                "avg_precision": r.data.metrics.get("avg_precision"),
                "cv_auc_mean": r.data.metrics.get("cv_auc_mean"),
                "model_type": r.data.params.get("model_type", "xgboost"),
                "n_estimators": r.data.params.get("n_estimators"),
                "max_depth": r.data.params.get("max_depth"),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Could not connect to MLflow: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_health() -> dict:
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False, "model_version": None, "uptime_seconds": 0}


def load_drift_summary() -> dict | None:
    path = REPORTS_DIR / "drift_summary_latest.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_gate_report() -> dict | None:
    path = REPORTS_DIR / "gate_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


st.sidebar.title("✈ Flight Delay ML")
health = fetch_health()
status_color = "green" if health["status"] == "ok" else "red"
st.sidebar.markdown(
    f"**API Status:** :{status_color}[{health['status'].upper()}]  \n"
    f"**Model Version:** {health.get('model_version') or 'N/A'}  \n"
    f"**Uptime:** {int(health.get('uptime_seconds', 0) // 60)} min"
)

page = st.sidebar.radio(
    "Navigate",
    ["Live Prediction", "Model Performance", "Drift Reports", "MLflow Run History"],
)

if page == "Live Prediction":
    st.title("Live Flight Delay Prediction")
    st.markdown("Enter flight details to get the predicted probability of a 15+ minute arrival delay.")

    carriers = ["AA", "DL", "UA", "WN", "AS", "B6", "NK", "F9", "G4", "HA"]
    airports = ["JFK", "LAX", "ORD", "ATL", "DFW", "DEN", "SFO", "LAS", "SEA", "MIA",
                "BOS", "EWR", "CLT", "PHX", "IAH", "MSP", "DTW", "PHL", "LGA", "BWI"]

    col1, col2, col3 = st.columns(3)
    with col1:
        carrier = st.selectbox("Airline", carriers)
        origin = st.selectbox("Origin Airport", airports)
        dest = st.selectbox("Destination Airport", [a for a in airports if a != origin])

    with col2:
        month = st.slider("Month", 1, 12, datetime.now().month)
        day_of_week = st.selectbox("Day of Week", list(range(1, 8)),
                                    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x-1])
        dep_hour = st.slider("Departure Hour", 0, 23, 9)

    with col3:
        distance = st.number_input("Distance (miles)", min_value=50, max_value=5000, value=800, step=50)
        crs_elapsed = st.number_input("Scheduled Flight Time (min)", min_value=30, max_value=600, value=120, step=10)

    if st.button("Predict Delay", type="primary"):
        payload = {
            "month": month,
            "day_of_week": day_of_week,
            "dep_hour": dep_hour,
            "carrier": carrier,
            "origin": origin,
            "dest": dest,
            "distance": float(distance),
            "crs_elapsed_time": float(crs_elapsed),
        }
        with st.spinner("Calling prediction API..."):
            try:
                resp = httpx.post(f"{API_BASE}/predict", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()["prediction"]

                prob = result["delay_probability"]
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Delay Probability", f"{prob:.1%}")
                col_b.metric("Prediction", "DELAYED" if result["is_delayed"] else "ON TIME")
                col_c.metric("Confidence", result["confidence"].upper())

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Delay Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "red" if prob >= 0.5 else "green"},
                        "steps": [
                            {"range": [0, 30], "color": "lightgreen"},
                            {"range": [30, 60], "color": "lightyellow"},
                            {"range": [60, 100], "color": "lightsalmon"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 4}, "value": 50},
                    },
                ))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure the FastAPI service is running (`docker-compose up api`)")


elif page == "Model Performance":
    st.title("Model Performance")
    gate_report = load_gate_report()

    if gate_report:
        st.subheader("Last Gate Decision")
        col1, col2, col3 = st.columns(3)
        decision = gate_report["decision"]
        color = "green" if decision in ("promoted", "first_model") else "orange"
        col1.metric("Decision", decision.upper())
        col2.metric("New AUC", f"{gate_report['new_auc']:.4f}")
        if gate_report.get("production_auc"):
            delta_val = gate_report.get("auc_delta", 0)
            col3.metric("vs Production", f"{gate_report['production_auc']:.4f}",
                        delta=f"{delta_val:+.4f}")
        st.caption(gate_report.get("reason", ""))

    st.subheader("MLflow Run Metrics")
    df = fetch_mlflow_runs()
    if df.empty:
        st.warning("No MLflow runs found. Make sure MLflow is running and models have been trained.")
    else:
        metric_cols = ["auc", "f1", "precision", "recall", "avg_precision"]
        available_metrics = [c for c in metric_cols if c in df.columns and df[c].notna().any()]

        selected_metric = st.selectbox("Plot metric", available_metrics, index=0)
        plot_df = df[df[selected_metric].notna()].copy()
        fig = px.line(
            plot_df.sort_values("start_time"),
            x="start_time", y=selected_metric,
            markers=True,
            title=f"{selected_metric.upper()} across training runs",
            labels={"start_time": "Run Time", selected_metric: selected_metric.upper()},
        )
        fig.add_hline(y=0.7, line_dash="dot", annotation_text="Target AUC 0.70",
                      line_color="green")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("All Runs Table")
        st.dataframe(df[["start_time", "run_name", "model_type", "auc", "f1",
                          "precision", "recall", "n_estimators", "max_depth"]],
                     use_container_width=True)


elif page == "Drift Reports":
    st.title("Data & Prediction Drift")
    summary = load_drift_summary()

    if summary is None:
        st.info("No drift reports found. The monitoring flow runs weekly (Monday 06:00 UTC).")
        st.code("python -m flows.monitoring_flow", language="bash")
    else:
        ts = summary.get("timestamp", "unknown")
        st.caption(f"Last report: {ts}")

        col1, col2, col3 = st.columns(3)
        drift_detected = summary.get("dataset_drift", False)
        col1.metric("Dataset Drift", "YES" if drift_detected else "NO",
                    delta=None,
                    delta_color="inverse" if drift_detected else "normal")
        col2.metric("Drifted Columns",
                    f"{summary.get('drifted_columns_count', 0)} / {summary.get('total_columns', 0)}")
        pred_drift = summary.get("prediction_drift") or {}
        col3.metric("Prediction Drift Score",
                    f"{pred_drift.get('drift_score', 0):.4f}" if pred_drift.get('drift_score') else "N/A")

        if summary.get("retrain_triggered"):
            st.error("Retraining was triggered by this report.")
        else:
            st.success("No retraining was triggered.")

        # Column drift table
        col_drift = summary.get("column_drift", {})
        if col_drift:
            st.subheader("Per-column Drift")
            drift_df = pd.DataFrame([
                {
                    "feature": col,
                    "drift_detected": v["drift_detected"],
                    "drift_score": round(v.get("drift_score") or 0, 4),
                    "test": v.get("stattest", ""),
                }
                for col, v in col_drift.items()
            ]).sort_values("drift_score", ascending=False)

            def highlight_drift(row):
                return ["background-color: #ffcccc" if row.drift_detected else "" for _ in row]

            st.dataframe(
                drift_df.style.apply(highlight_drift, axis=1),
                use_container_width=True,
            )

        # HTML report link
        html_report = summary.get("report_html")
        if html_report and Path(html_report).exists():
            st.subheader("Full Evidently Report")
            with open(html_report, "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)


elif page == "MLflow Run History":
    st.title("MLflow Run History")
    st.markdown(f"Tracking server: [{MLFLOW_URI}]({MLFLOW_URI})")

    df = fetch_mlflow_runs()
    if df.empty:
        st.warning("No runs found. Make sure MLflow is reachable.")
    else:
        st.metric("Total Runs", len(df))

        col1, col2 = st.columns(2)
        with col1:
            best_auc = df["auc"].max() if "auc" in df.columns else None
            if best_auc:
                st.metric("Best AUC", f"{best_auc:.4f}")
        with col2:
            best_f1 = df["f1"].max() if "f1" in df.columns else None
            if best_f1:
                st.metric("Best F1", f"{best_f1:.4f}")

        # Feature importance (from most recent run)
        fi_dir = REPORTS_DIR.parent / "data" / "reports"
        fi_files = list(Path("data/reports").glob("**/feature_importance/*.json")) if Path("data/reports").exists() else []
        if fi_files:
            with open(fi_files[0]) as f:
                fi = json.load(f)
            fi_df = pd.DataFrame(list(fi.items()), columns=["feature", "importance"]).head(12)
            fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                         title="Feature Importances (most recent run)")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True)

        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()
