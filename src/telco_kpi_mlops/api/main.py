from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Telco KPI MLOps Platform",
    version="0.1.0",
    description="Forecasting and anomaly detection API for telecom KPI time series.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "telco-kpi-mlops-platform",
        "status": "running",
    }
