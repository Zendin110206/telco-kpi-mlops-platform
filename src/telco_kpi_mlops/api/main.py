from __future__ import annotations

from fastapi import FastAPI

from telco_kpi_mlops.api.routes.health import router as health_router
from telco_kpi_mlops.api.routes.metadata import router as metadata_router

app = FastAPI(
    title="Telco KPI MLOps Platform",
    version="0.1.0",
    description="Forecasting and anomaly detection API for telecom KPI time series.",
)
app.include_router(health_router)
app.include_router(metadata_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "telco-kpi-mlops-platform",
        "status": "running",
    }
