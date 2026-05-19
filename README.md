# Telco KPI MLOps Platform

Production-style telecom KPI forecasting and anomaly detection service with Python, FastAPI, PostgreSQL, Docker, CI, and operator-focused MLOps.

## Overview

Telco KPI MLOps Platform is a backend-leaning machine learning project that connects telecommunication network analytics with practical ML engineering workflow.

The project is designed to ingest public network operator KPI time-series data, validate data quality, store cleaned records in PostgreSQL, train forecasting baselines, detect residual-based anomalies, and expose results through a FastAPI service.

This repository is built as a portfolio-grade learning project, with emphasis on reproducibility, API design, testing, containerization, CI, and deployment readiness.

## Why This Project

This project is intentionally not a dashboard-only data analyst project and not a notebook-only machine learning experiment.

The goal is to demonstrate a behind-the-scenes ML engineering workflow:

- data acquisition
- data validation
- database storage
- time-series forecasting
- anomaly scoring
- API serving
- Dockerized development
- automated testing
- CI/CD foundation
- deployment documentation

## Dataset

The planned dataset is the Network operator KPIs time series dataset from Zenodo:

<https://zenodo.org/records/8147768>

The dataset contains anonymized and scaled network operator KPI time-series data, including traffic-related and session-related indicators collected across multiple locations.

## Planned Architecture

```text
Zenodo dataset
  -> data download script
  -> data validation
  -> cleaned KPI records
  -> PostgreSQL
  -> forecasting baseline
  -> residual anomaly scoring
  -> FastAPI endpoints
  -> Docker and CI
```

## Tech Stack

- Python 3.12
- pandas, NumPy, statsmodels, scikit-learn
- FastAPI, Pydantic
- PostgreSQL, SQLAlchemy
- Docker, Docker Compose
- pytest, Ruff
- GitHub Actions
- MLflow
- Prometheus

## Project Status

Current phase:

```text
Repository foundation and project planning.
```

No production model or API is available yet.

## Repository Structure

```text
.
├── artifacts/
├── configs/
├── data/
├── docs/
├── scripts/
├── sql/
├── src/
│   └── telco_kpi_mlops/
└── tests/
```

## Scope

Version 1 focuses on a realistic but beginner-manageable backend ML system:

- batch dataset ingestion
- schema and quality validation
- PostgreSQL storage
- SARIMA forecasting baseline
- residual-based anomaly detection
- FastAPI service
- Docker local stack
- basic tests and CI

## Limitations

- This project uses public anonymized data, not private Indonesian operator data.
- This project is inspired by telecom network analytics workflows, but it is not a full NWDAF implementation.
- Initial anomaly detection is residual-threshold based, not supervised incident classification.
- Real-time streaming is outside the first version scope.

## License

MIT License.
