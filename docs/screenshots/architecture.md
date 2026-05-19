# Architecture

This document describes the planned architecture for Telco KPI MLOps Platform.

## High-Level Flow

```text
Raw KPI data
  -> validation
  -> cleaned records
  -> PostgreSQL
  -> forecasting model
  -> anomaly scoring
  -> FastAPI service
  -> monitoring and deployment
```

## Main Components

### Data Layer

The data layer is responsible for downloading, inspecting, validating, and loading KPI time-series data.

### Storage Layer

PostgreSQL stores cleaned KPI records, model metadata, and anomaly events.

### Modeling Layer

The first modeling version uses a forecasting baseline and residual-based anomaly scoring.

### API Layer

FastAPI exposes health, metadata, forecast, and anomaly endpoints.

### Ops Layer

Docker, tests, CI, MLflow, and monitoring are added gradually as the project matures.

## Version 1 Boundary

Version 1 focuses on a batch-oriented backend ML workflow. Real-time streaming and production-grade telecom integration are outside the first version.
