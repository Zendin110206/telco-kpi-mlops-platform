from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

DEFAULT_FORECAST_ARTIFACT_PATH = Path(
    "artifacts/models/sarima_baseline_r1_aggregated_internet_traffic.json"
)
DEFAULT_EVALUATION_REPORT_PATH = Path("artifacts/reports/evaluation_summary.json")
SEVERITY_LEVELS = ("low", "medium", "high", "critical")
MODEL_LIMITATIONS = [
    "Dataset is anonymized and scaled, not private operator production data.",
    "Anomaly events are residual-based candidates, not manually verified incidents.",
    "The baseline is evaluated on one KPI series for the current checkpoint.",
    "This is not a certified NWDAF implementation.",
    "This is not a replacement for production telecom monitoring systems.",
]


@dataclass(frozen=True)
class ForecastEvaluation:
    """Evaluation results for the forecasting model, including metadata and performance metrics."""

    model_name: str
    model_version: str
    location_id: str
    kpi_name: str
    train_rows: int
    test_rows: int
    mae: float
    rmse: float
    mape: float


@dataclass(frozen=True)
class AnomalyEvaluation:
    """Evaluation results for the anomaly detection model."""

    total_events: int
    anomaly_rate: float
    severity_counts: dict[str, int]


@dataclass(frozen=True)
class ModelEvaluationSummary:
    """
    Summary of the model evaluation,
    combining forecast and anomaly results, along with limitations.
    """

    forecast: ForecastEvaluation
    anomaly: AnomalyEvaluation
    limitations: list[str]


def load_forecast_artifact(
    artifact_path: Path = DEFAULT_FORECAST_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Load the forecast artifact from a JSON file and return it as a dictionary."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"Forecast artifact does not exist: {artifact_path}")

    with artifact_path.open("r", encoding="utf-8") as file:
        artifact = json.load(file)

    if not isinstance(artifact, dict):
        raise ValueError("Forecast artifact must be a JSON object.")

    return artifact


def build_forecast_evaluation(artifact: dict[str, Any]) -> ForecastEvaluation:
    """Build a ForecastEvaluation dataclass instance from the loaded artifact dictionary."""
    metrics = artifact["metrics"]

    return ForecastEvaluation(
        model_name=str(artifact["model_name"]),
        model_version=str(artifact["model_version"]),
        location_id=str(artifact["location_id"]),
        kpi_name=str(artifact["kpi_name"]),
        train_rows=int(artifact["train_rows"]),
        test_rows=int(artifact["test_rows"]),
        mae=float(metrics["mae"]),
        rmse=float(metrics["rmse"]),
        mape=float(metrics["mape"]),
    )


def count_anomaly_events_by_severity(
    engine: Engine,
    model_version: str,
    location_id: str,
    kpi_name: str,
) -> dict[str, int]:
    """
    Query the database to count anomaly events by severity for the
    given model version, location, and KPI.
    """
    query = text(
        """
        SELECT severity, COUNT(*) AS event_count
        FROM anomaly_events
        WHERE model_version = :model_version
          AND location_id = :location_id
          AND kpi_name = :kpi_name
        GROUP BY severity
        ORDER BY severity
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(
            query,
            {
                "model_version": model_version,
                "location_id": location_id,
                "kpi_name": kpi_name,
            },
        )
        raw_counts = {str(row.severity): int(row.event_count) for row in rows}

    return {severity: int(raw_counts.get(severity, 0)) for severity in SEVERITY_LEVELS}


def build_anomaly_evaluation(
    severity_counts: dict[str, int],
    scored_rows: int,
) -> AnomalyEvaluation:
    """
    Build an AnomalyEvaluation dataclass instance from the
    severity counts and total scored rows.
    """
    if scored_rows <= 0:
        raise ValueError("scored_rows must be greater than zero.")

    total_events = sum(severity_counts.values())

    return AnomalyEvaluation(
        total_events=total_events,
        anomaly_rate=total_events / scored_rows,
        severity_counts=severity_counts,
    )


def build_model_evaluation_summary(
    engine: Engine,
    artifact_path: Path = DEFAULT_FORECAST_ARTIFACT_PATH,
) -> ModelEvaluationSummary:
    """
    Build the complete ModelEvaluationSummary by
    loading the forecast artifact, counting anomaly events, and combining results.
    """
    artifact = load_forecast_artifact(artifact_path)
    forecast = build_forecast_evaluation(artifact)
    severity_counts = count_anomaly_events_by_severity(
        engine=engine,
        model_version=forecast.model_version,
        location_id=forecast.location_id,
        kpi_name=forecast.kpi_name,
    )
    anomaly = build_anomaly_evaluation(
        severity_counts=severity_counts,
        scored_rows=forecast.test_rows,
    )

    return ModelEvaluationSummary(
        forecast=forecast,
        anomaly=anomaly,
        limitations=MODEL_LIMITATIONS,
    )


def save_evaluation_summary(
    summary: ModelEvaluationSummary,
    report_path: Path = DEFAULT_EVALUATION_REPORT_PATH,
) -> None:
    """Save the ModelEvaluationSummary to a JSON file at the specified path."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )
