from __future__ import annotations

from telco_kpi_mlops.data.repository import create_database_engine
from telco_kpi_mlops.models.evaluate import (
    DEFAULT_EVALUATION_REPORT_PATH,
    DEFAULT_FORECAST_ARTIFACT_PATH,
    build_model_evaluation_summary,
    save_evaluation_summary,
)


def main() -> None:
    """
    Main function to execute the model evaluation process,
    including loading data, building the evaluation summary, and saving the report.
    """
    engine = create_database_engine()
    summary = build_model_evaluation_summary(
        engine=engine,
        artifact_path=DEFAULT_FORECAST_ARTIFACT_PATH,
    )
    save_evaluation_summary(
        summary=summary,
        report_path=DEFAULT_EVALUATION_REPORT_PATH,
    )

    print("Model evaluation summary")
    print(f"- model_name: {summary.forecast.model_name}")
    print(f"- model_version: {summary.forecast.model_version}")
    print(f"- location_id: {summary.forecast.location_id}")
    print(f"- kpi_name: {summary.forecast.kpi_name}")
    print(f"- train rows: {summary.forecast.train_rows}")
    print(f"- test rows: {summary.forecast.test_rows}")
    print(f"- MAE: {summary.forecast.mae:.4f}")
    print(f"- RMSE: {summary.forecast.rmse:.4f}")
    print(f"- MAPE: {summary.forecast.mape:.4f}")
    print(f"- anomaly events: {summary.anomaly.total_events}")
    print(f"- anomaly rate: {summary.anomaly.anomaly_rate:.4%}")
    print(f"- low severity: {summary.anomaly.severity_counts['low']}")
    print(f"- medium severity: {summary.anomaly.severity_counts['medium']}")
    print(f"- high severity: {summary.anomaly.severity_counts['high']}")
    print(f"- critical severity: {summary.anomaly.severity_counts['critical']}")
    print(f"- saved report: {DEFAULT_EVALUATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
