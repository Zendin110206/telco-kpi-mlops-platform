from __future__ import annotations

from time import perf_counter

from telco_kpi_mlops.data.repository import (
    create_database_engine,
    get_first_kpi_series_key,
    load_anomaly_events,
    read_kpi_series,
)
from telco_kpi_mlops.features.transforms import (
    ensure_regular_frequency,
    fill_missing_points,
)
from telco_kpi_mlops.features.windows import split_train_test_by_time
from telco_kpi_mlops.models.anomaly import (
    assign_anomaly_flag,
    assign_severity,
    calculate_dynamic_threshold,
    calculate_residuals,
    select_anomaly_events,
    summarize_anomalies,
)
from telco_kpi_mlops.models.forecast import (
    forecast_next_steps,
    load_model_config,
    sarima_order_from_config,
    seasonal_order_from_config,
    train_sarima_model,
)


def main() -> None:
    """
    Main function to execute the batch scoring process, including loading data,
    training a forecasting model, calculating anomaly scores, and writing results to the database.
    """
    total_started_at = perf_counter()

    print("[1/5] Loading config and KPI series")
    data_started_at = perf_counter()
    config = load_model_config()
    forecasting_config = config["forecasting"]
    anomaly_config = config["anomaly_scoring"]
    model_config = config["model"]

    engine = create_database_engine()
    location_id, kpi_name = get_first_kpi_series_key(engine)
    raw_series = read_kpi_series(
        engine=engine,
        location_id=location_id,
        kpi_name=kpi_name,
        limit=int(forecasting_config["training_sample_rows"]),
    )
    regular_series = ensure_regular_frequency(
        raw_series,
        frequency_seconds=300,
    )
    prepared_series = fill_missing_points(regular_series)
    split = split_train_test_by_time(
        prepared_series,
        train_ratio=float(forecasting_config["train_ratio"]),
    )
    print(f"- data preparation seconds: {perf_counter() - data_started_at:.2f}")

    print("[2/5] Training forecasting baseline")
    training_started_at = perf_counter()
    order = sarima_order_from_config(config)
    seasonal_order = seasonal_order_from_config(config)
    fitted_model = train_sarima_model(
        train_values=split.train["kpi_value"],
        order=order,
        seasonal_order=seasonal_order,
    )
    forecast = forecast_next_steps(fitted_model, steps=len(split.test))
    print(f"- training and forecast seconds: {perf_counter() - training_started_at:.2f}")

    print("[3/5] Calculating residual anomaly scores")
    scoring_started_at = perf_counter()
    residual_frame = calculate_residuals(
        test_frame=split.test,
        forecast=forecast,
    )
    scored_frame = calculate_dynamic_threshold(
        residual_frame=residual_frame,
        window_size=int(forecasting_config["forecast_horizon_steps"]),
        threshold_multiplier=float(anomaly_config["threshold_std_multiplier"]),
    )
    scored_frame = assign_anomaly_flag(scored_frame)
    scored_frame = assign_severity(scored_frame)
    anomaly_summary = summarize_anomalies(scored_frame)
    anomaly_events = select_anomaly_events(
        scored_frame=scored_frame,
        model_version=str(model_config["version"]),
    )
    print(f"- scoring seconds: {perf_counter() - scoring_started_at:.2f}")

    print("[4/5] Writing anomaly events to PostgreSQL")
    write_started_at = perf_counter()
    load_summary = load_anomaly_events(
        engine=engine,
        events=anomaly_events,
    )
    print(f"- database write seconds: {perf_counter() - write_started_at:.2f}")

    print("[5/5] Scoring summary")
    print("Scored anomalies.")
    print(f"- location_id: {location_id}")
    print(f"- kpi_name: {kpi_name}")
    print(f"- scored rows: {anomaly_summary.scored_rows}")
    print(f"- events detected: {anomaly_summary.anomaly_rows}")
    print(f"- events inserted: {load_summary.inserted_rows}")
    print(f"- skipped duplicates: {load_summary.skipped_rows}")
    print(f"- low severity: {anomaly_summary.severity_counts['low']}")
    print(f"- medium severity: {anomaly_summary.severity_counts['medium']}")
    print(f"- high severity: {anomaly_summary.severity_counts['high']}")
    print(f"- critical severity: {anomaly_summary.severity_counts['critical']}")
    print(f"- total anomaly_events rows: {load_summary.total_rows_after}")
    print(f"- total seconds: {perf_counter() - total_started_at:.2f}")


if __name__ == "__main__":
    main()
