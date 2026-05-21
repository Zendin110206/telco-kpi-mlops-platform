from __future__ import annotations

from pathlib import Path
from time import perf_counter

from telco_kpi_mlops.data.repository import (
    create_database_engine,
    get_first_kpi_series_key,
    read_kpi_series,
)
from telco_kpi_mlops.features.transforms import (
    ensure_regular_frequency,
    fill_missing_points,
)
from telco_kpi_mlops.features.windows import split_train_test_by_time
from telco_kpi_mlops.models.forecast import (
    ForecastArtifact,
    build_forecast_sample,
    calculate_forecast_metrics,
    forecast_next_steps,
    load_model_config,
    sarima_order_from_config,
    save_forecast_artifact,
    seasonal_order_from_config,
    train_sarima_model,
)


def main() -> None:
    """Main function to execute the model training, forecasting, and artifact saving process."""
    total_started_at = perf_counter()
    data_started_at = perf_counter()

    print("[1/4] Loading config and preparing KPI series")
    config = load_model_config()
    forecasting_config = config["forecasting"]
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

    order = sarima_order_from_config(config)
    seasonal_order = seasonal_order_from_config(config)

    print(f"- data preparation seconds: {perf_counter() - data_started_at:.2f}")

    training_started_at = perf_counter()
    print("[2/4] Training SARIMA baseline")
    print(f"- location_id: {location_id}")
    print(f"- kpi_name: {kpi_name}")
    print(f"- train rows: {len(split.train):,}")
    print(f"- test rows: {len(split.test):,}")
    print(f"- order: {order}")
    print(f"- seasonal order: {seasonal_order}")

    fitted_model = train_sarima_model(
        train_values=split.train["kpi_value"],
        order=order,
        seasonal_order=seasonal_order,
    )
    print(f"- training seconds: {perf_counter() - training_started_at:.2f}")

    forecast_started_at = perf_counter()
    print("[3/4] Forecasting and calculating metrics")
    forecast = forecast_next_steps(fitted_model, steps=len(split.test))
    metrics = calculate_forecast_metrics(
        actual=split.test["kpi_value"],
        forecast=forecast,
    )
    forecast_sample = build_forecast_sample(split.test, forecast)
    print(f"- forecasting seconds: {perf_counter() - forecast_started_at:.2f}")

    artifact = ForecastArtifact(
        model_name=str(model_config["name"]),
        model_version=str(model_config["version"]),
        location_id=location_id,
        kpi_name=kpi_name,
        train_rows=len(split.train),
        test_rows=len(split.test),
        forecast_horizon_steps=len(split.test),
        metrics=metrics,
        forecast_sample=forecast_sample,
    )
    artifact_path = Path("artifacts/models") / f"sarima_baseline_{location_id}_{kpi_name}.json"

    artifact_started_at = perf_counter()
    print("[4/4] Saving forecast artifact")
    save_forecast_artifact(artifact, artifact_path)
    print(f"- artifact seconds: {perf_counter() - artifact_started_at:.2f}")

    print("Forecast metrics")
    print(f"- MAE: {metrics.mae:.4f}")
    print(f"- RMSE: {metrics.rmse:.4f}")
    print(f"- MAPE: {metrics.mape:.4f}")
    print(f"- Saved artifact: {artifact_path}")
    print(f"- total seconds: {perf_counter() - total_started_at:.2f}")


if __name__ == "__main__":
    main()
