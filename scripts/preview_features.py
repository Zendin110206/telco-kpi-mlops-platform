from __future__ import annotations

from telco_kpi_mlops.data.repository import (
    create_database_engine,
    get_first_kpi_series_key,
    read_kpi_series,
)
from telco_kpi_mlops.features.transforms import (
    ensure_regular_frequency,
    fill_missing_points,
)
from telco_kpi_mlops.features.windows import (
    create_forecast_horizon,
    split_train_test_by_time,
)


def main() -> None:
    engine = create_database_engine()
    location_id, kpi_name = get_first_kpi_series_key(engine)
    series = read_kpi_series(
        engine=engine,
        location_id=location_id,
        kpi_name=kpi_name,
        limit=288,
    )
    regular_series = ensure_regular_frequency(series, frequency_seconds=300)
    filled_series = fill_missing_points(regular_series)
    split = split_train_test_by_time(filled_series, train_ratio=0.8)
    horizon = create_forecast_horizon(
        last_timestamp_index=int(filled_series["timestamp_index"].max()),
        horizon_steps=12,
        frequency_seconds=300,
    )

    print("Feature preparation preview")
    print(f"- series: {location_id} / {kpi_name}")
    print(f"- input rows: {len(series)}")
    print(f"- regular rows: {len(regular_series)}")
    print(f"- missing values after fill: {int(filled_series['kpi_value'].isna().sum())}")
    print(f"- train rows: {len(split.train)}")
    print(f"- test rows: {len(split.test)}")
    print(f"- forecast horizon steps: {len(horizon)}")
    print(f"- forecast horizon start: {horizon[0]}")
    print(f"- forecast horizon end: {horizon[-1]}")


if __name__ == "__main__":
    main()
