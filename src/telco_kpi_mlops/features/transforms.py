from __future__ import annotations

import pandas as pd

from telco_kpi_mlops.features.windows import sort_time_series


def ensure_regular_frequency(
    series: pd.DataFrame,
    frequency_seconds: int = 300,
) -> pd.DataFrame:
    """
    Ensure that a time series has regular timestamps at the specified frequency by reindexing
    and filling missing points.
    """
    if series.empty:
        raise ValueError("Cannot regularize an empty time series.")

    if frequency_seconds <= 0:
        raise ValueError("frequency_seconds must be greater than zero.")

    sorted_series = sort_time_series(series)
    location_ids = sorted_series["location_id"].unique()
    kpi_names = sorted_series["kpi_name"].unique()

    if len(location_ids) != 1 or len(kpi_names) != 1:
        raise ValueError("ensure_regular_frequency expects exactly one KPI series.")

    start_timestamp = int(sorted_series["timestamp_index"].min())
    end_timestamp = int(sorted_series["timestamp_index"].max())
    regular_index = range(start_timestamp, end_timestamp + frequency_seconds, frequency_seconds)

    regular_series = (
        sorted_series.set_index("timestamp_index")
        .reindex(regular_index)
        .rename_axis("timestamp_index")
        .reset_index()
    )
    regular_series["location_id"] = str(location_ids[0])
    regular_series["kpi_name"] = str(kpi_names[0])
    return regular_series[["location_id", "kpi_name", "timestamp_index", "kpi_value"]]


def fill_missing_points(
    series: pd.DataFrame,
    value_column: str = "kpi_value",
) -> pd.DataFrame:
    """Fill missing KPI values in a time series using linear interpolation, ffill, and bfill."""
    filled_series = sort_time_series(series).copy()
    filled_series[value_column] = (
        filled_series[value_column].interpolate(method="linear").ffill().bfill()
    )
    return filled_series


def scale_optional(
    series: pd.DataFrame,
    value_column: str = "kpi_value",
    enabled: bool = False,
) -> pd.DataFrame:
    """
    Optionally scale KPI values to the [0, 1] range using min-max scaling.
    If disabled, returns the original series.
    """
    scaled_series = series.copy()

    if not enabled:
        return scaled_series

    value_min = scaled_series[value_column].min()
    value_max = scaled_series[value_column].max()

    if value_max == value_min:
        scaled_series[value_column] = 0.0
        return scaled_series

    scaled_series[value_column] = (scaled_series[value_column] - value_min) / (
        value_max - value_min
    )
    return scaled_series
