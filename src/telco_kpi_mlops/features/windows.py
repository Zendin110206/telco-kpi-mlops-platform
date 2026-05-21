from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplit:
    """Train and test split for one chronological time series."""

    train: pd.DataFrame
    test: pd.DataFrame


def sort_time_series(
    series: pd.DataFrame,
    timestamp_column: str = "timestamp_index",
) -> pd.DataFrame:
    """Sort a time series DataFrame by the specified timestamp column."""
    return series.sort_values(timestamp_column).reset_index(drop=True)


def split_train_test_by_time(
    series: pd.DataFrame,
    train_ratio: float = 0.8,
    min_train_rows: int = 2,
) -> TimeSeriesSplit:
    """
    Split a time series DataFrame into train and test sets based on chronological order and
    a specified train ratio.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    sorted_series = sort_time_series(series)
    split_index = int(len(sorted_series) * train_ratio)

    if split_index < min_train_rows:
        raise ValueError("Not enough rows for the requested minimum train size.")

    if split_index >= len(sorted_series):
        raise ValueError("Train/test split would produce an empty test set.")

    return TimeSeriesSplit(
        train=sorted_series.iloc[:split_index].reset_index(drop=True),
        test=sorted_series.iloc[split_index:].reset_index(drop=True),
    )


def create_forecast_horizon(
    last_timestamp_index: int,
    horizon_steps: int,
    frequency_seconds: int,
) -> list[int]:
    """
    Create a list of future timestamp indices for forecasting,
    based on the last known timestamp index,
    the number of steps in the forecast horizon, and the frequency of the data.
    """
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be greater than zero.")

    if frequency_seconds <= 0:
        raise ValueError("frequency_seconds must be greater than zero.")

    return [
        last_timestamp_index + (step * frequency_seconds) for step in range(1, horizon_steps + 1)
    ]
