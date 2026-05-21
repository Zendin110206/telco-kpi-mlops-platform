from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANOMALY_EVENT_COLUMNS = [
    "location_id",
    "kpi_name",
    "timestamp_index",
    "actual_value",
    "forecast_value",
    "residual",
    "threshold_value",
    "severity",
    "model_version",
]

SEVERITY_LEVELS = ("normal", "low", "medium", "high", "critical")


@dataclass(frozen=True)
class AnomalySummary:
    """Summary of anomaly detection results."""

    scored_rows: int
    anomaly_rows: int
    severity_counts: dict[str, int]


def calculate_residuals(
    test_frame: pd.DataFrame,
    forecast: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate residuals and absolute residuals by comparing actual KPI values in the test frame
    to the forecasted values.
    """
    required_columns = {"location_id", "kpi_name", "timestamp_index", "kpi_value"}
    missing_columns = required_columns.difference(test_frame.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"test_frame is missing required columns: {missing}")

    forecast_values = np.asarray(forecast, dtype="float64")

    if len(test_frame) != len(forecast_values):
        raise ValueError("test_frame and forecast must have the same number of rows.")

    residual_frame = test_frame[["location_id", "kpi_name", "timestamp_index", "kpi_value"]].copy()
    residual_frame = residual_frame.rename(columns={"kpi_value": "actual_value"})
    residual_frame["forecast_value"] = forecast_values
    residual_frame["residual"] = residual_frame["actual_value"] - residual_frame["forecast_value"]
    residual_frame["absolute_residual"] = residual_frame["residual"].abs()

    return residual_frame


def calculate_dynamic_threshold(
    residual_frame: pd.DataFrame,
    window_size: int,
    threshold_multiplier: float,
) -> pd.DataFrame:
    """Calculate dynamic threshold for anomaly detection."""
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    if threshold_multiplier <= 0:
        raise ValueError("threshold_multiplier must be greater than zero.")

    if "absolute_residual" not in residual_frame.columns:
        raise ValueError("residual_frame must include an absolute_residual column.")

    scored_frame = residual_frame.copy()
    previous_residuals = scored_frame["absolute_residual"].shift(1)
    rolling_mean = previous_residuals.rolling(
        window=window_size,
        min_periods=window_size,
    ).mean()
    rolling_std = previous_residuals.rolling(
        window=window_size,
        min_periods=window_size,
    ).std(ddof=0)

    scored_frame["threshold_value"] = rolling_mean + threshold_multiplier * rolling_std

    return scored_frame


def assign_anomaly_flag(scored_frame: pd.DataFrame) -> pd.DataFrame:
    """Assign anomaly flags based on whether the absolute residual exceeds the dynamic threshold."""
    required_columns = {"absolute_residual", "threshold_value"}
    missing_columns = required_columns.difference(scored_frame.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"scored_frame is missing required columns: {missing}")

    result = scored_frame.copy()
    has_threshold = result["threshold_value"].notna()
    result["is_anomaly"] = has_threshold & (result["absolute_residual"] > result["threshold_value"])

    return result


def severity_from_ratio(
    is_anomaly: bool,
    absolute_residual: float,
    threshold_value: float,
) -> str:
    """Determine severity level based on the ratio of absolute residual to threshold value."""
    if not is_anomaly:
        return "normal"

    if threshold_value <= 0:
        return "critical"

    ratio = absolute_residual / threshold_value

    if ratio < 1.25:
        return "low"
    if ratio < 1.75:
        return "medium"
    if ratio < 2.50:
        return "high"
    return "critical"


def assign_severity(scored_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Assign severity levels to anomalies based on the ratio of
    absolute residual to threshold value.
    """
    required_columns = {"is_anomaly", "absolute_residual", "threshold_value"}
    missing_columns = required_columns.difference(scored_frame.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"scored_frame is missing required columns: {missing}")

    result = scored_frame.copy()
    result["severity"] = [
        severity_from_ratio(
            is_anomaly=bool(row.is_anomaly),
            absolute_residual=float(row.absolute_residual),
            threshold_value=float(row.threshold_value),
        )
        for row in result.itertuples(index=False)
    ]

    return result


def select_anomaly_events(
    scored_frame: pd.DataFrame,
    model_version: str,
) -> pd.DataFrame:
    """Select rows flagged as anomalies and include relevant information for each anomaly event."""
    if "is_anomaly" not in scored_frame.columns:
        raise ValueError("scored_frame must include an is_anomaly column.")

    anomaly_events = scored_frame.loc[scored_frame["is_anomaly"]].copy()

    if anomaly_events.empty:
        return pd.DataFrame(columns=ANOMALY_EVENT_COLUMNS)

    anomaly_events["model_version"] = model_version

    return anomaly_events[ANOMALY_EVENT_COLUMNS]


def summarize_anomalies(scored_frame: pd.DataFrame) -> AnomalySummary:
    """
    Summarize anomaly detection results, including counts of
    scored rows, anomaly rows, and severity levels.
    """
    if "is_anomaly" not in scored_frame.columns:
        raise ValueError("scored_frame must include an is_anomaly column.")

    if "severity" not in scored_frame.columns:
        raise ValueError("scored_frame must include a severity column.")

    raw_counts = scored_frame["severity"].value_counts().to_dict()
    severity_counts = {severity: int(raw_counts.get(severity, 0)) for severity in SEVERITY_LEVELS}

    return AnomalySummary(
        scored_rows=len(scored_frame),
        anomaly_rows=int(scored_frame["is_anomaly"].sum()),
        severity_counts=severity_counts,
    )
