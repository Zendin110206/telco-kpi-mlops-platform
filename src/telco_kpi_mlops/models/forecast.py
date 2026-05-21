from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

DEFAULT_MODEL_CONFIG_PATH = Path("configs/model_config.yaml")


@dataclass(frozen=True)
class ForecastMetrics:
    """Metrics for evaluating forecast performance. All values are floats."""

    mae: float
    rmse: float
    mape: float


@dataclass(frozen=True)
class ForecastArtifact:
    """Artifact containing forecast results and metadata for a specific KPI and location."""

    model_name: str
    model_version: str
    location_id: str
    kpi_name: str
    train_rows: int
    test_rows: int
    forecast_horizon_steps: int
    metrics: ForecastMetrics
    forecast_sample: list[dict[str, float | int]]


def load_model_config(path: Path = DEFAULT_MODEL_CONFIG_PATH) -> dict[str, Any]:
    """Load model configuration from a YAML file."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Model config must be a YAML mapping.")

    return config


def sarima_order_from_config(config: dict[str, Any]) -> tuple[int, int, int]:
    """Extract SARIMA order (p, d, q) from the model configuration."""
    order = config["forecasting"]["sarima_order"]
    return int(order["p"]), int(order["d"]), int(order["q"])


def seasonal_order_from_config(config: dict[str, Any]) -> tuple[int, int, int, int]:
    """Extract SARIMA seasonal order (P, D, Q, m) from the model configuration."""
    order = config["forecasting"]["seasonal_order"]
    seasonal_period = int(config["forecasting"]["seasonal_period_steps"])
    return int(order["p"]), int(order["d"]), int(order["q"]), seasonal_period


def train_sarima_model(
    train_values: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> SARIMAXResultsWrapper:
    """Train a SARIMA model using the provided training values and model orders."""
    model = SARIMAX(
        train_values.astype("float64"),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=True,
        concentrate_scale=True,
    )
    return model.fit(disp=False, low_memory=True)


def forecast_next_steps(
    fitted_model: SARIMAXResultsWrapper,
    steps: int,
) -> np.ndarray:
    """Generate forecasts for the next specified number of steps using the fitted SARIMA model."""
    if steps <= 0:
        raise ValueError("steps must be greater than zero.")

    forecast = fitted_model.forecast(steps=steps)
    return np.asarray(forecast, dtype="float64")


def calculate_forecast_metrics(
    actual: pd.Series,
    forecast: np.ndarray,
) -> ForecastMetrics:
    """
    Calculate forecast performance metrics (MAE, RMSE, MAPE)
    comparing actual values to forecasted values.
    """
    actual_values = actual.astype("float64").to_numpy()
    forecast_values = np.asarray(forecast, dtype="float64")
    errors = actual_values - forecast_values
    non_zero_mask = actual_values != 0

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mape = float(np.mean(np.abs(errors[non_zero_mask] / actual_values[non_zero_mask])) * 100)

    return ForecastMetrics(mae=mae, rmse=rmse, mape=mape)


def build_forecast_sample(
    test_frame: pd.DataFrame,
    forecast: np.ndarray,
    sample_size: int = 10,
) -> list[dict[str, float | int]]:
    """
    Build a sample of forecast results for the first few rows of the test frame,
    including actual values, forecasted values, and errors.
    """
    sample_frame = test_frame.head(sample_size).copy()
    sample_forecast = forecast[: len(sample_frame)]
    sample_frame["forecast_value"] = sample_forecast
    sample_frame["error"] = sample_frame["kpi_value"] - sample_frame["forecast_value"]

    return [
        {
            "timestamp_index": int(row.timestamp_index),
            "actual_value": float(row.kpi_value),
            "forecast_value": float(row.forecast_value),
            "error": float(row.error),
        }
        for row in sample_frame.itertuples(index=False)
    ]


def save_forecast_artifact(
    artifact: ForecastArtifact,
    artifact_path: Path,
) -> None:
    """Save the forecast artifact as a JSON file at the specified path."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dict = asdict(artifact)

    artifact_path.write_text(
        json.dumps(artifact_dict, indent=2),
        encoding="utf-8",
    )
