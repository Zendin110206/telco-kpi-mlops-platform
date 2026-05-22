from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
DEFAULT_DATA_CONTRACT_PATH = Path("configs/data_contract.yaml")


@dataclass(frozen=True)
class ModelMetadata:
    """
    Data class representing the metadata of a forecasting model,
    including details about the model, dataset, and forecasting configuration.
    """

    model_name: str
    model_version: str
    task_type: str
    dataset: str
    dataset_source_url: str
    baseline_model: str
    forecast_horizon: int
    seasonal_period: int
    target_use_cases: list[str]


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")

    return data


def build_model_metadata(
    model_config_path: Path = DEFAULT_MODEL_CONFIG_PATH,
    data_contract_path: Path = DEFAULT_DATA_CONTRACT_PATH,
) -> ModelMetadata:
    """
    Build the ModelMetadata object by loading and combining information
    from the model configuration and data contract YAML files.
    """
    model_config = load_yaml_mapping(model_config_path)
    data_contract = load_yaml_mapping(data_contract_path)

    model = model_config["model"]
    forecasting = model_config["forecasting"]
    dataset = data_contract["dataset"]

    return ModelMetadata(
        model_name=str(model["name"]),
        model_version=str(model["version"]),
        task_type=str(model["task_type"]),
        dataset=str(dataset["name"]),
        dataset_source_url=str(dataset["source_url"]),
        baseline_model=str(forecasting["baseline_model"]),
        forecast_horizon=int(forecasting["forecast_horizon_steps"]),
        seasonal_period=int(forecasting["seasonal_period_steps"]),
        target_use_cases=[str(use_case) for use_case in data_contract["target_use_cases"]],
    )


def model_metadata_as_dict(metadata: ModelMetadata) -> dict[str, object]:
    """
    Convert the ModelMetadata dataclass instance
    into a dictionary for easy serialization or API responses.
    """
    return asdict(metadata)
