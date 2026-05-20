from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path("configs/data_contract.yaml")
DEFAULT_DATASET_ROOT = (
    Path("data/raw/network_operator_kpis") / "network_operator_KPIs_time_series_dataset"
)
CANONICAL_COLUMNS = [
    "timestamp_index",
    "location_id",
    "kpi_name",
    "kpi_value",
]


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved paths for the raw Networks Operator KPIs dataset."""

    root: Path = DEFAULT_DATASET_ROOT

    @property
    def data_real_dir(self) -> Path:
        return self.root / "data_real"

    @property
    def data_series_dir(self) -> Path:
        return self.root / "data_series"

    @property
    def data_real_info_path(self) -> Path:
        return self.root / "data_real_info.txt"

    @property
    def data_series_info_path(self) -> Path:
        return self.root / "data_series_info.txt"


@dataclass(frozen=True)
class LoadResult:
    """Canonical dataset records plus raw metadata issues found while loading."""

    records: pd.DataFrame
    missing_series_files: list[str]


def load_data_contract(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the data contract from a YAML file, ensuring it is a mapping."""
    with path.open("r", encoding="utf-8") as file:
        contract = yaml.safe_load(file)

    if not isinstance(contract, dict):
        raise ValueError("Data contract must be a YAML mapping.")

    return contract


def read_info_file(path: Path) -> dict[str, str]:
    """Read a data info file and return a mapping of series_id to raw_kpi_label."""
    mapping: dict[str, str] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        series_id, raw_kpi_label = line.split()
        mapping[series_id] = raw_kpi_label

    return mapping


def read_series_file(path: Path) -> pd.DataFrame:
    """Read a series file into a DataFrame with canonical columns."""
    frame = pd.read_csv(
        path,
        sep=r"\s+",  # Split on any whitespace
        names=["timestamp_index", "kpi_value"],
        engine="python",  # Use Python engine for regex separator
    )
    frame["timestamp_index"] = frame["timestamp_index"].astype("int64")
    frame["kpi_value"] = frame["kpi_value"].astype("float64")
    return frame


def series_sort_key(series_id: str) -> tuple[str, int]:
    return series_id[0], int(series_id[1:])


def load_series_group(
    series_dir: Path,
    info_mapping: dict[str, str],
    raw_kpi_label_mapping: dict[str, str],
) -> tuple[list[pd.DataFrame], list[str]]:
    """Load all series files for a group, returning canonical DataFrames and missing file IDs."""
    frames: list[pd.DataFrame] = []
    missing_series_files: list[str] = []

    for series_id, raw_kpi_label in sorted(
        info_mapping.items(), key=lambda item: series_sort_key(item[0])
    ):
        series_path = series_dir / f"{series_id}.txt"

        if not series_path.exists():
            missing_series_files.append(series_id)
            continue

        if raw_kpi_label not in raw_kpi_label_mapping:
            raise KeyError(f"Raw KPI label is not mapped in data contract: {raw_kpi_label}")

        frame = read_series_file(series_path)
        frame["location_id"] = series_id
        frame["kpi_name"] = raw_kpi_label_mapping[raw_kpi_label]
        frames.append(frame[CANONICAL_COLUMNS])

    return frames, missing_series_files


def load_canonical_dataset(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> LoadResult:
    paths = DatasetPaths(root=dataset_root)

    if not paths.root.exists():
        raise FileNotFoundError(
            "Dataset folder was not found. Run scripts/download_dataset.py first."
        )

    contract = load_data_contract(config_path)
    raw_kpi_label_mapping = contract["raw_kpi_label_mapping"]

    real_info = read_info_file(paths.data_real_info_path)
    series_info = read_info_file(paths.data_series_info_path)

    real_frames, missing_real_files = load_series_group(
        series_dir=paths.data_real_dir,
        info_mapping=real_info,
        raw_kpi_label_mapping=raw_kpi_label_mapping,
    )
    series_frames, missing_series_files = load_series_group(
        series_dir=paths.data_series_dir,
        info_mapping=series_info,
        raw_kpi_label_mapping=raw_kpi_label_mapping,
    )

    records = pd.concat([*real_frames, *series_frames], ignore_index=True)

    return LoadResult(
        records=records,
        missing_series_files=[*missing_real_files, *missing_series_files],
    )
