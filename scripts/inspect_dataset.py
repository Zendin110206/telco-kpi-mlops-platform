from __future__ import annotations

from collections import Counter
from pathlib import Path

DATASET_ROOT = Path("data/raw/network_operator_kpis") / "network_operator_KPIs_time_series_dataset"
DATA_REAL_DIR = DATASET_ROOT / "data_real"
DATA_SERIES_DIR = DATASET_ROOT / "data_series"
DATA_REAL_INFO_PATH = DATASET_ROOT / "data_real_info.txt"
DATA_SERIES_INFO_PATH = DATASET_ROOT / "data_series_info.txt"
DATA_REAL_INCIDENTS_PATH = DATASET_ROOT / "data_real_incidents.txt"


def read_info_file(path: Path) -> dict[str, str]:
    """Read a dataset info file and return a mapping of series IDs to KPI labels."""
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        series_id, kpi_label = line.split()
        mapping[series_id] = kpi_label
    return mapping


def read_incidents(path: Path) -> list[tuple[str, int, int]]:
    """Read a dataset incidents file and return a list of incident tuples."""
    incidents: list[tuple[str, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        series_id, start_str, end_sample = line.split()
        incidents.append((series_id, int(start_str), int(end_sample)))
    return incidents


def summarize_series_file(path: Path) -> dict[str, float | int | str]:
    """Summarize a series file and return a dictionary with summary statistics."""
    timestamps: list[int] = []
    values: list[float] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        timestamp_second, value = line.split()
        timestamps.append(int(timestamp_second))
        values.append(float(value))

    return {
        "file": path.name,
        "rows": len(values),
        "timestamp_min": min(timestamps),
        "timestamp_max": max(timestamps),
        "value_min": min(values),
        "value_max": max(values),
    }


def print_mapping_summary(title: str, mapping: dict[str, str]) -> None:
    """Print a summary of the series ID to KPI label mapping."""
    print(title)
    print(f"- metadata rows: {len(mapping)}")
    print(f"- KPI counts: {dict(Counter(mapping.values()))}")


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            "Dataset folder was not found. Run scripts/download_dataset.py first."
        )

    real_files = sorted(DATA_REAL_DIR.glob("*.txt"), key=lambda path: int(path.stem[1:]))
    series_files = sorted(DATA_SERIES_DIR.glob("*.txt"), key=lambda path: int(path.stem[1:]))

    real_info = read_info_file(DATA_REAL_INFO_PATH)
    series_info = read_info_file(DATA_SERIES_INFO_PATH)
    incidents = read_incidents(DATA_REAL_INCIDENTS_PATH)

    real_file_ids = {path.stem for path in real_files}
    series_file_ids = {path.stem for path in series_files}

    print("Dataset inspection")
    print(f"- dataset root: {DATASET_ROOT}")
    print(f"- data_real files: {len(real_files)}")
    print(f"- data_series files: {len(series_files)}")
    print()

    print_mapping_summary("data_real_info.txt", real_info)
    print_mapping_summary("data_series_info.txt", series_info)
    print()

    print("Metadata vs file checks")
    print(f"- missing data_real files: {sorted(set(real_info) - real_file_ids)}")
    print(f"- extra data_real files: {sorted(real_file_ids - set(real_info))}")
    print(f"- missing data_series files: {sorted(set(series_info) - series_file_ids)}")
    print(f"- extra data_series files: {sorted(series_file_ids - set(series_info))}")
    print()

    print("Incident summary")
    print(f"- incident rows: {len(incidents)}")
    print(f"- incident counts by series: {dict(Counter(item[0] for item in incidents))}")
    print(f"- open-ended incidents: {[item for item in incidents if item[2] < 0]}")
    print()

    print("Sample file summaries")
    for path in [real_files[0], real_files[-1], series_files[0], series_files[-1]]:
        summary = summarize_series_file(path)
        print(f"- {summary}")


if __name__ == "__main__":
    main()
