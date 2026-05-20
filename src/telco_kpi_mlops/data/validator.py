from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    """Summary returned after the canonical dataset passes validation."""

    rows_checked: int
    series_checked: int
    checks_passed: list[str]
    missing_series_files: list[str]


def validate_required_columns(records: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that all required columns are present in the DataFrame."""
    missing_columns = [column for column in required_columns if column not in records.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_no_missing_required_values(
    records: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Validate that there are no missing values in the required columns."""
    missing_counts = records[required_columns].isna().sum()
    invalid_counts = {
        column: int(count) for column, count in missing_counts.items() if int(count) > 0
    }

    if invalid_counts:
        raise ValueError(f"Missing required values: {invalid_counts}")


def validate_no_duplicate_records(records: pd.DataFrame, key_columns: list[str]) -> None:
    """Validate that there are no duplicate records based on the specified key columns."""
    duplicate_count = int(records.duplicated(subset=key_columns).sum())

    if duplicate_count > 0:
        raise ValueError(f"Duplicate canonical records found: {duplicate_count}")


def validate_kpi_names(records: pd.DataFrame, allowed_kpi_names: list[str]) -> None:
    """Validate that all KPI names in the records are among the allowed KPI names."""
    observed_kpi_names = set(records["kpi_name"].unique())
    allowed_names = set(allowed_kpi_names)
    invalid_names = sorted(observed_kpi_names - allowed_names)

    if invalid_names:
        raise ValueError(f"Invalid KPI names: {invalid_names}")


def validate_value_range(
    records: pd.DataFrame,
    min_value: float,
    max_value: float,
) -> None:
    """Validate that all KPI values are within the specified range [min_value, max_value]."""
    invalid_records = records[
        (records["kpi_value"] < min_value) | (records["kpi_value"] > max_value)
    ]

    if not invalid_records.empty:
        raise ValueError(
            f"KPI values outside expected range [{min_value}, {max_value}]: {len(invalid_records)}"
        )


def validate_timestamp_order(records: pd.DataFrame) -> None:
    """Validate that the timestamp_index is monotonically increasing for each series."""
    invalid_series: list[str] = []

    for (location_id, kpi_name), group in records.groupby(
        ["location_id", "kpi_name"],
        sort=False,
    ):
        if not group["timestamp_index"].is_monotonic_increasing:
            invalid_series.append(f"{location_id}:{kpi_name}")

    if invalid_series:
        raise ValueError(f"Timestamp order is not monotonic for series: {invalid_series[:10]}")


def validate_expected_frequency(
    records: pd.DataFrame,
    expected_frequency_seconds: int,
) -> None:
    """Validate that the timestamp_index has the expected frequency for each series."""
    invalid_series: list[str] = []

    for (location_id, kpi_name), group in records.groupby(
        ["location_id", "kpi_name"],
        sort=False,
    ):
        timestamp_diffs = group["timestamp_index"].diff().dropna()

        if not timestamp_diffs.empty and not (timestamp_diffs == expected_frequency_seconds).all():
            invalid_series.append(f"{location_id}:{kpi_name}")

    if invalid_series:
        raise ValueError(f"Unexpected timestamp frequency for series: {invalid_series[:10]}")


def validate_non_empty_series(records: pd.DataFrame) -> None:
    """Validate that there are no empty KPI series in the dataset."""
    series_sizes = records.groupby(["location_id", "kpi_name"], sort=False).size()

    if series_sizes.empty:
        raise ValueError("No KPI series found in canonical dataset.")


def validate_expected_missing_series_files(
    actual_missing_files: list[str],
    expected_missing_files: list[str],
) -> None:
    actual = sorted(actual_missing_files)
    expected = sorted(expected_missing_files)

    if actual != expected:
        raise ValueError(f"Unexpected missing series files. Expected {expected}, found {actual}.")


def validate_canonical_dataset(
    records: pd.DataFrame,
    contract: dict[str, Any],
    missing_series_files: list[str],
) -> ValidationReport:
    """Validate the canonical dataset against the provided contract and return a summary report."""
    required_columns = list(contract["canonical_schema"].keys())
    key_columns = [*contract["series_key"], "timestamp_index"]
    quality_rules = contract["quality_rules"]
    checks_passed: list[str] = []

    validate_required_columns(records, required_columns)
    checks_passed.append("required_columns")

    if not quality_rules["allow_missing_required_fields"]:
        validate_no_missing_required_values(records, required_columns)
        checks_passed.append("missing_required_values")

    if not quality_rules["allow_duplicate_records"]:
        validate_no_duplicate_records(records, key_columns)
        checks_passed.append("duplicate_records")

    validate_kpi_names(records, contract["allowed_kpi_names"])
    checks_passed.append("kpi_names")

    if not quality_rules["allow_negative_values"]:
        validate_value_range(
            records=records,
            min_value=contract["dataset"]["value_min"],
            max_value=quality_rules["max_allowed_value"],
        )
        checks_passed.append("value_range")

    if quality_rules["require_monotonic_time_per_series"]:
        validate_timestamp_order(records)
        checks_passed.append("timestamp_order")

    expected_frequency_seconds = contract["dataset"]["expected_frequency_minutes"] * 60
    validate_expected_frequency(records, expected_frequency_seconds)
    checks_passed.append("expected_frequency")

    if quality_rules["require_non_empty_series"]:
        validate_non_empty_series(records)
        checks_passed.append("non_empty_series")

    expected_missing_files = contract["raw_metadata_expectations"]["known_missing_series_files"]
    validate_expected_missing_series_files(missing_series_files, expected_missing_files)
    checks_passed.append("expected_missing_series_files")

    series_checked = records[["location_id", "kpi_name"]].drop_duplicates().shape[0]

    return ValidationReport(
        rows_checked=len(records),
        series_checked=series_checked,
        checks_passed=checks_passed,
        missing_series_files=missing_series_files,
    )
