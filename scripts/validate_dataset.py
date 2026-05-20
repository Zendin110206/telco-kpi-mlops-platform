from __future__ import annotations

from telco_kpi_mlops.data.loader import load_canonical_dataset, load_data_contract
from telco_kpi_mlops.data.validator import validate_canonical_dataset


def main() -> None:
    contract = load_data_contract()
    load_result = load_canonical_dataset()
    report = validate_canonical_dataset(
        records=load_result.records,
        contract=contract,
        missing_series_files=load_result.missing_series_files,
    )

    print("Dataset validation passed")
    print(f"- rows checked: {report.rows_checked:,}")
    print(f"- KPI series checked: {report.series_checked}")
    print(f"- missing metadata files accepted: {report.missing_series_files}")
    print("- checks passed:")
    for check_name in report.checks_passed:
        print(f"  - {check_name}")


if __name__ == "__main__":
    main()
