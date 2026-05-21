from __future__ import annotations

from telco_kpi_mlops.data.loader import load_canonical_dataset, load_data_contract
from telco_kpi_mlops.data.repository import (
    count_kpi_records,
    create_database_engine,
    get_distinct_kpi_names,
    get_distinct_locations,
    load_kpi_records,
    read_kpi_series,
)
from telco_kpi_mlops.data.validator import validate_canonical_dataset


def main() -> None:
    contract = load_data_contract()
    load_result = load_canonical_dataset()
    validation_report = validate_canonical_dataset(
        records=load_result.records,
        contract=contract,
        missing_series_files=load_result.missing_series_files,
    )

    engine = create_database_engine()
    rows_before = count_kpi_records(engine)
    summary = load_kpi_records(engine, load_result.records)
    locations = get_distinct_locations(engine)
    kpi_names = get_distinct_kpi_names(engine)
    sample = read_kpi_series(
        engine=engine,
        location_id=locations[0],
        kpi_name=kpi_names[0],
        limit=5,
    )

    print("PostgreSQL load completed")
    print(f"- validation rows checked: {validation_report.rows_checked:,}")
    print(f"- rows before load: {rows_before:,}")
    print(f"- rows received: {summary.rows_received:,}")
    print(f"- inserted rows: {summary.inserted_rows:,}")
    print(f"- skipped duplicates: {summary.skipped_rows:,}")
    print(f"- rows after load: {summary.total_rows_after:,}")
    print(f"- distinct locations: {len(locations)}")
    print(f"- distinct KPI names: {kpi_names}")
    print()
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
