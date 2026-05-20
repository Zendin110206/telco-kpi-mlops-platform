from __future__ import annotations

from telco_kpi_mlops.data.loader import load_canonical_dataset


def main() -> None:
    result = load_canonical_dataset()
    records = result.records

    print("Canonical dataset preview:")
    print(f"- rows: {len(records):,}")
    print(f"- columns: {list(records.columns)}")
    print(f"- KPI names: {sorted(records['kpi_name'].unique())}")
    print(f"- KPI series: {records[['location_id', 'kpi_name']].drop_duplicates().shape[0]}")
    print(f"- timestamp min: {records['timestamp_index'].min()}")
    print(f"- timestamp max: {records['timestamp_index'].max()}")
    print(f"- KPI value min: {records['kpi_value'].min()}")
    print(f"- KPI value max: {records['kpi_value'].max()}")
    print(f"- missing metadata files skipped: {result.missing_series_files}")
    print()
    print(records.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
