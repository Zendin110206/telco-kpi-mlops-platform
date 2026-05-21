from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

DEFAULT_DATABASE_URL = "postgresql+psycopg://telco:telco@localhost:5432/telco_kpi_mlops"
KPI_RECORD_COLUMNS = [
    "location_id",
    "kpi_name",
    "timestamp_index",
    "kpi_value",
]


@dataclass(frozen=True)
class LoadSummary:
    """Summary returned after loading canonical KPI records into PostgreSQL."""

    rows_received: int
    inserted_rows: int
    skipped_rows: int
    total_rows_after: int


def get_database_url() -> str:
    """Get the database URL from the environment variable or return the default."""
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for the PostgreSQL database."""
    return create_engine(database_url or get_database_url())


def iter_record_batches(
    records: pd.DataFrame,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    """Yield batches of KPI records as lists of dictionaries."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    selected_records = records[KPI_RECORD_COLUMNS]

    for start in range(0, len(selected_records), batch_size):
        batch = selected_records.iloc[start : start + batch_size]
        yield batch.to_dict(orient="records")


def count_kpi_records(engine: Engine) -> int:
    """Count the total number of KPI records in the database."""
    with engine.connect() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM kpi_records"))
        return int(result.scalar_one())


def load_kpi_records(
    engine: Engine,
    records: pd.DataFrame,
    batch_size: int = 10_000,
) -> LoadSummary:
    """Load canonical KPI records into the PostgreSQL database with conflict handling."""
    metadata = MetaData()
    kpi_records = Table("kpi_records", metadata, autoload_with=engine)
    inserted_rows = 0

    with engine.begin() as connection:
        for batch in iter_record_batches(records, batch_size=batch_size):
            statement = insert(kpi_records).values(batch)
            statement = statement.on_conflict_do_nothing(
                index_elements=["location_id", "kpi_name", "timestamp_index"]
            )
            result = connection.execute(statement)
            inserted_rows += result.rowcount or 0

        total_rows_after = int(
            connection.execute(text("SELECT COUNT(*) FROM kpi_records")).scalar_one()
        )

    return LoadSummary(
        rows_received=len(records),
        inserted_rows=inserted_rows,
        skipped_rows=len(records) - inserted_rows,
        total_rows_after=total_rows_after,
    )


def get_distinct_locations(engine: Engine) -> list[str]:
    """Get a list of distinct location IDs from the KPI records in the database."""
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT DISTINCT location_id FROM kpi_records ORDER BY location_id")
        )
        return [str(row[0]) for row in result]


def get_distinct_kpi_names(engine: Engine) -> list[str]:
    """Get a list of distinct KPI names from the KPI records in the database."""
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT DISTINCT kpi_name FROM kpi_records ORDER BY kpi_name")
        )
        return [str(row[0]) for row in result]


def read_kpi_series(
    engine: Engine,
    location_id: str,
    kpi_name: str,
    limit: int = 10,
) -> pd.DataFrame:
    """Read a time series of KPI values for a specific location and KPI name."""
    query = text(
        """
        SELECT location_id, kpi_name, timestamp_index, kpi_value
        FROM kpi_records
        WHERE location_id = :location_id
          AND kpi_name = :kpi_name
        ORDER BY timestamp_index
        LIMIT :limit
        """
    )
    return pd.read_sql_query(
        query,
        con=engine,
        params={
            "location_id": location_id,
            "kpi_name": kpi_name,
            "limit": limit,
        },
    )
