CREATE TABLE IF NOT EXISTS kpi_records (
    id BIGSERIAL PRIMARY KEY,
    location_id TEXT NOT NULL,
    kpi_name TEXT NOT NULL,
    timestamp_index BIGINT NOT NULL CHECK (timestamp_index >= 0),
    kpi_value DOUBLE PRECISION NOT NULL CHECK (kpi_value >= 0 AND kpi_value <= 1000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (location_id, kpi_name, timestamp_index)
);

CREATE INDEX IF NOT EXISTS idx_kpi_records_series_time
ON kpi_records (location_id, kpi_name, timestamp_index);

CREATE INDEX IF NOT EXISTS idx_kpi_records_kpi_time
ON kpi_records (kpi_name, timestamp_index);

CREATE TABLE IF NOT EXISTS model_runs (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    location_id TEXT NOT NULL,
    kpi_name TEXT NOT NULL,
    train_rows INTEGER NOT NULL CHECK (train_rows >= 0),
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    artifact_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_runs_series
ON model_runs (location_id, kpi_name);

CREATE TABLE IF NOT EXISTS anomaly_events (
    id BIGSERIAL PRIMARY KEY,
    location_id TEXT NOT NULL,
    kpi_name TEXT NOT NULL,
    timestamp_index BIGINT NOT NULL CHECK (timestamp_index >= 0),
    actual_value DOUBLE PRECISION NOT NULL,
    forecast_value DOUBLE PRECISION NOT NULL,
    residual DOUBLE PRECISION NOT NULL,
    threshold_value DOUBLE PRECISION NOT NULL CHECK (threshold_value >= 0),
    severity TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (location_id, kpi_name, timestamp_index, model_version)
);

CREATE INDEX IF NOT EXISTS idx_anomaly_events_series_time
ON anomaly_events (location_id, kpi_name, timestamp_index);

CREATE INDEX IF NOT EXISTS idx_anomaly_events_severity
ON anomaly_events (severity);
