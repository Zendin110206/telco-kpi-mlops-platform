# Model Card

## Model Name

telco_kpi_sarima_residual_anomaly_detector

## Version

0.1.0

## Intended Use

This model forecasts telecom KPI time-series values and flags unusual residual patterns as anomaly candidates.

The current portfolio checkpoint focuses on one validated PostgreSQL-backed KPI series:

- Location: `r1`
- KPI: `aggregated_internet_traffic`

## Not Intended Use

- Not a replacement for production telecom monitoring systems.
- Not trained on private Indonesian operator production data.
- Not a certified NWDAF implementation.
- Not a cybersecurity IDS.
- Not a confirmed incident detector.

## Dataset

The project uses a public network operator KPI time-series dataset from Zenodo.

The dataset is anonymized and scaled. It is suitable for portfolio-scale engineering and modeling practice, but it should not be treated as private production telecom data.

## Modeling Approach

The current approach uses:

- SARIMA forecasting baseline.
- Chronological train/test split.
- Residual calculation from `actual_value - forecast_value`.
- Dynamic residual thresholding.
- Severity assignment for anomaly candidates.
- PostgreSQL persistence for anomaly events.

## Evaluation Summary

Evaluation snapshot from the current checkpoint:

| Metric                   |   Value |
| ------------------------ | ------: |
| Train rows               |     230 |
| Test rows                |      58 |
| MAE                      | 78.2955 |
| RMSE                     | 99.5583 |
| MAPE                     | 11.2903 |
| Anomaly events           |       2 |
| Anomaly rate             |   3.45% |
| Low severity events      |       2 |
| Medium severity events   |       0 |
| High severity events     |       0 |
| Critical severity events |       0 |

## Example Anomaly Candidates

| timestamp_index | actual_value | forecast_value | residual | threshold_value | severity |
| --------------: | -----------: | -------------: | -------: | --------------: | -------- |
|           73800 |       722.79 |         658.81 |    63.98 |           59.07 | low      |
|           84000 |       639.47 |         782.05 |  -142.58 |          139.40 | low      |

## Limitations

- Evaluation currently focuses on one KPI series for the current checkpoint.
- Anomaly labels are derived from residual thresholds, not manually verified incident labels.
- Detected events should be described as anomaly candidates.
- The dataset is anonymized and scaled.
- The model is a baseline, not the final optimized model.
- The project is not a certified telecom production system.

## Reproducibility

Run the forecasting baseline:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py
```

Run residual anomaly scoring:

```powershell
.\.venv\Scripts\python.exe scripts\batch_score.py
```

Run model evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_model.py
```

The evaluation script writes a local report to:

```text
artifacts/reports/evaluation_summary.json
```

Runtime artifacts are intentionally ignored by Git.
