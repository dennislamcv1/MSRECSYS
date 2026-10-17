# Arcturus Feature Store — Delta Lake Schema Reference

## Table: `feature_baseline_snapshot`

The drift monitoring loop reads this table as the reference distribution against which live traffic is compared. In this lab environment, this table is represented as a local Parquet file (`feature_baseline_snapshot.parquet`) rather than a live Delta table, since the lab environment does not provision a Spark/Databricks runtime with Delta Lake support. On a real Azure Databricks cluster, load it instead with:

```python
baseline_df = spark.read.format("delta").load("/mnt/arcturus/feature_baseline_snapshot")
```

| Column | Type | Description |
|---|---|---|
| `item_id` | string | Unique item identifier |
| `recency` | double | Normalized recency score, 0.0-1.0 |
| `freshness` | double | Normalized freshness score, 0.0-1.0 |
| `trending` | double | Normalized trending score, 0.0-1.0 |
| `ctr_7d` | double | 7-day rolling click-through rate, 0.0-1.0 |
| `watch_time_avg` | double | Average watch time, normalized 0.0-1.0 |
| `user_engagement_score` | double | Composite engagement score, 0.0-1.0 |

## PSI Threshold Configuration

The Population Stability Index (PSI) threshold used to gate automated retraining is defined in `arcturus_synthetic_load_config.json` as `psi_threshold` (0.25). A PSI score above this threshold on any single feature column indicates significant distributional drift and should trigger the retraining workflow.

Standard PSI interpretation bands (for reference, not enforced programmatically):
- PSI < 0.10: no significant shift
- 0.10 <= PSI < 0.25: moderate shift, monitor
- PSI >= 0.25: significant shift, retraining recommended

## Table: `feature_live_window` (conceptual)

The drift monitoring loop maintains a rolling in-memory window of the most recent `psi_rolling_window_size` served feature vectors (see config) rather than reading a second physical table. On a real deployment, this window could instead be backed by a streaming Delta table (e.g., a Delta Live Tables pipeline) if a persisted audit trail of every served feature vector is required.
