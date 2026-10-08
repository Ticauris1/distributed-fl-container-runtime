"""Aggregate completed replicated runs into mean/std comparison tables."""

from pathlib import Path

import pandas as pd


GROUP_FIELDS = [
    "experiment", "source", "model_name", "clients", "distribution",
    "participation", "method", "comparison_variant", "client_weight_method",
    "privacy_agg_enabled",
]
METRIC_FIELDS = [
    "accuracy", "precision", "recall", "f1_score", "auc_score",
    "avg_time_per_sample", "client_weighting_total_seconds",
    "privacy_agg_total_seconds", "privacy_agg_max_abs_error",
]


def write_multiseed_summary(summary_csv, output_path=None):
    """Write one row per condition with replicate count and mean/std metrics."""
    summary_csv = Path(summary_csv)
    output_path = Path(output_path or summary_csv.with_name("_multiseed_summary.csv"))
    frame = pd.read_csv(summary_csv)
    if "experiment_seed" not in frame:
        return None
    frame["experiment_seed"] = pd.to_numeric(frame["experiment_seed"], errors="coerce")
    frame = frame.dropna(subset=["experiment_seed"])
    if frame.empty:
        return None
    available_groups = [field for field in GROUP_FIELDS if field in frame]
    available_metrics = [field for field in METRIC_FIELDS if field in frame]
    for field in available_metrics:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    aggregation = {field: ["mean", "std"] for field in available_metrics}
    aggregation["experiment_seed"] = "nunique"
    result = frame.groupby(available_groups, dropna=False).agg(aggregation).reset_index()
    result.columns = [
        column if isinstance(column, str) else
        ("seed_count" if column == ("experiment_seed", "nunique") else
         "_".join(part for part in column if part))
        for column in result.columns
    ]
    result.to_csv(output_path, index=False)
    return output_path
