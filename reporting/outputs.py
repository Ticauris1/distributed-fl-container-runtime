"""CSV, metadata, prediction, and checkpoint persistence."""

import csv
import json
from pathlib import Path
import pandas as pd #type: ignore
import torch #type: ignore


SUMMARY_FIELDS = [
    "experiment", "experiment_seed", "comparison_variant", "source", "model_name",
    "clients", "distribution", "participation",
    "method", "eval_mode", "lightweight_model_name", "heavy_model_name",
    "hybrid_threshold_low", "hybrid_threshold_high", "hybrid_lightweight_count",
    "hybrid_heavy_count", "hybrid_heavy_fraction", "avg_time_per_sample",
    "client_weight_method", "client_weighting_total_seconds",
    "privacy_agg_enabled", "privacy_agg_target", "privacy_agg_status",
    "privacy_agg_rounds", "privacy_agg_failures", "privacy_agg_fallback_used",
    "privacy_agg_total_seconds", "privacy_agg_mean_seconds",
    "privacy_agg_max_abs_error", "privacy_agg_mean_abs_error",
    "privacy_agg_plaintext_bytes", "privacy_agg_protected_bytes",
    "privacy_agg_size_expansion_ratio", "accuracy", "precision", "recall",
    "f1_score", "auc_score", "save_dir",
]
CLIENT_SUMMARY_FIELDS = [
    "experiment", "experiment_seed", "source", "model_name", "client_id",
    "accuracy", "precision",
    "recall", "f1_score", "auc_score",
]


def initialize_csv(path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fieldnames).writeheader()
    return path


def append_csv_row(path, row, fieldnames):
    with Path(path).open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def initialize_summary_csv(path):
    path = Path(path)
    if not path.exists():
        return initialize_csv(path, SUMMARY_FIELDS)

    with path.open("r", newline="") as stream:
        reader = csv.DictReader(stream)
        existing_fields = reader.fieldnames or []
        if existing_fields == SUMMARY_FIELDS:
            return path
        existing_rows = list(reader)

    temporary_path = path.with_name(f".{path.name}.schema.tmp")
    initialize_csv(temporary_path, SUMMARY_FIELDS)
    for row in existing_rows:
        append_csv_row(temporary_path, row, SUMMARY_FIELDS)
    temporary_path.replace(path)
    return path


def initialize_client_summary_csv(path):
    return initialize_csv(path, CLIENT_SUMMARY_FIELDS)


def append_summary_row(path, row):
    append_csv_row(path, row, SUMMARY_FIELDS)


def append_client_summary_row(path, row):
    append_csv_row(path, row, CLIENT_SUMMARY_FIELDS)


def save_metadata(directory, payload):
    path = Path(directory) / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2)
    return path


def save_predictions(path, y_true, y_pred):
    pd.DataFrame({"y_true": list(y_true), "y_pred": list(y_pred)}).to_csv(path, index=False)
    return Path(path)


def save_model_checkpoint(path, model):
    torch.save(model.state_dict(), path)
    return Path(path)
