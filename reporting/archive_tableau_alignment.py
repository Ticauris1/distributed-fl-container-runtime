"""Align the preserved segmentation export with the current Tableau schema."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_TABLEAU = (
    PROJECT_ROOT
    / "results"
    / "archive"
    / "pre_matched_6encoder_2026-10-04"
    / "tableau"
)
SOURCE = ARCHIVE_TABLEAU / "segmentation_experiments.csv"
CATALOG = ARCHIVE_TABLEAU / "artifact_catalog.csv"
OUTPUT = ARCHIVE_TABLEAU / "archived_segmentation_experiments_aligned.csv"

LEADING_FIELDS = [
    "experiment_id",
    "result_status",
    "training_category",
    "training_type",
    "variant_name",
    "model",
    "encoder",
    "seed",
    "method",
    "weighting",
    "client_distribution",
    "clients",
    "rounds",
    "local_epochs",
    "configured_epochs",
    "epochs_trained",
    "total_local_epochs",
    "observed_training_budget",
    "image_size",
    "batch_size",
    "best_epoch",
    "best_round",
    "selected_threshold",
    "validation_dice",
    "test_loss",
    "training_seconds",
    "mean_round_seconds",
    "checkpoint_size_mb",
    "dice",
    "iou",
    "pixel_precision",
    "pixel_recall",
    "specificity",
    "mean_cracked_sample_dice",
    "empty_mask_accuracy",
    "metadata_status",
    "source_file",
]


def _read(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _number(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _checkpoint_sizes():
    sizes = {}
    if not CATALOG.is_file():
        return sizes
    for row in _read(CATALOG):
        if row.get("artifact_extension") != ".pth":
            continue
        source = row.get("source_file", "")
        run_dir = source.split("/checkpoints/", 1)[0]
        size = _number(row.get("artifact_bytes"))
        if run_dir and size is not None:
            sizes[run_dir] = max(size, sizes.get(run_dir, 0.0))
    return sizes


def build():
    rows = _read(SOURCE)
    checkpoint_sizes = _checkpoint_sizes()
    aligned = []
    all_fields = list(LEADING_FIELDS)
    seen = set(all_fields)

    for source_row in rows:
        row = dict(source_row)
        training_type = row.get("training_type", "").strip()
        training_category = "federated" if training_type == "federated" else "centralized"
        rounds = _number(row.get("rounds"))
        local_epochs = _number(row.get("local_epochs"))
        epochs_trained = _number(row.get("epochs_trained"))
        total_local_epochs = (
            rounds * local_epochs
            if rounds is not None and local_epochs is not None
            else ""
        )
        observed_budget = (
            total_local_epochs
            if training_category == "federated"
            else (epochs_trained if epochs_trained is not None else "")
        )
        run_dir = row.get("source_file", "").rsplit("/", 1)[0]
        checkpoint_bytes = checkpoint_sizes.get(run_dir)

        missing = []
        for field in (
            "configured_epochs",
            "image_size",
            "batch_size",
            "training_seconds",
            "mean_round_seconds",
        ):
            if not row.get(field):
                missing.append(field)

        row.update(
            {
                "result_status": "archived",
                "training_category": training_category,
                "training_type": training_type or "centralized",
                "configured_epochs": row.get("configured_epochs", ""),
                "total_local_epochs": total_local_epochs,
                "observed_training_budget": observed_budget,
                "image_size": row.get("image_size", ""),
                "batch_size": row.get("batch_size", ""),
                "training_seconds": row.get("training_seconds", ""),
                "mean_round_seconds": row.get("mean_round_seconds", ""),
                "checkpoint_size_mb": (
                    checkpoint_bytes / (1024**2)
                    if checkpoint_bytes is not None
                    else ""
                ),
                "metadata_status": (
                    "complete"
                    if not missing
                    else "historical_metadata_unavailable:" + ";".join(missing)
                ),
            }
        )
        for key in row:
            if key not in seen:
                all_fields.append(key)
                seen.add(key)
        aligned.append(row)

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aligned)
    return len(aligned)


if __name__ == "__main__":
    count = build()
    print(f"[archive-tableau] rows={count}")
    print(f"[archive-tableau] output={OUTPUT}")
