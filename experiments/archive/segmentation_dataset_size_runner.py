"""Controlled 600/2,000/4,000-image segmentation comparison."""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.segmentation_runner import run_segmentation_condition


SUMMARY_FIELDS = [
    "dataset_label", "model", "encoder", "seed", "train_samples",
    "validation_samples", "test_samples", "dice", "iou",
    "pixel_precision", "pixel_recall", "pixel_accuracy", "specificity",
    "false_positive_rate", "mean_cracked_sample_dice",
    "empty_mask_accuracy", "loss", "best_validation_dice", "best_epoch",
    "epochs_trained", "training_manifest", "evaluation_manifest",
]


def _write_summary(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = [key for key in rows[0] if key not in SUMMARY_FIELDS]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS + extra)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_dataset_size_experiment(config):
    results = []
    for dataset_label, train_manifest in config.SEGMENTATION_DATASET_SIZE_MANIFESTS:
        for model_name in config.SEGMENTATION_MODELS:
            for encoder_name in config.DEEP_MODELS:
                print(
                    f"[dataset-size] condition={dataset_label} model={model_name} "
                    f"encoder={encoder_name}"
                )
                results.append(
                    run_segmentation_condition(
                        config,
                        model_name,
                        encoder_name,
                        experiment_name=f"dataset_size_comparison/{dataset_label}",
                        train_manifest=train_manifest,
                        evaluation_manifest=config.SEGMENTATION_FIXED_EVALUATION_MANIFEST,
                        dataset_label=dataset_label,
                    )
                )
    summary = _write_summary(
        Path(config.SEGMENTATION_RESULT_DIR)
        / "dataset_size_comparison"
        / "dataset_size_summary.csv",
        results,
    )
    print(f"[dataset-size] comparison summary: {summary}")
    return results
