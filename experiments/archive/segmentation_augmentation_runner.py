"""Controlled baseline, moderate, and strong augmentation comparison."""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.segmentation_runner import run_segmentation_condition


SUMMARY_FIELDS = [
    "augmentation_profile", "model", "encoder", "seed", "train_samples",
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


def run_augmentation_experiment(config):
    results = []
    for profile in config.SEGMENTATION_AUGMENTATION_PROFILES:
        for model_name in config.SEGMENTATION_MODELS:
            for encoder_name in config.DEEP_MODELS:
                print(
                    f"[augmentation] profile={profile} model={model_name} "
                    f"encoder={encoder_name}"
                )
                results.append(
                    run_segmentation_condition(
                        config,
                        model_name,
                        encoder_name,
                        experiment_name=f"augmentation_comparison/{profile}",
                        train_manifest=config.SEGMENTATION_AUGMENTATION_TRAIN_MANIFEST,
                        evaluation_manifest=config.SEGMENTATION_AUGMENTATION_EVALUATION_MANIFEST,
                        dataset_label="2000_images",
                        augmentation_profile=profile,
                    )
                )
    summary = _write_summary(
        Path(config.SEGMENTATION_RESULT_DIR)
        / "augmentation_comparison"
        / "augmentation_summary.csv",
        results,
    )
    print(f"[augmentation] comparison summary: {summary}")
    return results
