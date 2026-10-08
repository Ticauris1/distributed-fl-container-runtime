"""Controlled segmentation learning-curve experiment."""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.segmentation_runner import run_segmentation_condition


def _write_summary(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    preferred = [
        "model", "encoder", "seed", "train_fraction", "train_samples",
        "validation_samples", "test_samples", "dice", "iou",
        "pixel_precision", "pixel_recall", "pixel_accuracy", "specificity",
        "false_positive_rate", "mean_cracked_sample_dice",
        "empty_mask_accuracy", "loss", "best_validation_dice",
        "best_epoch", "epochs_trained",
    ]
    fields = preferred + [key for key in rows[0] if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_sample_size_experiment(config):
    results = []
    root = Path(config.SEGMENTATION_RESULT_DIR) / "sample_size_experiment"
    for model_name in config.SEGMENTATION_MODELS:
        for encoder_name in config.DEEP_MODELS:
            for fraction in config.SEGMENTATION_SAMPLE_FRACTIONS:
                label = f"fraction_{round(float(fraction) * 100):03d}pct"
                experiment_name = f"sample_size_experiment/{label}"
                print(
                    f"[sample-size] model={model_name} encoder={encoder_name} "
                    f"training_fraction={float(fraction):.0%}"
                )
                results.append(
                    run_segmentation_condition(
                        config,
                        model_name,
                        encoder_name,
                        train_fraction=fraction,
                        experiment_name=experiment_name,
                    )
                )
    summary_path = _write_summary(root / "sample_size_summary.csv", results)
    print(f"[sample-size] comparison summary: {summary_path}")
    return results
