"""Run segmentation variants declared entirely in config.py."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

from experiments.segmentation_runner import run_segmentation_condition


AGGREGATE_METRICS = [
    "dice", "iou", "pixel_precision", "pixel_recall", "pixel_accuracy",
    "specificity", "false_positive_rate", "mean_cracked_sample_dice",
    "empty_mask_accuracy", "loss", "best_validation_dice",
]


def _write_multiseed_summary(path, rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant_name"], []).append(row)
    summaries = []
    for variant_name, values in grouped.items():
        summary = {
            "variant_name": variant_name,
            "run_count": len(values),
            "seeds": ",".join(str(value["seed"]) for value in values),
            "augmentation_profile": values[0]["augmentation_profile"],
            "learning_rate": values[0]["learning_rate"],
        }
        for metric in AGGREGATE_METRICS:
            observations = [float(value[metric]) for value in values]
            summary[f"{metric}_mean"] = mean(observations)
            summary[f"{metric}_std"] = stdev(observations) if len(observations) > 1 else 0.0
        summaries.append(summary)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return path


def run_segmentation_comparison(config):
    results = []
    for variant in config.SEGMENTATION_COMPARISON_VARIANTS:
        for experiment_seed in config.SEGMENTATION_EXPERIMENT_SEEDS:
            name = variant["name"]
            print(
                f"[segmentation-comparison] {name} | seed={experiment_seed} | "
                f"model={variant['model']} encoder={variant['encoder']} "
                f"augmentation={variant['augmentation']} "
                f"learning_rate={float(variant['learning_rate']):g}"
            )
            results.append(
                run_segmentation_condition(
                    config,
                    variant["model"],
                    variant["encoder"],
                    experiment_name=(
                        f"config_comparison/{name}/seed_{experiment_seed}"
                    ),
                    train_manifest=config.SEGMENTATION_COMPARISON_TRAIN_MANIFEST,
                    evaluation_manifest=config.SEGMENTATION_COMPARISON_EVALUATION_MANIFEST,
                    dataset_label="600_images",
                    augmentation_profile=variant["augmentation"],
                    learning_rate=variant["learning_rate"],
                    experiment_seed=experiment_seed,
                    variant_name=name,
                )
            )
    summary = Path(config.SEGMENTATION_RESULT_DIR) / "config_comparison/summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"[segmentation-comparison] summary: {summary}")
    multiseed_summary = _write_multiseed_summary(
        summary.parent / "multiseed_summary.csv", results
    )
    print(f"[segmentation-comparison] multi-seed summary: {multiseed_summary}")
    return results
