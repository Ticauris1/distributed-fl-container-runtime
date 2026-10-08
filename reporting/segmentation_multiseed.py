"""Aggregate matched segmentation replications without hiding seed-level rows."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "validation_dice", "test_loss", "dice", "iou", "pixel_precision",
    "pixel_recall", "specificity", "mean_sample_dice",
    "mean_cracked_sample_dice", "empty_mask_accuracy",
)


def write_segmentation_multiseed_summary(path, rows, group_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    summary = []
    for key, group in groups.items():
        output = dict(zip(group_fields, key))
        seeds = sorted({int(row["seed"]) for row in group})
        output.update(seed_count=len(seeds), seeds=";".join(map(str, seeds)))
        for metric in METRICS:
            values = [float(row[metric]) for row in group if metric in row]
            if values:
                output[f"{metric}_mean"] = float(np.mean(values))
                output[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(output)
    if not summary:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return path
