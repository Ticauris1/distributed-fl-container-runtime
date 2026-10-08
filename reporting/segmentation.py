"""Persistence and visual reporting for segmentation experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_history(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=True)
    return path


def _overlay(image, mask, color, alpha=0.45):
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    selected = np.asarray(mask, dtype=bool)
    array[selected] = (1.0 - alpha) * array[selected] + alpha * np.asarray(color)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def save_prediction_panels(dataset, probabilities, output_dir, threshold=0.5, limit=24):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(int(limit), len(dataset), len(probabilities))
    for index in range(count):
        row = dataset.rows[index]
        with Image.open(dataset.root / row["image_path"]) as source:
            image = source.convert("RGB").resize(
                (dataset.image_size, dataset.image_size), Image.Resampling.BILINEAR
            )
        with Image.open(dataset.root / row["mask_path"]) as source:
            target = np.asarray(
                source.convert("L").resize(
                    (dataset.image_size, dataset.image_size), Image.Resampling.NEAREST
                )
            ) >= 128
        prediction = probabilities[index, 0] >= float(threshold)
        target_overlay = _overlay(image, target, (0, 255, 0))
        prediction_overlay = _overlay(image, prediction, (255, 0, 0))
        panel = Image.new("RGB", (dataset.image_size * 3, dataset.image_size + 28), "white")
        panel.paste(image, (0, 28))
        panel.paste(target_overlay, (dataset.image_size, 28))
        panel.paste(prediction_overlay, (dataset.image_size * 2, 28))
        draw = ImageDraw.Draw(panel)
        draw.text((4, 6), "Image", fill="black")
        draw.text((dataset.image_size + 4, 6), "Target (green)", fill="black")
        draw.text((dataset.image_size * 2 + 4, 6), "Prediction (red)", fill="black")
        panel.save(output_dir / f"{index:03d}_{row['sample_id']}.jpg", quality=92)
