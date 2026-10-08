"""Select segmentation threshold and component filtering on validation data."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from data.segmentation import SegmentationDataset
from models.segmentation import (
    make_segmentation_model,
    normalize_segmentation_encoder_name,
    normalize_segmentation_model_name,
)
from training.segmentation import (
    BCEDiceLoss,
    evaluate_segmentation,
    segmentation_metrics_from_arrays,
)


def _loader(manifest, split, config):
    dataset = SegmentationDataset(
        manifest, split, config.SEGMENTATION_IMAGE_SIZE, augment=False
    )
    return DataLoader(
        dataset,
        batch_size=config.SEGMENTATION_BATCH_SIZE,
        shuffle=False,
        num_workers=config.SEGMENTATION_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _checkpoint_path(config, variant, seed):
    model = normalize_segmentation_model_name(variant["model"])
    encoder = normalize_segmentation_encoder_name(variant["encoder"])
    return (
        Path(config.SEGMENTATION_RESULT_DIR)
        / "config_comparison"
        / variant["name"]
        / f"seed_{seed}"
        / f"{model}_{encoder}"
        / "checkpoints"
        / f"best_{model}.pth"
    )


def remove_small_components(predictions, minimum_pixels):
    """Remove each 8-connected foreground component smaller than the limit."""
    predictions = np.asarray(predictions, dtype=bool)
    if minimum_pixels <= 0:
        return predictions.copy()
    filtered = np.zeros_like(predictions, dtype=bool)
    structure = np.ones((3, 3), dtype=np.uint8)
    for sample_index in range(predictions.shape[0]):
        # Accept [N, 1, H, W] and [N, H, W] arrays.
        sample = predictions[sample_index]
        plane = sample[0] if sample.ndim == 3 and sample.shape[0] == 1 else sample
        labels, component_count = ndimage.label(plane, structure=structure)
        if component_count == 0:
            continue
        sizes = np.bincount(labels.ravel())
        keep = sizes >= int(minimum_pixels)
        keep[0] = False
        filtered_plane = keep[labels]
        if sample.ndim == 3 and sample.shape[0] == 1:
            filtered[sample_index, 0] = filtered_plane
        else:
            filtered[sample_index] = filtered_plane
    return filtered


def _filter_component_grid(predictions, minimum_values):
    """Produce all component-size candidates while labeling each mask once."""
    predictions = np.asarray(predictions, dtype=bool)
    minimum_values = [int(value) for value in minimum_values]
    filtered = {
        value: predictions.copy() if value <= 0 else np.zeros_like(predictions)
        for value in minimum_values
    }
    positive_values = [value for value in minimum_values if value > 0]
    if not positive_values:
        return filtered
    structure = np.ones((3, 3), dtype=np.uint8)
    for sample_index in range(predictions.shape[0]):
        sample = predictions[sample_index]
        plane = sample[0] if sample.ndim == 3 and sample.shape[0] == 1 else sample
        labels, component_count = ndimage.label(plane, structure=structure)
        if component_count == 0:
            continue
        sizes = np.bincount(labels.ravel())
        for minimum_pixels in positive_values:
            keep = sizes >= minimum_pixels
            keep[0] = False
            filtered_plane = keep[labels]
            if sample.ndim == 3 and sample.shape[0] == 1:
                filtered[minimum_pixels][sample_index, 0] = filtered_plane
            else:
                filtered[minimum_pixels][sample_index] = filtered_plane
    return filtered


def _candidate_rows(probabilities, targets, loss, variant, seed, config, split):
    rows = []
    for threshold in config.SEGMENTATION_THRESHOLD_RANGE:
        thresholded = probabilities >= float(threshold)
        filtered_grid = _filter_component_grid(
            thresholded, config.SEGMENTATION_MIN_COMPONENT_PIXELS
        )
        for minimum_pixels, predictions in filtered_grid.items():
            rows.append({
                "variant_name": variant["name"],
                "model": normalize_segmentation_model_name(variant["model"]),
                "encoder": normalize_segmentation_encoder_name(variant["encoder"]),
                "seed": seed,
                "threshold": float(threshold),
                "minimum_component_pixels": int(minimum_pixels),
                "connectivity": 8,
                "split": split,
                "loss": loss,
                **segmentation_metrics_from_arrays(targets, predictions),
            })
    return rows


def run_postprocess_sweep(config):
    validation_loader = _loader(
        config.SEGMENTATION_COMPARISON_EVALUATION_MANIFEST, "val", config
    )
    test_loader = _loader(
        config.SEGMENTATION_COMPARISON_EVALUATION_MANIFEST, "test", config
    )
    criterion = BCEDiceLoss()
    validation_rows, selected_test_rows = [], []
    output_dir = Path(config.SEGMENTATION_RESULT_DIR) / "postprocess_sweep"

    for variant in config.SEGMENTATION_COMPARISON_VARIANTS:
        for seed in config.SEGMENTATION_EXPERIMENT_SEEDS:
            checkpoint = _checkpoint_path(config, variant, seed)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Post-processing checkpoint not found: {checkpoint}"
                )
            print(f"[postprocess] variant={variant['name']} seed={seed}")
            model = make_segmentation_model(
                variant["model"], config.DEVICE, variant["encoder"],
                encoder_weights=None,
            )
            model.load_state_dict(
                torch.load(checkpoint, map_location=config.DEVICE, weights_only=True)
            )
            validation_base, validation_probabilities, validation_targets = (
                evaluate_segmentation(
                    model, validation_loader, criterion, config.DEVICE,
                    threshold=0.5, return_predictions=True,
                )
            )
            candidates = _candidate_rows(
                validation_probabilities, validation_targets,
                validation_base["loss"], variant, seed, config, "validation",
            )
            validation_rows.extend(candidates)
            # Dice is primary. The tie-breakers prefer fewer false positives,
            # then the less aggressive filter if performance is identical.
            selected = max(
                candidates,
                key=lambda row: (
                    row["dice"], row["pixel_precision"],
                    -row["minimum_component_pixels"],
                ),
            )

            test_base, test_probabilities, test_targets = evaluate_segmentation(
                model, test_loader, criterion, config.DEVICE,
                threshold=selected["threshold"], return_predictions=True,
            )
            test_predictions = remove_small_components(
                test_probabilities >= selected["threshold"],
                selected["minimum_component_pixels"],
            )
            test_metrics = segmentation_metrics_from_arrays(
                test_targets, test_predictions
            )
            selected_test_rows.append({
                "variant_name": selected["variant_name"],
                "model": selected["model"],
                "encoder": selected["encoder"],
                "seed": seed,
                "selected_threshold": selected["threshold"],
                "selected_minimum_component_pixels": selected[
                    "minimum_component_pixels"
                ],
                "connectivity": 8,
                "validation_dice_at_selection": selected["dice"],
                "validation_precision_at_selection": selected["pixel_precision"],
                "validation_recall_at_selection": selected["pixel_recall"],
                "test_loss": test_base["loss"],
                **test_metrics,
            })
            print(
                f"[postprocess] selected threshold={selected['threshold']:.2f} "
                f"min_pixels={selected['minimum_component_pixels']} "
                f"val_dice={selected['dice']:.4f} "
                f"test_dice={test_metrics['dice']:.4f} "
                f"precision={test_metrics['pixel_precision']:.4f} "
                f"recall={test_metrics['pixel_recall']:.4f}"
            )
            del model
            if config.DEVICE.type == "mps":
                torch.mps.empty_cache()
            elif config.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    validation_path = _write_csv(
        output_dir / "validation_postprocess_sweep.csv", validation_rows
    )
    test_path = _write_csv(
        output_dir / "selected_postprocess_test_results.csv", selected_test_rows
    )
    print(f"[postprocess] validation sweep: {validation_path}")
    print(f"[postprocess] selected test results: {test_path}")
    return selected_test_rows
