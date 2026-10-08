"""Select a segmentation-derived image-level crack-presence gate."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
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


def apply_presence_gate(predictions, minimum_presence_pixels):
    """Zero an entire predicted mask when its total crack area is too small."""
    predictions = np.asarray(predictions, dtype=bool)
    gated = predictions.copy()
    if minimum_presence_pixels <= 0:
        return gated
    positive_counts = gated.reshape(gated.shape[0], -1).sum(axis=1)
    gated[positive_counts < int(minimum_presence_pixels)] = False
    return gated


def _candidate_rows(probabilities, targets, loss, variant, seed, config):
    rows = []
    for threshold in config.SEGMENTATION_THRESHOLD_RANGE:
        thresholded = probabilities >= float(threshold)
        for minimum_pixels in config.SEGMENTATION_MIN_PRESENCE_PIXELS:
            predictions = apply_presence_gate(thresholded, minimum_pixels)
            rows.append({
                "variant_name": variant["name"],
                "model": normalize_segmentation_model_name(variant["model"]),
                "encoder": normalize_segmentation_encoder_name(variant["encoder"]),
                "seed": seed,
                "threshold": float(threshold),
                "minimum_presence_pixels": int(minimum_pixels),
                "split": "validation",
                "loss": loss,
                **segmentation_metrics_from_arrays(targets, predictions),
            })
    return rows


def select_dice_constrained_gate(candidates, maximum_dice_loss):
    """Select the best absence gate without exceeding the Dice-loss budget."""
    controls = [
        row for row in candidates if int(row["minimum_presence_pixels"]) == 0
    ]
    if not controls:
        raise ValueError("Presence-gate sweep requires a zero-pixel control")
    baseline = max(controls, key=lambda row: row["dice"])
    dice_floor = baseline["dice"] - float(maximum_dice_loss)
    eligible = [row for row in candidates if row["dice"] >= dice_floor]
    if not eligible:
        raise RuntimeError("No presence-gate candidate satisfies the Dice constraint")
    selected = max(
        eligible,
        key=lambda row: (
            row["empty_mask_accuracy"],
            row["mean_sample_dice"],
            row["dice"],
            -row["minimum_presence_pixels"],
        ),
    )
    return baseline, dice_floor, selected


def run_presence_gate_sweep(config):
    validation_loader = _loader(
        config.SEGMENTATION_COMPARISON_EVALUATION_MANIFEST, "val", config
    )
    test_loader = _loader(
        config.SEGMENTATION_COMPARISON_EVALUATION_MANIFEST, "test", config
    )
    criterion = BCEDiceLoss()
    validation_rows, selected_test_rows = [], []
    output_dir = Path(config.SEGMENTATION_RESULT_DIR) / "presence_gate_sweep"

    for variant in config.SEGMENTATION_COMPARISON_VARIANTS:
        for seed in config.SEGMENTATION_EXPERIMENT_SEEDS:
            checkpoint = _checkpoint_path(config, variant, seed)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Presence-gate checkpoint not found: {checkpoint}"
                )
            print(f"[presence-gate] variant={variant['name']} seed={seed}")
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
                validation_base["loss"], variant, seed, config,
            )
            validation_rows.extend(candidates)
            baseline, dice_floor, selected = select_dice_constrained_gate(
                candidates, config.SEGMENTATION_GATE_MAX_DICE_LOSS
            )

            test_base, test_probabilities, test_targets = evaluate_segmentation(
                model, test_loader, criterion, config.DEVICE,
                threshold=selected["threshold"], return_predictions=True,
            )
            test_predictions = apply_presence_gate(
                test_probabilities >= selected["threshold"],
                selected["minimum_presence_pixels"],
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
                "selected_minimum_presence_pixels": selected[
                    "minimum_presence_pixels"
                ],
                "selection_metric": "empty_mask_accuracy_with_dice_constraint",
                "maximum_validation_dice_loss": config.SEGMENTATION_GATE_MAX_DICE_LOSS,
                "baseline_validation_threshold": baseline["threshold"],
                "baseline_validation_dice": baseline["dice"],
                "minimum_allowed_validation_dice": dice_floor,
                "actual_validation_dice_change": (
                    selected["dice"] - baseline["dice"]
                ),
                "validation_mean_sample_dice_at_selection": selected[
                    "mean_sample_dice"
                ],
                "validation_dice_at_selection": selected["dice"],
                "validation_empty_mask_accuracy_at_selection": selected[
                    "empty_mask_accuracy"
                ],
                "test_loss": test_base["loss"],
                **test_metrics,
            })
            print(
                f"[presence-gate] selected threshold={selected['threshold']:.2f} "
                f"min_pixels={selected['minimum_presence_pixels']} "
                f"val_dice={selected['dice']:.4f} "
                f"dice_change={selected['dice'] - baseline['dice']:+.4f} "
                f"val_empty_accuracy={selected['empty_mask_accuracy']:.4f} "
                f"test_dice={test_metrics['dice']:.4f} "
                f"empty_accuracy={test_metrics['empty_mask_accuracy']:.4f}"
            )
            del model
            if config.DEVICE.type == "mps":
                torch.mps.empty_cache()
            elif config.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    validation_path = _write_csv(
        output_dir / "validation_presence_gate_sweep.csv", validation_rows
    )
    test_path = _write_csv(
        output_dir / "selected_presence_gate_test_results.csv", selected_test_rows
    )
    print(f"[presence-gate] validation sweep: {validation_path}")
    print(f"[presence-gate] selected test results: {test_path}")
    return selected_test_rows
