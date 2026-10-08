"""Select segmentation thresholds on validation data without retraining."""

from __future__ import annotations

import csv
from pathlib import Path

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


def run_threshold_sweep(config):
    validation_loader = _loader(
        config.SEGMENTATION_COMPARISON_EVALUATION_MANIFEST, "val", config
    )
    test_loader = _loader(
        config.SEGMENTATION_COMPARISON_EVALUATION_MANIFEST, "test", config
    )
    criterion = BCEDiceLoss()
    validation_rows, selected_test_rows = [], []
    output_dir = Path(config.SEGMENTATION_RESULT_DIR) / "threshold_sweep"
    for variant in config.SEGMENTATION_COMPARISON_VARIANTS:
        for seed in config.SEGMENTATION_EXPERIMENT_SEEDS:
            checkpoint = _checkpoint_path(config, variant, seed)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Threshold-sweep checkpoint not found: {checkpoint}")
            print(f"[threshold] variant={variant['name']} seed={seed}")
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
            candidate_rows = []
            for threshold in config.SEGMENTATION_THRESHOLD_RANGE:
                metrics = segmentation_metrics_from_arrays(
                    validation_targets, validation_probabilities >= float(threshold)
                )
                row = {
                    "variant_name": variant["name"],
                    "model": normalize_segmentation_model_name(variant["model"]),
                    "encoder": normalize_segmentation_encoder_name(variant["encoder"]),
                    "seed": seed,
                    "threshold": float(threshold),
                    "split": "validation",
                    "loss": validation_base["loss"],
                    **metrics,
                }
                candidate_rows.append(row)
                validation_rows.append(row)
            selected = max(candidate_rows, key=lambda row: row["dice"])
            test_base, test_probabilities, test_targets = evaluate_segmentation(
                model, test_loader, criterion, config.DEVICE,
                threshold=selected["threshold"], return_predictions=True,
            )
            test_metrics = segmentation_metrics_from_arrays(
                test_targets, test_probabilities >= selected["threshold"]
            )
            selected_test_rows.append({
                "variant_name": variant["name"],
                "model": selected["model"],
                "encoder": selected["encoder"],
                "seed": seed,
                "selected_threshold": selected["threshold"],
                "validation_dice_at_selection": selected["dice"],
                "test_loss": test_base["loss"],
                **test_metrics,
            })
            print(
                f"[threshold] selected={selected['threshold']:.2f} "
                f"val_dice={selected['dice']:.4f} test_dice={test_metrics['dice']:.4f}"
            )
            del model
            if config.DEVICE.type == "mps":
                torch.mps.empty_cache()
            elif config.DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    validation_path = _write_csv(
        output_dir / "validation_threshold_sweep.csv", validation_rows
    )
    test_path = _write_csv(
        output_dir / "selected_threshold_test_results.csv", selected_test_rows
    )
    print(f"[threshold] validation sweep: {validation_path}")
    print(f"[threshold] selected-threshold test results: {test_path}")
    return selected_test_rows
