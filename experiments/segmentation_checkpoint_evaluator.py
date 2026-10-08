"""Evaluate saved segmentation checkpoints on one common test manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from data.segmentation import SegmentationDataset
from models.segmentation import make_segmentation_model
from training.segmentation import BCEDiceLoss, evaluate_segmentation


CHECKPOINTS = {
    "600_images": Path(config.SEGMENTATION_RESULT_DIR)
    / "dataset_size_comparison/600_images/unetplusplus_densenet121/checkpoints/best_unetplusplus.pth",
    "2000_images": Path(config.SEGMENTATION_RESULT_DIR)
    / "dataset_size_comparison/2000_images/unetplusplus_densenet121/checkpoints/best_unetplusplus.pth",
    "4000_images": Path(config.SEGMENTATION_RESULT_DIR)
    / "dataset_size_comparison/4000_images/unetplusplus_densenet121/checkpoints/best_unetplusplus.pth",
}


def evaluate_checkpoints(test_manifest, output_dir, threshold=0.75):
    test_manifest = Path(test_manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dataset = SegmentationDataset(
        test_manifest, "test", config.SEGMENTATION_IMAGE_SIZE, augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.SEGMENTATION_BATCH_SIZE,
        shuffle=False,
        num_workers=config.SEGMENTATION_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    criterion = BCEDiceLoss()
    rows = []
    for dataset_label, checkpoint in CHECKPOINTS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"[expanded-test] evaluating {dataset_label}: {checkpoint}")
        model = make_segmentation_model(
            "unetplusplus", config.DEVICE, "densenet121", encoder_weights=None
        )
        state = torch.load(checkpoint, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(state)
        metrics = evaluate_segmentation(
            model, test_loader, criterion, config.DEVICE, threshold
        )
        row = {
            "dataset_label": dataset_label,
            "training_images_total": int(dataset_label.split("_")[0]),
            "test_samples": len(test_dataset),
            "threshold": float(threshold),
            "test_manifest": str(test_manifest),
            "checkpoint": str(checkpoint.resolve()),
            **metrics,
        }
        rows.append(row)
        with (output_dir / f"{dataset_label}_metrics.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(row, stream, indent=2, allow_nan=True)
        print(
            f"[expanded-test] {dataset_label} dice={metrics['dice']:.4f} "
            f"iou={metrics['iou']:.4f} precision={metrics['pixel_precision']:.4f} "
            f"recall={metrics['pixel_recall']:.4f}"
        )
        del model
        if config.DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif config.DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    summary = output_dir / "fixed_threshold_075_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[expanded-test] summary: {summary}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=config.PROJECT_ROOT
        / "segmentation_pilot/workspace_4000/manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.SEGMENTATION_RESULT_DIR) / "expanded_test_evaluation",
    )
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    evaluate_checkpoints(
        arguments.test_manifest, arguments.output_dir, arguments.threshold
    )
