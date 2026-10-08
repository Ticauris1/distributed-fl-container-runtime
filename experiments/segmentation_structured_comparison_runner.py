"""Train and evaluate config-defined centralized segmentation variants."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.segmentation import SegmentationDataset
from experiments.segmentation_runner import run_segmentation_condition
from models.segmentation import (
    make_segmentation_model,
    normalize_segmentation_encoder_name,
    normalize_segmentation_model_name,
)
from reporting.segmentation_multiseed import write_segmentation_multiseed_summary
from reporting.segmentation import save_metrics
from training.segmentation import (
    evaluate_segmentation,
    make_segmentation_loss,
    segmentation_metrics_from_arrays,
)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


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


def _threshold_candidates(probabilities, targets, loss, variant, seed, config):
    rows = []
    for threshold in config.SEGMENTATION_THRESHOLD_RANGE:
        metrics = segmentation_metrics_from_arrays(
            targets, probabilities >= float(threshold)
        )
        rows.append({
            "variant_name": variant["name"],
            "seed": seed,
            "threshold": float(threshold),
            "minimum_presence_pixels": 0,
            "loss": loss,
            **metrics,
        })
    return rows


def _checkpoint(config, variant, seed):
    model = normalize_segmentation_model_name(variant["model"])
    encoder = normalize_segmentation_encoder_name(variant["encoder"])
    return (
        Path(config.SEGMENTATION_RESULT_DIR)
        / "centralized"
        / config.SEGMENTATION_STRUCTURED_STUDY_NAME
        / variant["name"]
        / f"seed_{seed}"
        / f"{model}_{encoder}"
        / "checkpoints"
        / f"best_{model}.pth"
    )


def _completed_training(config, variant, seed):
    checkpoint = _checkpoint(config, variant, seed)
    run_dir = checkpoint.parent.parent
    return (
        checkpoint.is_file()
        and (run_dir / "training_history.csv").is_file()
        and (run_dir / "test_metrics.json").is_file()
    )


def _select_threshold(candidates):
    return max(
        candidates,
        key=lambda row: (
            row["dice"], row["mean_cracked_sample_dice"], row["pixel_precision"]
        ),
    )


def run_structured_segmentation_comparison(config):
    validation_loader = _loader(
        config.SEGMENTATION_STRUCTURED_EVALUATION_MANIFEST, "val", config
    )
    test_loader = _loader(
        config.SEGMENTATION_STRUCTURED_EVALUATION_MANIFEST, "test", config
    )
    rows = []
    output_dir = (
        Path(config.SEGMENTATION_RESULT_DIR)
        / "centralized"
        / config.SEGMENTATION_STRUCTURED_STUDY_NAME
    )

    variants = config.enabled_variants(
        config.SEGMENTATION_STRUCTURED_VARIANTS,
        config.SEGMENTATION_STRUCTURED_ENABLED_VARIANTS,
    )
    for variant in variants:
        loss_name = variant.get("loss", "bce_dice")
        positive_class_weight = float(variant.get("positive_class_weight", 5.0))
        attention = variant.get("attention", "none")
        cbam_reduction = int(variant.get("cbam_reduction", 16))
        cbam_spatial_kernel_size = int(
            variant.get("cbam_spatial_kernel_size", 7)
        )
        criterion = make_segmentation_loss(
            loss_name, positive_class_weight=positive_class_weight
        )
        for seed in config.SEGMENTATION_EXPERIMENT_SEEDS:
            print(f"[segmentation-grid] training {variant['name']} seed={seed}")
            checkpoint = _checkpoint(config, variant, seed)
            if _completed_training(config, variant, seed):
                print(
                    f"[segmentation-grid] reusing completed checkpoint: {checkpoint}"
                )
            else:
                run_segmentation_condition(
                    config,
                    variant["model"],
                    variant["encoder"],
                    experiment_name=(
                        "centralized/"
                        f"{config.SEGMENTATION_STRUCTURED_STUDY_NAME}/"
                        f"{variant['name']}/seed_{seed}"
                    ),
                    train_manifest=config.SEGMENTATION_STRUCTURED_TRAIN_MANIFEST,
                    evaluation_manifest=config.SEGMENTATION_STRUCTURED_EVALUATION_MANIFEST,
                    dataset_label=Path(
                        config.SEGMENTATION_STRUCTURED_TRAIN_MANIFEST
                    ).parent.name,
                    augmentation_profile=variant["augmentation"],
                    learning_rate=variant["learning_rate"],
                    experiment_seed=seed,
                    variant_name=variant["name"],
                    scheduler_name=variant.get("scheduler", "none"),
                    encoder_freeze_epochs=variant.get("encoder_freeze_epochs", 0),
                    encoder_learning_rate=variant.get("encoder_learning_rate"),
                    loss_name=loss_name,
                    positive_class_weight=positive_class_weight,
                    attention=attention,
                    cbam_reduction=cbam_reduction,
                    cbam_spatial_kernel_size=cbam_spatial_kernel_size,
                )

            model = make_segmentation_model(
                variant["model"], config.DEVICE, variant["encoder"],
                encoder_weights=None,
                attention=attention,
                cbam_reduction=cbam_reduction,
                cbam_spatial_kernel_size=cbam_spatial_kernel_size,
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
            candidates = _threshold_candidates(
                validation_probabilities, validation_targets,
                validation_base["loss"], variant, seed, config,
            )
            test_base, test_probabilities, test_targets = evaluate_segmentation(
                model, test_loader, criterion, config.DEVICE,
                threshold=0.5, return_predictions=True,
            )

            for policy_name in variant["gate_policies"]:
                policy = config.SEGMENTATION_GATE_POLICIES[policy_name]
                if policy["type"] != "none":
                    raise ValueError("The streamlined runner supports no_gate only")
                selected = _select_threshold(candidates)
                baseline = selected
                dice_floor = selected["dice"]
                cracked_floor = selected["mean_cracked_sample_dice"]
                predictions = np.asarray(
                    test_probabilities >= selected["threshold"], dtype=bool
                )
                metrics = segmentation_metrics_from_arrays(test_targets, predictions)
                row = {
                    "variant_name": variant["name"],
                    "model": normalize_segmentation_model_name(variant["model"]),
                    "encoder": normalize_segmentation_encoder_name(variant["encoder"]),
                    "seed": seed,
                    "augmentation_profile": variant["augmentation"],
                    "learning_rate": float(variant["learning_rate"]),
                    "scheduler": variant.get("scheduler", "none"),
                    "loss_name": loss_name,
                    "positive_class_weight": positive_class_weight,
                    "attention": attention,
                    "cbam_reduction": cbam_reduction,
                    "cbam_spatial_kernel_size": cbam_spatial_kernel_size,
                    "encoder_freeze_epochs": int(
                        variant.get("encoder_freeze_epochs", 0)
                    ),
                    "encoder_learning_rate": float(
                        variant.get("encoder_learning_rate", variant["learning_rate"])
                    ),
                    "gate_policy": policy_name,
                    "gate_policy_type": policy["type"],
                    "maximum_global_dice_loss": policy.get(
                        "maximum_global_dice_loss", 0.0
                    ),
                    "maximum_cracked_dice_loss": policy.get(
                        "maximum_cracked_dice_loss", 0.0
                    ),
                    "baseline_validation_dice": baseline["dice"],
                    "baseline_validation_cracked_dice": baseline[
                        "mean_cracked_sample_dice"
                    ],
                    "global_dice_floor": dice_floor,
                    "cracked_dice_floor": cracked_floor,
                    "selected_threshold": selected["threshold"],
                    "selected_minimum_presence_pixels": selected[
                        "minimum_presence_pixels"
                    ],
                    "validation_dice": selected["dice"],
                    "validation_cracked_dice": selected[
                        "mean_cracked_sample_dice"
                    ],
                    "validation_empty_mask_accuracy": selected[
                        "empty_mask_accuracy"
                    ],
                    "test_loss": test_base["loss"],
                    **metrics,
                    "training_manifest": str(
                        Path(config.SEGMENTATION_STRUCTURED_TRAIN_MANIFEST).resolve()
                    ),
                    "evaluation_manifest": str(
                        Path(config.SEGMENTATION_STRUCTURED_EVALUATION_MANIFEST).resolve()
                    ),
                }
                rows.append(row)
                # Replace the fixed-threshold training artifact with the
                # validation-selected threshold result used by the structured
                # comparison and Tableau export. Preserve training-budget and
                # runtime metadata written by run_segmentation_condition.
                metrics_path = checkpoint.parent.parent / "test_metrics.json"
                existing = {}
                if metrics_path.is_file():
                    import json

                    existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                save_metrics(metrics_path, {**existing, **row})
                print(
                    f"[segmentation-grid] {variant['name']} seed={seed} "
                    f"gate={policy_name} threshold={selected['threshold']:.2f} "
                    f"min_pixels={selected['minimum_presence_pixels']} "
                    f"test_dice={metrics['dice']:.4f}"
                )

            del model
            if config.DEVICE.type == "mps":
                torch.mps.empty_cache()
            elif config.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    summary = _write_csv(output_dir / "structured_comparison_summary.csv", rows)
    aggregate = write_segmentation_multiseed_summary(
        output_dir / "aggregate_results.csv",
        rows,
        ("variant_name", "model", "encoder", "gate_policy"),
    )
    print(f"[segmentation-grid] summary: {summary}")
    print(f"[segmentation-grid] aggregate: {aggregate}")
    return rows
