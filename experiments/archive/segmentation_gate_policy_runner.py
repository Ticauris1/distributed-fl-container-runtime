"""Compare no-gate, global-Dice, and dual-constraint gate policies."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

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
from experiments.segmentation_presence_gate_runner import (
    _candidate_rows,
    _checkpoint_path,
    _loader,
    apply_presence_gate,
    select_dice_constrained_gate,
)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def select_dual_constrained_gate(
    candidates, maximum_dice_loss, maximum_cracked_dice_loss
):
    """Maximize empty accuracy while enforcing both validation Dice floors."""
    controls = [
        row for row in candidates if int(row["minimum_presence_pixels"]) == 0
    ]
    if not controls:
        raise ValueError("Gate-policy comparison requires a zero-pixel control")
    baseline = max(controls, key=lambda row: row["dice"])
    dice_floor = baseline["dice"] - float(maximum_dice_loss)
    cracked_dice_floor = (
        baseline["mean_cracked_sample_dice"]
        - float(maximum_cracked_dice_loss)
    )
    eligible = [
        row for row in candidates
        if row["dice"] >= dice_floor
        and row["mean_cracked_sample_dice"] >= cracked_dice_floor
    ]
    if not eligible:
        raise RuntimeError("No gate candidate satisfies both Dice constraints")
    selected = max(
        eligible,
        key=lambda row: (
            row["empty_mask_accuracy"],
            row["mean_sample_dice"],
            row["dice"],
            -row["minimum_presence_pixels"],
        ),
    )
    return baseline, dice_floor, cracked_dice_floor, selected


def _policies(candidates, config):
    controls = [
        row for row in candidates if int(row["minimum_presence_pixels"]) == 0
    ]
    baseline = max(controls, key=lambda row: row["dice"])
    policies = [{
        "policy": "no_gate",
        "maximum_global_dice_loss": 0.0,
        "maximum_cracked_dice_loss": 0.0,
        "global_dice_floor": baseline["dice"],
        "cracked_dice_floor": baseline["mean_cracked_sample_dice"],
        "baseline": baseline,
        "selected": baseline,
    }]

    global_baseline, global_floor, global_selected = select_dice_constrained_gate(
        candidates, config.SEGMENTATION_GATE_MAX_DICE_LOSS
    )
    policies.append({
        "policy": "global_dice_constraint",
        "maximum_global_dice_loss": config.SEGMENTATION_GATE_MAX_DICE_LOSS,
        "maximum_cracked_dice_loss": "",
        "global_dice_floor": global_floor,
        "cracked_dice_floor": "",
        "baseline": global_baseline,
        "selected": global_selected,
    })

    for cracked_loss in config.SEGMENTATION_GATE_MAX_CRACKED_DICE_LOSS_VALUES:
        dual_baseline, global_floor, cracked_floor, selected = (
            select_dual_constrained_gate(
                candidates,
                config.SEGMENTATION_GATE_MAX_DICE_LOSS,
                cracked_loss,
            )
        )
        policies.append({
            "policy": f"dual_constraint_cracked_loss_{float(cracked_loss):g}",
            "maximum_global_dice_loss": config.SEGMENTATION_GATE_MAX_DICE_LOSS,
            "maximum_cracked_dice_loss": float(cracked_loss),
            "global_dice_floor": global_floor,
            "cracked_dice_floor": cracked_floor,
            "baseline": dual_baseline,
            "selected": selected,
        })
    return policies


def run_gate_policy_comparison(config):
    validation_loader = _loader(
        config.SEGMENTATION_COMPARISON_EVALUATION_MANIFEST, "val", config
    )
    test_loader = _loader(
        config.SEGMENTATION_COMPARISON_EVALUATION_MANIFEST, "test", config
    )
    criterion = BCEDiceLoss()
    selection_rows, test_rows = [], []
    output_dir = Path(config.SEGMENTATION_RESULT_DIR) / "gate_policy_comparison"

    for variant in config.SEGMENTATION_COMPARISON_VARIANTS:
        for seed in config.SEGMENTATION_EXPERIMENT_SEEDS:
            checkpoint = _checkpoint_path(config, variant, seed)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Gate-policy checkpoint not found: {checkpoint}"
                )
            print(f"[gate-policy] variant={variant['name']} seed={seed}")
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
            policies = _policies(candidates, config)

            test_base, test_probabilities, test_targets = evaluate_segmentation(
                model, test_loader, criterion, config.DEVICE,
                threshold=0.5, return_predictions=True,
            )
            for policy in policies:
                baseline = policy["baseline"]
                selected = policy["selected"]
                common = {
                    "variant_name": variant["name"],
                    "model": normalize_segmentation_model_name(variant["model"]),
                    "encoder": normalize_segmentation_encoder_name(variant["encoder"]),
                    "seed": seed,
                    "policy": policy["policy"],
                    "maximum_global_dice_loss": policy["maximum_global_dice_loss"],
                    "maximum_cracked_dice_loss": policy["maximum_cracked_dice_loss"],
                    "baseline_threshold": baseline["threshold"],
                    "baseline_validation_dice": baseline["dice"],
                    "baseline_validation_cracked_dice": baseline[
                        "mean_cracked_sample_dice"
                    ],
                    "global_dice_floor": policy["global_dice_floor"],
                    "cracked_dice_floor": policy["cracked_dice_floor"],
                    "selected_threshold": selected["threshold"],
                    "selected_minimum_presence_pixels": selected[
                        "minimum_presence_pixels"
                    ],
                    "validation_dice_change": selected["dice"] - baseline["dice"],
                    "validation_cracked_dice_change": (
                        selected["mean_cracked_sample_dice"]
                        - baseline["mean_cracked_sample_dice"]
                    ),
                    "validation_dice": selected["dice"],
                    "validation_mean_cracked_sample_dice": selected[
                        "mean_cracked_sample_dice"
                    ],
                    "validation_empty_mask_accuracy": selected[
                        "empty_mask_accuracy"
                    ],
                    "validation_mean_sample_dice": selected["mean_sample_dice"],
                }
                selection_rows.append(common)

                predictions = apply_presence_gate(
                    test_probabilities >= selected["threshold"],
                    selected["minimum_presence_pixels"],
                )
                metrics = segmentation_metrics_from_arrays(test_targets, predictions)
                test_rows.append({
                    **common,
                    "split": "test",
                    "test_loss": test_base["loss"],
                    **metrics,
                })
                print(
                    f"[gate-policy] {policy['policy']} "
                    f"threshold={selected['threshold']:.2f} "
                    f"min_pixels={selected['minimum_presence_pixels']} "
                    f"test_dice={metrics['dice']:.4f} "
                    f"cracked_dice={metrics['mean_cracked_sample_dice']:.4f} "
                    f"empty_accuracy={metrics['empty_mask_accuracy']:.4f}"
                )

            del model
            if config.DEVICE.type == "mps":
                torch.mps.empty_cache()
            elif config.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    selection_path = _write_csv(
        output_dir / "validation_gate_policy_selections.csv", selection_rows
    )
    test_path = _write_csv(
        output_dir / "test_gate_policy_results.csv", test_rows
    )
    print(f"[gate-policy] validation selections: {selection_path}")
    print(f"[gate-policy] test results: {test_path}")
    return test_rows
