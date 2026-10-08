"""Controlled federated grid for binary crack segmentation."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from data.segmentation import SegmentationDataset
from models.segmentation import make_segmentation_model, normalize_segmentation_encoder_name, normalize_segmentation_model_name
from reporting.segmentation import save_history, save_metrics, save_prediction_panels
from reporting.segmentation_multiseed import write_segmentation_multiseed_summary
from training.aggregation import weighted_average_weights
from training.segmentation import evaluate_segmentation, make_segmentation_loss, segmentation_metrics_from_arrays, train_segmentation_epoch, train_segmentation_fedprox_epoch
from training.segmentation_celm import (
    PaperSegmentationCELMContributionEstimator,
    SegmentationCELMContributionEstimator,
)
from privacy.ckks_integration import ckks_aggregate_selected_params


def _release_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _partition_indices(dataset, client_count, seed, distribution):
    if distribution == "crack_area_skew":
        ranked = []
        for index, row in enumerate(dataset.rows):
            with Image.open(dataset.root / row["mask_path"]) as mask:
                ratio = float((np.asarray(mask.convert("L")) >= 128).mean())
            ranked.append((ratio, index))
        ranked.sort()
        # Contiguous ranked partitions create progressively different crack-area
        # distributions while remaining exhaustive, disjoint, and near-equal in size.
        clients = [
            [ranked[int(position)][1] for position in positions]
            for positions in np.array_split(np.arange(len(ranked)), int(client_count))
        ]
    elif distribution == "source_non_iid":
        if int(client_count) != 2:
            raise ValueError("Source-domain non-IID requires exactly two clients")
        clients = [[], []]
        for index, row in enumerate(dataset.rows):
            clients[0 if row.get("dataset_source") == "ccss" else 1].append(index)
        for client_id, indices in enumerate(clients):
            random.Random(int(seed) + client_id).shuffle(indices)
    elif distribution == "iid":
        groups = {False: [], True: []}
        for index, row in enumerate(dataset.rows):
            cracked = str(row["is_cracked"]).strip().lower() in {"1", "true", "yes"}
            groups[cracked].append(index)
        clients = [[] for _ in range(int(client_count))]
        for offset, indices in enumerate(groups.values()):
            ordered = list(indices)
            random.Random(int(seed) + offset).shuffle(ordered)
            for position, index in enumerate(ordered):
                clients[position % len(clients)].append(index)
        for client_id, indices in enumerate(clients):
            random.Random(int(seed) + 1000 + client_id).shuffle(indices)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    flat = [index for indices in clients for index in indices]
    if len(flat) != len(dataset) or len(set(flat)) != len(dataset):
        raise RuntimeError("Federated partition is not exhaustive and disjoint")
    if any(not indices for indices in clients):
        raise RuntimeError("Federated partition produced an empty client")
    return clients


def _client_loaders(dataset, indices, config, seed):
    return [
        DataLoader(
            Subset(dataset, client_indices),
            batch_size=config.SEGMENTATION_BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed) + client_id),
            num_workers=config.SEGMENTATION_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        for client_id, client_indices in enumerate(indices)
    ]


def _evaluation_loader(manifest, split, config):
    dataset = SegmentationDataset(manifest, split, config.SEGMENTATION_IMAGE_SIZE, augment=False)
    return DataLoader(dataset, batch_size=config.SEGMENTATION_BATCH_SIZE, shuffle=False,
                      num_workers=config.SEGMENTATION_NUM_WORKERS,
                      pin_memory=torch.cuda.is_available())


def _select_validation_threshold(probabilities, targets, thresholds):
    candidates = []
    for threshold in thresholds:
        metrics = segmentation_metrics_from_arrays(targets, probabilities >= float(threshold))
        candidates.append({"threshold": float(threshold), **metrics})
    return max(candidates, key=lambda row: (row["dice"], row["mean_cracked_sample_dice"], row["pixel_precision"]))


class _FedAdamState:
    def __init__(self, parameter_names, lr, beta1, beta2, epsilon):
        self.parameter_names = set(parameter_names)
        self.lr, self.beta1, self.beta2, self.epsilon = map(float, (lr, beta1, beta2, epsilon))
        self.m, self.v, self.step_number = {}, {}, 0

    def step(self, base_state, averaged_state):
        self.step_number += 1
        updated = {}
        for key, base in base_state.items():
            averaged = averaged_state[key]
            if key not in self.parameter_names or not base.is_floating_point():
                updated[key] = averaged.clone()
                continue
            delta = averaged.float() - base.float()
            if key not in self.m:
                self.m[key], self.v[key] = torch.zeros_like(delta), torch.zeros_like(delta)
            self.m[key].mul_(self.beta1).add_(delta, alpha=1.0 - self.beta1)
            self.v[key].mul_(self.beta2).addcmul_(delta, delta, value=1.0 - self.beta2)
            moment = self.m[key] / (1.0 - self.beta1 ** self.step_number)
            variance = self.v[key] / (1.0 - self.beta2 ** self.step_number)
            updated[key] = (base.float() + self.lr * moment / (torch.sqrt(variance) + self.epsilon)).to(base.dtype)
        return updated


def _centralized_reference(config):
    path = Path(config.SEGMENTATION_FEDERATED_CENTRALIZED_SUMMARY)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    variants = config.enabled_variants(
        config.SEGMENTATION_FEDERATED_VARIANTS,
        config.SEGMENTATION_FEDERATED_ENABLED_VARIANTS,
    )
    encoders = {
        normalize_segmentation_encoder_name(
            variant.get("encoder", config.SEGMENTATION_FEDERATED_ENCODER)
        )
        for variant in variants
    }
    return [
        row for row in rows
        if row.get("encoder") in encoders and row.get("gate_policy") == "no_gate"
    ]


def _write_csv(path, rows):
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run_condition(config, variant, seed, output_dir):
    training_started = perf_counter()
    config.set_seed(seed)
    training_manifest = Path(
        variant.get("manifest", config.SEGMENTATION_FEDERATED_MANIFEST)
    )
    evaluation_manifest = Path(
        variant.get("evaluation_manifest", training_manifest)
    )
    augmentation_profile = variant.get(
        "augmentation", config.SEGMENTATION_FEDERATED_AUGMENTATION
    )
    train_dataset = SegmentationDataset(
        training_manifest, "train", config.SEGMENTATION_IMAGE_SIZE,
        augment=augmentation_profile != "none",
        augmentation_profile=augmentation_profile,
    )
    client_count = int(variant.get("clients", config.SEGMENTATION_FEDERATED_CLIENTS))
    client_indices = _partition_indices(
        train_dataset, client_count, seed, variant["distribution"]
    )
    client_loaders = _client_loaders(train_dataset, client_indices, config, seed)
    validation_loader = _evaluation_loader(evaluation_manifest, "val", config)
    test_loader = _evaluation_loader(evaluation_manifest, "test", config)
    selected_per_round = max(1, math.ceil(config.SEGMENTATION_FEDERATED_CLIENT_FRACTION * len(client_loaders)))
    rounds = int(variant.get("rounds", config.SEGMENTATION_FEDERATED_ROUNDS))
    local_epochs = int(variant.get("local_epochs", config.SEGMENTATION_FEDERATED_LOCAL_EPOCHS))
    encoder = variant.get("encoder", config.SEGMENTATION_FEDERATED_ENCODER)
    loss_name = variant.get("loss", "bce_dice")
    positive_class_weight = float(variant.get("positive_class_weight", 5.0))
    fedprox_mu = float(variant.get("fedprox_mu", 0.0))
    attention = variant.get("attention", "none")
    cbam_reduction = int(variant.get("cbam_reduction", 16))
    cbam_spatial_kernel_size = int(variant.get("cbam_spatial_kernel_size", 7))
    model_kwargs = {
        "attention": attention,
        "cbam_reduction": cbam_reduction,
        "cbam_spatial_kernel_size": cbam_spatial_kernel_size,
    }
    rng = random.Random(seed)
    criterion = make_segmentation_loss(
        loss_name, positive_class_weight=positive_class_weight
    )
    initial_model = make_segmentation_model(
        config.SEGMENTATION_FEDERATED_MODEL, config.DEVICE,
        encoder, config.SEGMENTATION_ENCODER_WEIGHTS,
        **model_kwargs,
    )
    global_state = _cpu_state(initial_model)
    parameter_names = [name for name, _ in initial_model.named_parameters()]
    segmentation_head_keys = tuple(
        key for key in global_state if key.startswith("segmentation_head.")
    )
    del initial_model
    _release_cache(config.DEVICE)
    server_optimizer = None
    if variant["method"].lower() == "fedadam":
        server_learning_rate = float(
            variant.get("server_learning_rate", config.SEGMENTATION_FEDADAM_SERVER_LR)
        )
        server_optimizer = _FedAdamState(
            parameter_names, server_learning_rate,
            config.SEGMENTATION_FEDADAM_BETA1, config.SEGMENTATION_FEDADAM_BETA2,
            config.SEGMENTATION_FEDADAM_EPS,
        )
    celm = None
    if variant["weighting"] == "celm":
        celm = SegmentationCELMContributionEstimator(
            config.SEGMENTATION_FEDERATED_MODEL, encoder,
            config.SEGMENTATION_IMAGE_SIZE, len(client_loaders), rounds, config.DEVICE,
            warmup_rounds=config.CELM_WARMUP_ROUNDS,
            warmup_fraction=config.CELM_WARMUP_FRACTION, steps=config.CELM_LM_STEPS,
            lr=config.CELM_LM_LEARNING_RATE, l2_penalty=config.CELM_L2_PENALTY,
            ema_beta=config.CELM_EMA_BETA, seed=seed,
        )
    elif variant["weighting"] == "paper_celm":
        if variant["method"].lower() != "fedavg":
            raise ValueError(
                "paper_celm is the paper-aligned FedAvg condition; use the existing "
                "celm mode for FedAdam extension experiments"
            )
        if selected_per_round != len(client_loaders):
            raise ValueError("paper_celm requires full client participation")
        if not segmentation_head_keys:
            raise ValueError("No segmentation_head state keys found for paper_celm")
        celm = PaperSegmentationCELMContributionEstimator(
            config.SEGMENTATION_FEDERATED_MODEL, encoder,
            config.SEGMENTATION_IMAGE_SIZE, len(client_loaders), rounds, config.DEVICE,
            warmup_rounds=config.PAPER_CELM_WARMUP_ROUNDS,
            warmup_fraction=config.PAPER_CELM_WARMUP_FRACTION,
            steps=config.PAPER_CELM_LM_STEPS,
            lr=config.PAPER_CELM_LM_LEARNING_RATE,
            l2_penalty=config.PAPER_CELM_L2_PENALTY,
            ema_beta=config.PAPER_CELM_EMA_BETA, seed=seed,
        )

    local_segmentation_heads = [None] * len(client_loaders)

    condition_dir = output_dir / variant["name"] / f"seed_{seed}"
    checkpoint_dir = condition_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "best_federated_unetplusplus.pth"
    history, best_validation_dice = [], float("-inf")
    best_round, best_threshold = 0, None
    for round_index in range(1, rounds + 1):
        round_started = perf_counter()
        selected = rng.sample(range(len(client_loaders)), selected_per_round)
        local_states, sample_counts, client_losses = [], [], []
        base_state = global_state
        print(f"[segmentation-fed] variant={variant['name']} seed={seed} round={round_index}/{rounds}")
        for client_id in selected:
            local_model = make_segmentation_model(
                config.SEGMENTATION_FEDERATED_MODEL, config.DEVICE,
                encoder, encoder_weights=None,
                **model_kwargs,
            )
            local_initial_state = base_state
            if (
                variant["weighting"] == "paper_celm"
                and round_index <= celm.warmup_rounds
                and local_segmentation_heads[client_id] is not None
            ):
                local_initial_state = dict(base_state)
                local_initial_state.update(local_segmentation_heads[client_id])
            local_model.load_state_dict(local_initial_state, strict=True)
            optimizer = torch.optim.AdamW(local_model.parameters(),
                                          lr=config.SEGMENTATION_FEDERATED_LEARNING_RATE,
                                          weight_decay=config.SEGMENTATION_WEIGHT_DECAY)
            global_parameters = {name: value.detach().clone() for name, value in local_model.named_parameters()}
            if variant["method"].lower() == "fedprox":
                losses = [train_segmentation_fedprox_epoch(
                    local_model, client_loaders[client_id], optimizer, criterion,
                    config.DEVICE, global_parameters, fedprox_mu,
                ) for _ in range(local_epochs)]
            else:
                losses = [train_segmentation_epoch(
                    local_model, client_loaders[client_id], optimizer, criterion,
                    config.DEVICE,
                ) for _ in range(local_epochs)]
            local_state = _cpu_state(local_model)
            local_states.append(local_state)
            if (
                variant["weighting"] == "paper_celm"
                and round_index <= celm.warmup_rounds
            ):
                local_segmentation_heads[client_id] = {
                    key: local_state[key].clone() for key in segmentation_head_keys
                }
            sample_counts.append(len(client_loaders[client_id].dataset))
            client_losses.append(float(np.mean(losses)))
            del optimizer, local_model
            _release_cache(config.DEVICE)
        if variant["weighting"] == "sample_size":
            aggregation_weights = np.asarray(sample_counts, dtype=np.float64)
            weighting_stats = {"phase": "sample_size", "estimation_seconds": 0.0}
        elif variant["weighting"] == "uniform":
            aggregation_weights = np.ones(len(local_states), dtype=np.float64)
            weighting_stats = {"phase": "uniform", "estimation_seconds": 0.0}
        else:
            aggregation_weights, weighting_stats = celm.weights_for_round(
                round_index - 1, base_state, local_states, selected
            )
        averaged_state = weighted_average_weights(local_states, aggregation_weights)
        ckks_stats = {"enabled": False}
        if config.USE_CKKS_AGGREGATION:
            averaged_state, ckks_stats = ckks_aggregate_selected_params(
                averaged_state, local_states, aggregation_weights,
                "segmentation", config.CKKS_SEGMENTATION_TARGET,
                config.CKKS_POLY_MODULUS_DEGREE, config.CKKS_COEFF_MOD_BIT_SIZES,
                config.CKKS_GLOBAL_SCALE,
            )
        global_state = averaged_state if server_optimizer is None else server_optimizer.step(base_state, averaged_state)
        validation_model = make_segmentation_model(
            config.SEGMENTATION_FEDERATED_MODEL, config.DEVICE,
            encoder, encoder_weights=None,
            **model_kwargs,
        )
        validation_model.load_state_dict(global_state, strict=True)
        validation_base, probabilities, targets = evaluate_segmentation(
            validation_model, validation_loader, criterion, config.DEVICE,
            threshold=0.5, return_predictions=True,
        )
        selected_threshold = _select_validation_threshold(probabilities, targets,
                                                          config.SEGMENTATION_THRESHOLD_RANGE)
        row = {
            "round": round_index,
            "selected_clients": ";".join(str(value + 1) for value in selected),
            "selected_client_samples": ";".join(map(str, sample_counts)),
            "aggregation_weights": ";".join(f"{float(value):.10f}" for value in aggregation_weights),
            "weighting_phase": weighting_stats["phase"],
            "weighting_seconds": weighting_stats["estimation_seconds"],
            "ckks_enabled": ckks_stats.get("enabled", False),
            "ckks_target": ckks_stats.get("target", "disabled"),
            "ckks_protected_tensor_count": ckks_stats.get("protected_tensor_count", 0),
            "ckks_protected_parameter_count": ckks_stats.get("protected_parameter_count", 0),
            "ckks_plaintext_bytes": ckks_stats.get("plaintext_bytes", 0),
            "ckks_encrypt_seconds": ckks_stats.get("encrypt_seconds", 0.0),
            "ckks_aggregation_seconds": ckks_stats.get("aggregation_seconds", 0.0),
            "ckks_decrypt_seconds": ckks_stats.get("decrypt_seconds", 0.0),
            "ckks_ciphertext_bytes": ckks_stats.get("ciphertext_bytes", 0),
            "ckks_max_abs_error": ckks_stats.get("max_abs_error", 0.0),
            "ckks_mean_abs_error": ckks_stats.get("mean_abs_error", 0.0),
            "mean_client_train_loss": float(np.mean(client_losses)),
            "validation_loss": validation_base["loss"],
            "selected_threshold": selected_threshold["threshold"],
            "validation_dice": selected_threshold["dice"],
            "validation_iou": selected_threshold["iou"],
            "validation_precision": selected_threshold["pixel_precision"],
            "validation_recall": selected_threshold["pixel_recall"],
            "validation_cracked_sample_dice": selected_threshold["mean_cracked_sample_dice"],
        }
        row["round_seconds"] = perf_counter() - round_started
        history.append(row)
        print(
            f"[segmentation-fed] round={round_index:03d} "
            f"val_loss={row['validation_loss']:.4f} "
            f"val_dice={row['validation_dice']:.4f} "
            f"val_iou={row['validation_iou']:.4f} "
            f"threshold={row['selected_threshold']:.2f}"
        )
        if row["validation_dice"] > best_validation_dice:
            best_validation_dice = row["validation_dice"]
            best_round, best_threshold = round_index, row["selected_threshold"]
            torch.save(global_state, checkpoint)
        del validation_model
        _release_cache(config.DEVICE)
    save_history(condition_dir / "round_history.csv", history)
    final_model = make_segmentation_model(config.SEGMENTATION_FEDERATED_MODEL, config.DEVICE,
                                          encoder,
                                          encoder_weights=None, **model_kwargs)
    final_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    test_base, test_probabilities, test_targets = evaluate_segmentation(
        final_model, test_loader, criterion, config.DEVICE,
        threshold=best_threshold, return_predictions=True,
    )
    metrics = segmentation_metrics_from_arrays(test_targets, test_probabilities >= float(best_threshold))
    result = {
        "training_type": "federated", "variant_name": variant["name"],
        "method": variant["method"], "weighting": variant["weighting"],
        "model": normalize_segmentation_model_name(config.SEGMENTATION_FEDERATED_MODEL),
        "encoder": normalize_segmentation_encoder_name(encoder),
        "seed": seed, "client_distribution": variant["distribution"],
        "clients": client_count,
        "client_sample_counts": ";".join(str(len(indices)) for indices in client_indices),
        "client_fraction": config.SEGMENTATION_FEDERATED_CLIENT_FRACTION,
        "rounds": rounds, "local_epochs": local_epochs,
        "total_local_epochs": rounds * local_epochs,
        "configured_epochs": 0,
        "image_size": int(config.SEGMENTATION_IMAGE_SIZE),
        "batch_size": int(config.SEGMENTATION_BATCH_SIZE),
        "loss_name": loss_name,
        "positive_class_weight": positive_class_weight,
        "fedprox_mu": fedprox_mu,
        "attention": attention, "cbam_reduction": cbam_reduction,
        "cbam_spatial_kernel_size": cbam_spatial_kernel_size,
        "augmentation_profile": augmentation_profile,
        "training_manifest": str(training_manifest.resolve()),
        "evaluation_manifest": str(evaluation_manifest.resolve()),
        "learning_rate": config.SEGMENTATION_FEDERATED_LEARNING_RATE,
        "server_learning_rate": (
            float(variant.get("server_learning_rate", config.SEGMENTATION_FEDADAM_SERVER_LR))
            if variant["method"].lower() == "fedadam" else 0.0
        ),
        "best_round": best_round, "selected_threshold": best_threshold,
        "validation_dice": best_validation_dice, "test_loss": test_base["loss"], **metrics,
        "training_seconds": perf_counter() - training_started,
        "mean_round_seconds": (
            sum(float(row["round_seconds"]) for row in history) / len(history)
            if history else 0.0
        ),
        "checkpoint_size_mb": checkpoint.stat().st_size / (1024 ** 2),
    }
    save_metrics(condition_dir / "test_metrics.json", result)
    save_prediction_panels(test_loader.dataset, test_probabilities,
                           condition_dir / "test_predictions", best_threshold,
                           config.SEGMENTATION_PREVIEW_COUNT)
    del final_model
    _release_cache(config.DEVICE)
    return result


def run_federated_segmentation_comparison(config):
    output_dir = Path(config.SEGMENTATION_RESULT_DIR) / "federated" / config.SEGMENTATION_FEDERATED_STUDY_NAME
    results = []
    variants = config.enabled_variants(
        config.SEGMENTATION_FEDERATED_VARIANTS,
        config.SEGMENTATION_FEDERATED_ENABLED_VARIANTS,
    )
    for variant in variants:
        for seed in config.SEGMENTATION_FEDERATED_SEEDS:
            results.append(_run_condition(config, variant, seed, output_dir))
            _write_csv(output_dir / "federated_summary.csv", results)
    comparison = []
    for row in _centralized_reference(config):
        comparison.append({
            "training_type": "centralized", "variant_name": "centralized_no_gate",
            "method": "centralized", "weighting": "not_applicable",
            "client_distribution": "not_applicable", "seed": row["seed"],
            "selected_threshold": row["selected_threshold"], "dice": row["dice"],
            "iou": row["iou"], "pixel_precision": row["pixel_precision"],
            "pixel_recall": row["pixel_recall"], "specificity": row["specificity"],
            "mean_cracked_sample_dice": row["mean_cracked_sample_dice"],
            "empty_mask_accuracy": row["empty_mask_accuracy"],
        })
    keys = list(comparison[0]) if comparison else []
    comparison.extend({key: row[key] for key in keys} for row in results)
    comparison_path = _write_csv(output_dir / "centralized_vs_federated.csv", comparison)
    aggregate_path = write_segmentation_multiseed_summary(
        output_dir / "aggregate_results.csv",
        results,
        ("variant_name", "method", "weighting", "client_distribution", "clients"),
    )
    print(f"[segmentation-fed] summary: {output_dir / 'federated_summary.csv'}")
    print(f"[segmentation-fed] aggregate: {aggregate_path}")
    print(f"[segmentation-fed] centralized comparison: {comparison_path}")
    return results
