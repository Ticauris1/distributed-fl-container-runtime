"""Federated deep-learning orchestration for FedAvg and FedAdam."""

import copy
import random
import numpy as np #type: ignore
import torch #type: ignore
import torch.nn as nn #type: ignore
from sklearn.metrics import confusion_matrix #type: ignore
from torch.optim import AdamW #type: ignore

from config import (
    EVAL_AGG_MODE, PREDICTION_WEIGHT_MODE, PRIVACY_AGG_TARGET,
    TRAIN_AGG_MODE, USE_PRIVACY_AWARE_AGGREGATION,
)
from models.factory import make_dl_model
from privacy.integration import PrivacyAggregationError, privacy_aware_aggregate_selected_params
from privacy.ckks_integration import ckks_aggregate_selected_params
from .aggregation import FedAdamServer, weighted_average_weights
from .celm import CELMContributionEstimator, PaperCELMContributionEstimator
from .evaluation import epoch_eval_fed, evaluate_client_ensemble_on_loader
from .local import epoch_train


def _clone_state_dict_cpu(model):
    """Clone a model state to CPU without retaining accelerator storage."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _release_accelerator_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _classifier_state_keys(model):
    """Return classifier/head state keys that remain client-local in paper CELM."""
    if hasattr(model, "head"):
        prefix = "head."
    elif hasattr(model, "classifier"):
        prefix = "classifier."
    else:
        raise ValueError(
            "paper_celm requires a model exposing a 'head' or 'classifier' module"
        )
    keys = tuple(key for key in model.state_dict() if key.startswith(prefix))
    if not keys:
        raise ValueError(f"No classifier state keys found with prefix {prefix!r}")
    return keys


def _state_with_local_classifier(base_state, local_classifier_state, classifier_keys):
    """Overlay a retained client head onto the broadcast global backbone."""
    if local_classifier_state is None:
        return base_state
    missing = set(classifier_keys) - set(local_classifier_state)
    if missing:
        raise ValueError(f"Retained classifier state is missing keys: {sorted(missing)}")
    merged = dict(base_state)
    merged.update({key: local_classifier_state[key] for key in classifier_keys})
    return merged


def _apply_privacy_if_enabled(
    averaged_state, local_weights, model_name, device,
    privacy_agg_enabled=None, privacy_agg_target=None, client_weights=None,
    weighting_mode=None,
):
    enabled = USE_PRIVACY_AWARE_AGGREGATION if privacy_agg_enabled is None else privacy_agg_enabled
    target = PRIVACY_AGG_TARGET if privacy_agg_target is None else privacy_agg_target
    if not enabled:
        return averaged_state, {
            "enabled": False, "status": "disabled", "fallback_used": False,
            "privacy_agg_target": target,
        }
    print(f"[PRIVACY-AGG] Applying privacy-aware aggregation to target={target}")
    try:
        state, stats = privacy_aware_aggregate_selected_params(
            averaged_state, local_weights, model_name, device, target,
            client_weights=client_weights, weighting_mode=weighting_mode,
        )
        print(f"[PRIVACY-AGG] target={stats['privacy_agg_target']} | clients={stats['num_clients']} | vector_dim={stats['vector_dim']} | keys={stats['param_keys']}")
        return state, stats
    except Exception as error:
        print(f"[PRIVACY-AGG][error] Aggregation failed; aborting run: {error}")
        raise PrivacyAggregationError(
            f"Privacy-aware aggregation failed for target={target}; "
            "plaintext fallback is disabled."
        ) from error


def _final_client_results(
    global_model, last_local_states, client_dataloaders, test_loader,
    criterion, device, num_classes, model_name,
):
    confusion_matrices = {}
    client_pairs = []
    for client_id, loaders in enumerate(client_dataloaders):
        state = last_local_states[client_id]
        if state is None:
            model = global_model
            owns_model = False
        else:
            model = make_dl_model(model_name, num_classes, device, pretrained=False)
            model.load_state_dict(state, strict=True)
            model.eval()
            owns_model = True
        local_eval_loader = loaders.get("test") or loaders.get("val") or loaders["train"]
        _, _, labels, probabilities = epoch_eval_fed(model, local_eval_loader, criterion, device)
        confusion_matrices[client_id] = confusion_matrix(labels, np.argmax(probabilities, axis=1), labels=list(range(num_classes)))
        pair_loader = loaders.get("test") or test_loader
        _, _, pair_labels, pair_probabilities = epoch_eval_fed(model, pair_loader, criterion, device)
        client_pairs.append((pair_labels, pair_probabilities))
        if owns_model:
            del model
            _release_accelerator_cache(device)
    return confusion_matrices, client_pairs


def federated_training(
    model_name,
    source_name,
    client_dataloaders,
    server_val_loader,
    test_loader,
    num_classes,
    device,
    rounds=10,
    local_epochs=5,
    clients_per_round=4,
    lr=1e-3,
    server_method="FedAvg",
    server_lr=1e-2,
    server_beta1=0.9,
    server_beta2=0.999,
    server_eps=1e-8,
    train_agg_mode=None,
    eval_agg_mode=None,
    prediction_weight_mode=None,
    pretrained=True,
    privacy_agg_enabled=None,
    privacy_agg_target=None,
    client_weight_method="sample_size",
    celm_warmup_rounds=None,
    celm_warmup_fraction=0.05,
    celm_steps=25,
    celm_lr=1e-2,
    celm_l2_penalty=1e-3,
    celm_ema_beta=0.5,
    celm_seed=42,
    paper_celm_warmup_rounds=None,
    paper_celm_warmup_fraction=0.05,
    paper_celm_steps=200,
    paper_celm_lr=1e-2,
    paper_celm_l2_penalty=1e-3,
    paper_celm_ema_beta=0.5,
):
    del source_name
    train_agg_mode = TRAIN_AGG_MODE if train_agg_mode is None else train_agg_mode
    eval_agg_mode = EVAL_AGG_MODE if eval_agg_mode is None else eval_agg_mode
    prediction_weight_mode = PREDICTION_WEIGHT_MODE if prediction_weight_mode is None else prediction_weight_mode
    print(f"\n{'=' * 20} Starting Federated Training for {model_name} {'=' * 20}")
    print(f"  TRAIN_AGG_MODE={train_agg_mode}\n  EVAL_AGG_MODE={eval_agg_mode}\n  PREDICTION_WEIGHT_MODE={prediction_weight_mode}")
    print(f"  CLIENT_LOCAL_LR={lr}")
    global_model = make_dl_model(model_name, num_classes, device, pretrained)
    normalized_method = server_method.strip().lower()
    if normalized_method not in {"fedavg", "fedadam"}:
        raise ValueError(f"Unsupported server_method='{server_method}'. Use 'FedAvg' or 'FedAdam'.")
    server_optimizer = FedAdamServer(global_model, server_lr, server_beta1, server_beta2, server_eps) if normalized_method == "fedadam" else None
    normalized_weight_method = client_weight_method.strip().lower()
    if normalized_weight_method not in {"sample_size", "uniform", "celm", "paper_celm"}:
        raise ValueError(
            "client_weight_method must be 'sample_size', 'uniform', 'celm', "
            "or 'paper_celm'"
        )
    celm_estimator = None
    if normalized_weight_method == "celm":
        celm_estimator = CELMContributionEstimator(
            model_name, num_classes, len(client_dataloaders), rounds, device,
            warmup_rounds=celm_warmup_rounds,
            warmup_fraction=celm_warmup_fraction, steps=celm_steps,
            lr=celm_lr, l2_penalty=celm_l2_penalty,
            ema_beta=celm_ema_beta, seed=celm_seed,
        )
    elif normalized_weight_method == "paper_celm":
        if clients_per_round != len(client_dataloaders):
            raise ValueError(
                "paper_celm currently requires full client participation to match "
                "the paper's all-client evidence matrix"
            )
        celm_estimator = PaperCELMContributionEstimator(
            model_name, num_classes, len(client_dataloaders), rounds, device,
            warmup_rounds=paper_celm_warmup_rounds,
            warmup_fraction=paper_celm_warmup_fraction,
            steps=paper_celm_steps, lr=paper_celm_lr,
            l2_penalty=paper_celm_l2_penalty,
            ema_beta=paper_celm_ema_beta, seed=celm_seed,
        )
    print(f"  SERVER_METHOD={normalized_method.upper()}")
    print(f"  CLIENT_WEIGHT_METHOD={normalized_weight_method.upper()}")
    if server_optimizer:
        print(f"  FedAdam server settings: lr={server_lr}, beta1={server_beta1}, beta2={server_beta2}")
    global_history = {
        "loss": [], "accuracy": [], "privacy_aggregation": [],
        "client_weighting": [],
    }
    client_histories = {index: {"loss": [], "accuracy": []} for index in range(len(client_dataloaders))}
    last_local_state = [None] * len(client_dataloaders)
    classifier_keys = (
        _classifier_state_keys(global_model)
        if normalized_weight_method == "paper_celm" else ()
    )
    local_classifier_states = [None] * len(client_dataloaders)
    criterion = nn.CrossEntropyLoss()
    for communication_round in range(rounds):
        print(f"\n--- Communication Round {communication_round + 1}/{rounds} ---")
        local_weights, data_lengths = [], []
        selected_models, selected_sizes = [], []
        base_state = _clone_state_dict_cpu(global_model)
        selected = random.sample(range(len(client_dataloaders)), min(clients_per_round, len(client_dataloaders)))
        print(f"  Selected clients this round: {[index + 1 for index in selected]}")
        for client_id in selected:
            train_loader = client_dataloaders[client_id]["train"]
            client_size = len(train_loader.dataset)
            data_lengths.append(client_size)
            local_model = make_dl_model(model_name, num_classes, device, pretrained=False)
            local_initial_state = base_state
            if (
                normalized_weight_method == "paper_celm"
                and communication_round < celm_estimator.warmup_rounds
                and local_classifier_states[client_id] is not None
            ):
                local_initial_state = _state_with_local_classifier(
                    base_state, local_classifier_states[client_id], classifier_keys
                )
            local_model.load_state_dict(local_initial_state)
            optimizer = AdamW(local_model.parameters(), lr=lr)
            print(f"  -> Training on client {client_id + 1} ({client_size} samples)")
            try:
                for _ in range(local_epochs):
                    epoch_train(
                        local_model, train_loader, optimizer, criterion, device,
                        max_grad_norm=1.0,
                    )
                if device.type == "mps":
                    torch.mps.synchronize()
                state = _clone_state_dict_cpu(local_model)
                non_finite = {
                    key: int((~torch.isfinite(value)).sum().item())
                    for key, value in state.items()
                    if value.is_floating_point() and not torch.isfinite(value).all()
                }
                if non_finite:
                    raise FloatingPointError(
                        f"Client {client_id + 1} produced a non-finite model state: {non_finite}"
                    )
                local_weights.append(state)
                last_local_state[client_id] = state
                if (
                    normalized_weight_method == "paper_celm"
                    and communication_round < celm_estimator.warmup_rounds
                ):
                    local_classifier_states[client_id] = {
                        key: state[key].clone() for key in classifier_keys
                    }
                selected_sizes.append(client_size)
                if eval_agg_mode != "weight_avg":
                    selected_models.append(copy.deepcopy(local_model).to(device).eval())
                client_loss, client_accuracy, _, _ = epoch_eval_fed(
                    local_model, server_val_loader, criterion, device,
                )
                client_histories[client_id]["loss"].append(client_loss)
                client_histories[client_id]["accuracy"].append(client_accuracy)
            finally:
                del optimizer
                del local_model
                _release_accelerator_cache(device)
        if train_agg_mode != "weight_avg":
            raise ValueError(f"Unsupported TRAIN_AGG_MODE: {train_agg_mode}. Use 'weight_avg' for FedAvg or FedAdam.")
        if normalized_weight_method == "sample_size":
            aggregation_weights = np.asarray(data_lengths, dtype=np.float64)
            weighting_phase = "sample_size"
        elif normalized_weight_method == "uniform":
            aggregation_weights = np.ones(len(local_weights), dtype=np.float64)
            weighting_phase = "uniform"
        else:
            aggregation_weights, weighting_stats = celm_estimator.weights_for_round(
                communication_round, base_state, local_weights, selected
            )
            weighting_phase = None
        if weighting_phase is not None:
            normalized = aggregation_weights / aggregation_weights.sum()
            weighting_stats = {
                "status": "success", "fallback_used": False,
                "round": communication_round + 1, "phase": weighting_phase,
                "selected_client_ids": [value + 1 for value in selected],
                "client_weights": normalized.tolist(),
                "estimation_seconds": 0.0,
            }
        weighting_stats["method"] = normalized_weight_method
        global_history["client_weighting"].append(weighting_stats)
        averaged_state, privacy_stats = _apply_privacy_if_enabled(
            weighted_average_weights(local_weights, aggregation_weights), local_weights,
            model_name, torch.device("cpu"), privacy_agg_enabled, privacy_agg_target,
            client_weights=aggregation_weights,
            weighting_mode=normalized_weight_method,
        )
        if __import__("config").USE_CKKS_AGGREGATION:
            cfg = __import__("config")
            averaged_state, ckks_stats = ckks_aggregate_selected_params(
                averaged_state, local_weights, aggregation_weights,
                "classification", cfg.CKKS_CLASSIFICATION_TARGET,
                cfg.CKKS_POLY_MODULUS_DEGREE, cfg.CKKS_COEFF_MOD_BIT_SIZES,
                cfg.CKKS_GLOBAL_SCALE,
            )
            privacy_stats["ckks"] = ckks_stats
        privacy_stats["round"] = communication_round + 1
        global_history["privacy_aggregation"].append(privacy_stats)
        if normalized_method == "fedavg":
            global_model.load_state_dict(averaged_state)
        else:
            parameter_names = {name for name, _ in global_model.named_parameters()}
            server_optimizer.step([averaged_state[name].to(device) - base_state[name].to(device) for name, _ in global_model.named_parameters()])
            with torch.no_grad():
                current_state = global_model.state_dict()
                for key in current_state:
                    if key not in parameter_names:
                        current_state[key].copy_(averaged_state[key].to(device=current_state[key].device, dtype=current_state[key].dtype))
        if eval_agg_mode == "weight_avg":
            global_loss, global_accuracy, _, _ = epoch_eval_fed(global_model, server_val_loader, criterion, device)
        elif eval_agg_mode in ("prediction_avg", "confidence_select"):
            weights = np.asarray(selected_sizes, dtype=float) if prediction_weight_mode == "data_size" else None
            if prediction_weight_mode not in ("data_size", "uniform"):
                raise ValueError(f"Unsupported PREDICTION_WEIGHT_MODE: {prediction_weight_mode}")
            _, _, _, global_accuracy, extra = evaluate_client_ensemble_on_loader(
                selected_models, server_val_loader, device, eval_agg_mode, weights if eval_agg_mode == "prediction_avg" else None
            )
            global_loss = np.nan
            if "chosen_clients" in extra:
                counts = np.bincount(extra["chosen_clients"], minlength=len(selected_models))
                print(f"  Confidence-selected usage counts this round: {counts.tolist()}")
        else:
            raise ValueError(f"Unsupported EVAL_AGG_MODE: {eval_agg_mode}")
        global_history["loss"].append(global_loss)
        global_history["accuracy"].append(global_accuracy)
        print(f"Round {communication_round + 1} | Server ({eval_agg_mode} on server-val) acc: {global_accuracy * 100:.2f}%")
    client_cms, client_pairs = _final_client_results(
        global_model, last_local_state, client_dataloaders, test_loader,
        criterion, device, num_classes, model_name,
    )
    return global_model, global_history, client_histories, client_cms, last_local_state, client_pairs
