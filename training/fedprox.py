"""FedProx local objective and federated orchestration."""

import copy
from contextlib import nullcontext
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from config import PRIVACY_AGG_TARGET, USE_PRIVACY_AWARE_AGGREGATION
from models.factory import make_dl_model
from privacy.integration import PrivacyAggregationError, privacy_aware_aggregate_selected_params
from .aggregation import average_weights
from .evaluation import epoch_eval_fed


def _clone_state_dict_cpu(model):
    """Clone a model state to CPU so completed clients do not occupy accelerator memory."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _release_accelerator_cache(device):
    """Release cached allocator blocks after a temporary client model is discarded."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def local_train_fedprox(
    model, train_loader, criterion, optimizer, device, global_parameters, mu,
    local_epochs=1, use_amp=True, max_grad_norm=1.0,
):
    """Train locally with a differentiable proximal penalty."""
    is_cuda = device.type == "cuda"
    amp_enabled = use_amp and is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if is_cuda else None
    model.train()
    total = correct = 0
    running_loss = 0.0
    for _ in range(local_epochs):
        progress = tqdm(train_loader, leave=False, desc="FedProx local")
        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=is_cuda)
            labels = labels.to(device, non_blocking=is_cuda)
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with amp_context:
                logits = model(inputs)
                if not torch.isfinite(logits).all():
                    invalid = int((~torch.isfinite(logits)).sum().item())
                    raise FloatingPointError(
                        f"FedProx produced {invalid} non-finite logits before the optimizer step."
                    )
                base_loss = criterion(logits, labels)
                proximal_term = 0
                for name, parameter in model.named_parameters():
                    global_parameter = global_parameters[name].to(device=parameter.device, dtype=parameter.dtype)
                    proximal_term = proximal_term + ((parameter - global_parameter) ** 2).sum()
                loss = base_loss + (mu / 2.0) * proximal_term
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite FedProx loss: "
                    f"base={base_loss.detach().item()}, "
                    f"proximal={proximal_term.detach().item()}, mu={mu}."
                )
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(1) == labels).sum().item()
            total += batch_size
            progress.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.4f}")
    return running_loss / max(total, 1), correct / max(total, 1)


def _privacy_average(
    averaged_state, local_weights, model_name, device,
    privacy_agg_enabled=None, privacy_agg_target=None, client_weights=None,
):
    enabled = (
        USE_PRIVACY_AWARE_AGGREGATION
        if privacy_agg_enabled is None else privacy_agg_enabled
    )
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
            client_weights=client_weights,
        )
        print(f"[PRIVACY-AGG] target={stats['privacy_agg_target']} | clients={stats['num_clients']} | vector_dim={stats['vector_dim']} | keys={stats['param_keys']}")
        return state, stats
    except Exception as error:
        print(f"[PRIVACY-AGG][error] Aggregation failed; aborting run: {error}")
        raise PrivacyAggregationError(
            f"Privacy-aware aggregation failed for target={target}; "
            "plaintext fallback is disabled."
        ) from error


def federated_prox_training(
    model_name, source_name, client_dataloaders, server_val_loader, test_loader,
    num_classes, device, rounds=10, local_epochs=1, clients_per_round=None,
    lr=1e-3, mu=0.001, pretrained=True, make_model_fn=None, build_model_fn=None,
    criterion=None,
    max_grad_norm=1.0,
    privacy_agg_enabled=None,
    privacy_agg_target=None,
):
    del source_name
    if not client_dataloaders:
        raise AssertionError("No client dataloaders provided.")
    print(
        f"\n{'=' * 20} Starting FedProx Training for {model_name} {'=' * 20}\n"
        f"  FEDPROX_MU={mu}\n  CLIENT_LOCAL_LR={lr}"
    )
    if make_model_fn is None:
        make_model_fn = lambda name, count: make_dl_model(name, count, device, pretrained)
    if make_model_fn is not None:
        global_model = make_model_fn(model_name, num_classes)
    elif build_model_fn is not None:
        global_model = build_model_fn(model_name=model_name, num_classes=num_classes)
    else:
        raise ValueError("Provide make_model_fn or build_model_fn to create the model.")
    global_model.to(device)
    criterion = criterion or torch.nn.CrossEntropyLoss()
    rng = np.random.RandomState(42)
    client_count = len(client_dataloaders)
    clients_per_round = client_count if not clients_per_round or clients_per_round <= 0 else min(clients_per_round, client_count)
    global_history = {"loss": [], "accuracy": [], "privacy_aggregation": []}
    client_histories = {index: {"loss": [], "accuracy": []} for index in range(client_count)}
    last_local_state = [None] * client_count
    for round_number in range(1, rounds + 1):
        print(f"\n[Round {round_number}/{rounds}] FedProx (mu={mu})")
        selected = sorted(rng.choice(client_count, size=clients_per_round, replace=False).tolist())
        print(f"  Selected clients this round: {[index + 1 for index in selected]}")
        # The global model is unchanged during all local updates in this round.
        # Detached parameter references avoid another full accelerator-side copy.
        global_parameters = {
            name: parameter.detach()
            for name, parameter in global_model.named_parameters()
        }
        local_weights, local_sizes = [], []
        for client_id in selected:
            client_model = copy.deepcopy(global_model).to(device)
            optimizer = torch.optim.AdamW(client_model.parameters(), lr=lr)
            train_loader = client_dataloaders[client_id]["train"]
            client_size = len(train_loader.dataset)
            print(f"  -> FedProx training on client {client_id + 1} ({client_size} samples)")
            try:
                loss, accuracy = local_train_fedprox(
                    client_model, train_loader, criterion, optimizer, device,
                    global_parameters, mu, local_epochs,
                    max_grad_norm=max_grad_norm,
                )
                # Surface asynchronous MPS failures here instead of continuing
                # with a corrupted model and failing later during evaluation.
                if device.type == "mps":
                    torch.mps.synchronize()
                state = _clone_state_dict_cpu(client_model)
                non_finite = {
                    key: int((~torch.isfinite(value)).sum().item())
                    for key, value in state.items()
                    if value.is_floating_point() and not torch.isfinite(value).all()
                }
                if non_finite:
                    raise FloatingPointError(
                        f"Client {client_id + 1} produced a non-finite model state: {non_finite}"
                    )
                client_histories[client_id]["loss"].append(loss)
                client_histories[client_id]["accuracy"].append(accuracy)
                local_weights.append(state)
                local_sizes.append(client_size)
                last_local_state[client_id] = state
            finally:
                del optimizer
                del client_model
                _release_accelerator_cache(device)
        averaged, privacy_stats = _privacy_average(
            average_weights(local_weights, local_sizes), local_weights,
            model_name, torch.device("cpu"), privacy_agg_enabled, privacy_agg_target,
            client_weights=local_sizes,
        )
        privacy_stats["round"] = round_number
        global_history["privacy_aggregation"].append(privacy_stats)
        non_finite = {
            key: int((~torch.isfinite(value)).sum().item())
            for key, value in averaged.items()
            if value.is_floating_point() and not torch.isfinite(value).all()
        }
        if non_finite:
            raise FloatingPointError(
                f"Round {round_number} produced a non-finite aggregate: {non_finite}"
            )
        global_model.load_state_dict(averaged, strict=True)
        validation_loss, validation_accuracy, _, _ = epoch_eval_fed(global_model, server_val_loader, criterion, device)
        global_history["loss"].append(validation_loss)
        global_history["accuracy"].append(validation_accuracy)
        print(f"  [Server] val_loss={validation_loss:.4f}  val_acc={validation_accuracy:.4f}")
    final_client_cms, client_pairs = {}, []
    for client_id, loaders in enumerate(client_dataloaders):
        state = last_local_state[client_id]
        if state is None:
            model = global_model
            owns_model = False
        else:
            model = make_model_fn(model_name, num_classes)
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            owns_model = True
        eval_loader = loaders.get("test") or loaders.get("val") or loaders["train"]
        _, _, labels, probabilities = epoch_eval_fed(model, eval_loader, criterion, device)
        final_client_cms[client_id] = confusion_matrix(labels, np.argmax(probabilities, axis=1), labels=list(range(num_classes)))
        pair_loader = loaders.get("test") or test_loader
        _, _, pair_labels, pair_probabilities = epoch_eval_fed(model, pair_loader, criterion, device)
        client_pairs.append((pair_labels, pair_probabilities))
        if owns_model:
            del model
            _release_accelerator_cache(device)
    return global_model, global_history, client_histories, final_client_cms, last_local_state, client_pairs


_local_train_fedprox = local_train_fedprox
