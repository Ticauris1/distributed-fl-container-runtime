"""Shared deep and client-ensemble evaluation helpers."""

from contextlib import nullcontext
import numpy as np
import torch
from tqdm.auto import tqdm

from .aggregation import aggregate_predictions_avg, aggregate_predictions_confidence_select


def clone_state_dict(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def evaluate_server(model, loader, criterion, device, use_amp=True):
    is_cuda = device.type == "cuda"
    amp_context = torch.amp.autocast("cuda") if use_amp and is_cuda else nullcontext()
    model.eval()
    total = correct = 0
    running_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)
        with amp_context:
            logits = model(inputs)
            loss = criterion(logits, labels)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(1) == labels).sum().item()
        total += batch_size
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def epoch_eval(model, loader, criterion, device, use_amp=True):
    model.eval()
    is_cuda = device.type == "cuda"
    amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp and is_cuda else nullcontext()
    running_loss = running_correct = total = 0.0
    progress = tqdm(loader, desc="Validating", leave=False)
    for inputs, labels in progress:
        inputs = inputs.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)
        with amp_context:
            logits = model(inputs)
            loss = criterion(logits, labels)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(1) == labels).sum().item()
        total += batch_size
        progress.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{running_correct / max(total, 1):.4f}")
    return running_loss / max(total, 1), running_correct / max(total, 1)


def epoch_eval_fed(model, loader, criterion, device, use_amp=True):
    model.eval()
    is_cuda = device.type == "cuda"
    amp_context = torch.amp.autocast("cuda") if use_amp and is_cuda else nullcontext()
    running_loss = running_correct = total = 0.0
    all_probabilities, all_labels = [], []
    progress = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=is_cuda)
            labels = labels.to(device, non_blocking=is_cuda)
            with amp_context:
                logits = model(inputs)
                if not torch.isfinite(logits).all():
                    invalid = int((~torch.isfinite(logits)).sum().item())
                    raise FloatingPointError(
                        f"Evaluation produced {invalid} non-finite logits."
                    )
                loss = criterion(logits, labels)
            batch_probabilities = logits.softmax(dim=1)
            if not torch.isfinite(batch_probabilities).all():
                invalid = int((~torch.isfinite(batch_probabilities)).sum().item())
                raise FloatingPointError(
                    f"Evaluation produced {invalid} non-finite probabilities."
                )
            all_probabilities.append(batch_probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(1) == labels).sum().item()
            total += batch_size
            progress.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{running_correct / max(total, 1):.4f}")
    return (running_loss / max(total, 1), running_correct / max(total, 1),
            np.concatenate(all_labels), np.concatenate(all_probabilities))


@torch.no_grad()
def collect_client_probabilities(client_models, loader, device, use_amp=True):
    probabilities_by_client = []
    reference_labels = None
    is_cuda = device.type == "cuda"
    for model in client_models:
        model.eval()
        probabilities, labels = [], []
        for inputs, batch_labels in loader:
            inputs = inputs.to(device, non_blocking=is_cuda)
            amp_context = torch.amp.autocast("cuda") if use_amp and is_cuda else nullcontext()
            with amp_context:
                probabilities.append(model(inputs).softmax(dim=1).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
        probabilities_by_client.append(np.concatenate(probabilities))
        if reference_labels is None:
            reference_labels = np.concatenate(labels)
    return reference_labels, probabilities_by_client


def evaluate_client_ensemble_on_loader(client_models, loader, device, agg_mode="prediction_avg", client_weights=None, use_amp=True):
    labels, client_probabilities = collect_client_probabilities(client_models, loader, device, use_amp)
    if agg_mode == "prediction_avg":
        predictions, probabilities = aggregate_predictions_avg(client_probabilities, client_weights)
        extra = {}
    elif agg_mode == "confidence_select":
        predictions, probabilities, selected = aggregate_predictions_confidence_select(client_probabilities)
        extra = {"chosen_clients": selected}
    else:
        raise ValueError(f"Unsupported agg_mode: {agg_mode}")
    return labels, predictions, probabilities, float((predictions == labels).mean()), extra


_clone_state_dict = clone_state_dict
_evaluate_server = evaluate_server
