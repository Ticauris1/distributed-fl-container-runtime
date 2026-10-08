"""Supplied-threshold hybrid inference and metrics."""

import time
import numpy as np #type: ignore
import torch #type: ignore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score #type: ignore
import torch.nn.functional as F #type: ignore

def prepare_lightweight_inputs(inputs):
    if inputs.shape[-2:] == (224, 224):
        return inputs

    return F.interpolate(
        inputs,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

@torch.no_grad()
def predict_probs_dl(model, inputs):
    logits = model(inputs)
    if not torch.isfinite(logits).all():
        invalid = int((~torch.isfinite(logits)).sum().item())
        raise FloatingPointError(
            f"Inference produced {invalid} non-finite logits."
        )
    probabilities = torch.softmax(logits, dim=1)
    if not torch.isfinite(probabilities).all():
        invalid = int((~torch.isfinite(probabilities)).sum().item())
        raise FloatingPointError(
            f"Inference produced {invalid} non-finite probabilities."
        )
    return probabilities, probabilities.argmax(dim=1), None


def compute_eval_metrics(y_true, y_pred, probabilities):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc_score": np.nan,
    }
    try:
        if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] > 2:
            metrics["auc_score"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted"))
        elif probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] == 2:
            metrics["auc_score"] = float(roc_auc_score(y_true, probabilities[:, 1]))
    except Exception:
        pass
    return metrics


def compute_binary_eval_metrics(y_true, y_pred, probabilities):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_score": np.nan,
    }
    try:
        if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] == 2:
            metrics["auc_score"] = float(roc_auc_score(y_true, probabilities[:, 1]))
    except Exception:
        pass
    return metrics


def collapse_to_binary_labels(labels, positive_classes):
    positive = set(positive_classes)
    return np.array([1 if label in positive else 0 for label in np.asarray(labels)], dtype=np.int64)


@torch.no_grad()
def evaluate_heavy_only_on_loader(heavy_model, loader, device):
    heavy_model.eval()
    labels, predictions, probabilities = [], [], []
    start_time = time.time()
    for inputs, batch_labels in loader:
        batch_probabilities, batch_predictions, _ = predict_probs_dl(heavy_model, inputs.to(device))
        probabilities.append(batch_probabilities.cpu().numpy())
        predictions.append(batch_predictions.cpu().numpy())
        labels.append(batch_labels.numpy())
    elapsed = time.time() - start_time
    labels = np.concatenate(labels)
    return {"y_true": labels, "y_pred": np.concatenate(predictions), "probs": np.concatenate(probabilities),
            "avg_time_per_sample": elapsed / max(len(labels), 1), "lightweight_count": 0,
            "heavy_count": len(labels), "heavy_fraction": 1.0}


@torch.no_grad()
def evaluate_hybrid_on_loader(lightweight_model, heavy_model, loader, device, threshold_high):
    lightweight_model.eval()
    heavy_model.eval()
    labels, predictions, probabilities = [], [], []
    lightweight_count = heavy_count = 0
    start_time = time.time()
    for inputs, batch_labels in loader:
        inputs = inputs.to(device)
        light_probabilities, light_predictions, _ = predict_probs_dl(lightweight_model, inputs)
        use_light = light_probabilities.max(dim=1).values >= threshold_high
        use_heavy = ~use_light
        final_probabilities = light_probabilities.clone()
        final_predictions = light_predictions.clone()
        if use_heavy.any():
            heavy_probabilities, heavy_predictions, _ = predict_probs_dl(heavy_model, inputs[use_heavy])
            final_probabilities[use_heavy] = heavy_probabilities
            final_predictions[use_heavy] = heavy_predictions
        lightweight_count += int(use_light.sum().item())
        heavy_count += int(use_heavy.sum().item())
        probabilities.append(final_probabilities.cpu().numpy())
        predictions.append(final_predictions.cpu().numpy())
        labels.append(batch_labels.numpy())
    elapsed = time.time() - start_time
    labels = np.concatenate(labels)
    total = max(len(labels), 1)
    return {"y_true": labels, "y_pred": np.concatenate(predictions), "probs": np.concatenate(probabilities),
            "avg_time_per_sample": elapsed / total, "lightweight_count": lightweight_count,
            "heavy_count": heavy_count, "heavy_fraction": heavy_count / total}


@torch.no_grad()
def evaluate_hybrid_on_loader_binary(lightweight_model, heavy_model, loader, device, threshold_high, positive_classes):
    lightweight_model.eval()
    heavy_model.eval()
    positive_classes = list(positive_classes)
    labels, predictions, probabilities = [], [], []
    lightweight_count = heavy_count = 0
    start_time = time.time()
    for inputs, batch_labels in loader:
        inputs = inputs.to(device)
        light_inputs = prepare_lightweight_inputs(inputs)
        light_full, _, _ = predict_probs_dl(lightweight_model, light_inputs)
        positive_probability = light_full[:, positive_classes].sum(dim=1)
        light_binary = torch.stack([1.0 - positive_probability, positive_probability], dim=1)
        light_predictions = (positive_probability >= 0.5).long()
        use_light = torch.maximum(positive_probability, 1.0 - positive_probability) >= threshold_high
        use_heavy = ~use_light
        final_probabilities, final_predictions = light_binary.clone(), light_predictions.clone()
        if use_heavy.any():
            heavy_full, _, _ = predict_probs_dl(heavy_model, inputs[use_heavy])
            heavy_positive = heavy_full[:, positive_classes].sum(dim=1)
            final_probabilities[use_heavy] = torch.stack([1.0 - heavy_positive, heavy_positive], dim=1)
            final_predictions[use_heavy] = (heavy_positive >= 0.5).long()
        lightweight_count += int(use_light.sum().item())
        heavy_count += int(use_heavy.sum().item())
        probabilities.append(final_probabilities.cpu().numpy())
        predictions.append(final_predictions.cpu().numpy())
        labels.append(collapse_to_binary_labels(batch_labels.numpy(), positive_classes))
    elapsed = time.time() - start_time
    labels = np.concatenate(labels)
    total = max(len(labels), 1)
    return {"y_true": labels, "y_pred": np.concatenate(predictions), "probs": np.concatenate(probabilities),
            "avg_time_per_sample": elapsed / total, "lightweight_count": lightweight_count,
            "heavy_count": heavy_count, "heavy_fraction": heavy_count / total}


def evaluate_threshold(threshold_high, lightweight_model, heavy_model, loader, device, binary_mode=False, positive_classes=None):
    if binary_mode:
        output = evaluate_hybrid_on_loader_binary(lightweight_model, heavy_model, loader, device, float(threshold_high), positive_classes)
        metrics = compute_binary_eval_metrics(output["y_true"], output["y_pred"], output["probs"])
    else:
        output = evaluate_hybrid_on_loader(lightweight_model, heavy_model, loader, device, float(threshold_high))
        metrics = compute_eval_metrics(output["y_true"], output["y_pred"], output["probs"])
    return {"threshold_high": float(threshold_high), **metrics,
            "avg_time_per_sample": output["avg_time_per_sample"],
            "lightweight_count": output["lightweight_count"], "heavy_count": output["heavy_count"],
            "heavy_fraction": output["heavy_fraction"]}


_compute_eval_metrics = compute_eval_metrics
_compute_binary_eval_metrics = compute_binary_eval_metrics
_hybrid_eval_at_threshold = evaluate_threshold
