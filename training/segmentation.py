"""Centralized binary-segmentation training and evaluation primitives."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        ).mean()
        return bce + dice_loss


class WeightedBCEDiceLoss(nn.Module):
    """Positive-pixel-weighted BCE plus soft Dice for sparse crack masks."""

    def __init__(self, positive_class_weight=5.0, smooth=1.0):
        super().__init__()
        positive_class_weight = float(positive_class_weight)
        if positive_class_weight <= 0:
            raise ValueError("Positive class weight must be greater than zero")
        self.register_buffer(
            "positive_class_weight", torch.tensor([positive_class_weight])
        )
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.positive_class_weight.to(
                device=logits.device, dtype=logits.dtype
            ),
        )
        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        ).mean()
        return bce + dice_loss


class FocalDiceLoss(nn.Module):
    """Binary focal loss plus soft Dice loss for sparse crack pixels."""

    def __init__(self, alpha=0.75, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha, self.gamma, self.smooth = float(alpha), float(gamma), float(smooth)

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        probability_true = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_true = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = (alpha_true * (1.0 - probability_true).pow(self.gamma) * bce).mean()
        dims = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return focal + dice


def _soft_boundary_map(values, kernel_size=3):
    padding = int(kernel_size) // 2
    dilation = nn.functional.max_pool2d(
        values, kernel_size=kernel_size, stride=1, padding=padding
    )
    erosion = -nn.functional.max_pool2d(
        -values, kernel_size=kernel_size, stride=1, padding=padding
    )
    return dilation - erosion


class BCEDiceBoundaryLoss(nn.Module):
    """BCE + soft Dice + differentiable morphological boundary agreement."""

    def __init__(self, boundary_weight=0.5, smooth=1.0):
        super().__init__()
        self.base = BCEDiceLoss(smooth=smooth)
        self.boundary_weight = float(boundary_weight)

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        predicted_boundary = _soft_boundary_map(probabilities)
        target_boundary = _soft_boundary_map(targets)
        boundary_loss = nn.functional.l1_loss(
            predicted_boundary, target_boundary
        )
        return self.base(logits, targets) + self.boundary_weight * boundary_loss


class FocalTverskyLoss(nn.Module):
    """Recall-oriented focal Tversky loss for sparse crack masks."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.ndim))
        true_positive = (probabilities * targets).sum(dim=dims)
        false_positive = (probabilities * (1.0 - targets)).sum(dim=dims)
        false_negative = ((1.0 - probabilities) * targets).sum(dim=dims)
        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return (1.0 - tversky).pow(self.gamma).mean()


def make_segmentation_loss(name, positive_class_weight=5.0):
    normalized = str(name).strip().lower().replace("+", "_").replace("-", "_")
    if normalized in {"bce_dice", "bcedice"}:
        return BCEDiceLoss()
    if normalized in {"weighted_bce_dice", "weightedbcedice"}:
        return WeightedBCEDiceLoss(positive_class_weight=positive_class_weight)
    if normalized in {"focal_dice", "focaldice"}:
        return FocalDiceLoss()
    if normalized in {"bce_dice_boundary", "bcediceboundary"}:
        return BCEDiceBoundaryLoss()
    if normalized in {"focal_tversky", "focaltversky"}:
        return FocalTverskyLoss()
    raise ValueError(f"Unsupported segmentation loss: {name}")


def train_segmentation_fedprox_epoch(model, loader, optimizer, criterion, device,
                                     global_parameters, mu):
    """One segmentation epoch with the FedProx proximal objective."""
    model.train()
    running_loss, total = 0.0, 0
    for images, masks in tqdm(loader, desc="Segmentation FedProx", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        base_loss = criterion(model(images), masks)
        proximal = sum(
            (parameter - global_parameters[name].to(parameter.device)).square().sum()
            for name, parameter in model.named_parameters()
        )
        loss = base_loss + float(mu) * 0.5 * proximal
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite segmentation FedProx loss: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        batch = masks.size(0)
        running_loss += float(loss.item()) * batch
        total += batch
    return running_loss / max(total, 1)


def segmentation_metrics_from_arrays(targets, predictions):
    targets = np.asarray(targets, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    if targets.shape != predictions.shape:
        raise ValueError(f"Target/prediction shape mismatch: {targets.shape} vs {predictions.shape}")
    flat_target, flat_prediction = targets.reshape(-1), predictions.reshape(-1)
    tp = int(np.logical_and(flat_target, flat_prediction).sum())
    tn = int(np.logical_and(~flat_target, ~flat_prediction).sum())
    fp = int(np.logical_and(~flat_target, flat_prediction).sum())
    fn = int(np.logical_and(flat_target, ~flat_prediction).sum())
    epsilon = 1e-12
    metrics = {
        "dice": (2 * tp) / max(2 * tp + fp + fn, epsilon),
        "iou": tp / max(tp + fp + fn, epsilon),
        "pixel_precision": tp / max(tp + fp, epsilon),
        "pixel_recall": tp / max(tp + fn, epsilon),
        "specificity": tn / max(tn + fp, epsilon),
        "false_positive_rate": fp / max(fp + tn, epsilon),
        "pixel_accuracy": (tp + tn) / max(tp + tn + fp + fn, epsilon),
        "true_positive_pixels": tp,
        "true_negative_pixels": tn,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
    }
    per_sample_dice, cracked_dice, empty_correct = [], [], []
    for target, prediction in zip(targets, predictions):
        sample_tp = int(np.logical_and(target, prediction).sum())
        sample_fp = int(np.logical_and(~target, prediction).sum())
        sample_fn = int(np.logical_and(target, ~prediction).sum())
        target_empty = not target.any()
        if target_empty:
            score = 1.0 if not prediction.any() else 0.0
            empty_correct.append(score)
        else:
            score = (2 * sample_tp) / max(2 * sample_tp + sample_fp + sample_fn, epsilon)
            cracked_dice.append(score)
        per_sample_dice.append(score)
    metrics.update({
        "mean_sample_dice": float(np.mean(per_sample_dice)),
        "mean_cracked_sample_dice": float(np.mean(cracked_dice)) if cracked_dice else float("nan"),
        "empty_mask_accuracy": float(np.mean(empty_correct)) if empty_correct else float("nan"),
        "sample_count": len(targets),
        "cracked_sample_count": len(cracked_dice),
        "empty_sample_count": len(empty_correct),
    })
    return metrics


def train_segmentation_epoch(
    model, loader, optimizer, criterion, device, freeze_encoder=False
):
    model.train()
    if freeze_encoder:
        # Keep encoder normalization statistics fixed during the warm-up.
        model.encoder.eval()
    running_loss = 0.0
    total = 0
    progress = tqdm(loader, desc="Segmentation train", leave=False)
    for images, masks in progress:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite segmentation loss: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        batch = masks.size(0)
        running_loss += float(loss.item()) * batch
        total += batch
        progress.set_postfix(loss=f"{running_loss / max(total, 1):.4f}")
    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate_segmentation(model, loader, criterion, device, threshold=0.5, return_predictions=False):
    model.eval()
    running_loss = 0.0
    total = 0
    targets, predictions, probabilities = [], [], []
    for images, masks in tqdm(loader, desc="Segmentation evaluate", leave=False):
        images, masks = images.to(device), masks.to(device)
        amp = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
        with amp:
            logits = model(images)
            loss = criterion(logits, masks)
        probability = torch.sigmoid(logits)
        prediction = probability >= float(threshold)
        batch = masks.size(0)
        running_loss += float(loss.item()) * batch
        total += batch
        targets.append(masks.cpu().numpy() > 0.5)
        predictions.append(prediction.cpu().numpy())
        if return_predictions:
            probabilities.append(probability.cpu().numpy())
    target_array = np.concatenate(targets)
    prediction_array = np.concatenate(predictions)
    metrics = segmentation_metrics_from_arrays(target_array, prediction_array)
    metrics["loss"] = running_loss / max(total, 1)
    if return_predictions:
        return metrics, np.concatenate(probabilities), target_array
    return metrics
