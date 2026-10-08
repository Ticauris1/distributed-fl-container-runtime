"""Data-free CELM client-contribution weighting for federated aggregation.

This prototype follows the paper's logit-maximization/evidence/EMA stages while
leaving the existing local-training and FedAdam server equations unchanged.
"""

import math
from time import perf_counter

import numpy as np
import torch

from models.factory import make_dl_model, resolve_timm_input_hw


class CELMEstimationError(RuntimeError):
    """Raised when explicitly enabled CELM weighting cannot be completed."""


def evidence_to_weights(global_logits, client_logits, epsilon=1e-12):
    """Convert optimized logits to class-normalized client contributions."""
    global_values = np.asarray(global_logits, dtype=np.float64).reshape(-1)
    client_values = np.asarray(client_logits, dtype=np.float64)
    if client_values.ndim != 2 or client_values.shape[1] != global_values.size:
        raise ValueError("client_logits must have shape [clients, classes]")
    if not np.isfinite(global_values).all() or not np.isfinite(client_values).all():
        raise ValueError("CELM logits must be finite")
    baseline = float(global_values.mean())
    evidence = np.maximum(client_values - baseline, 0.0)
    class_normalized = (evidence + epsilon) / (
        evidence.sum(axis=0, keepdims=True) + epsilon * evidence.shape[0]
    )
    weights = class_normalized.mean(axis=1)
    weights /= weights.sum()
    return weights, evidence, baseline


def paper_evidence_to_weights(global_logits, client_logits, epsilon=1e-12):
    """Apply the paper's denominator-only epsilon in class-wise normalization."""
    global_values = np.asarray(global_logits, dtype=np.float64).reshape(-1)
    client_values = np.asarray(client_logits, dtype=np.float64)
    if client_values.ndim != 2 or client_values.shape[1] != global_values.size:
        raise ValueError("client_logits must have shape [clients, classes]")
    if not np.isfinite(global_values).all() or not np.isfinite(client_values).all():
        raise ValueError("CELM logits must be finite")
    baseline = float(global_values.mean())
    evidence = np.maximum(client_values - baseline, 0.0)
    class_normalized = evidence / (
        evidence.sum(axis=0, keepdims=True) + epsilon
    )
    weights = class_normalized.mean(axis=1)
    total = float(weights.sum())
    if total <= epsilon:
        weights = np.full(client_values.shape[0], 1.0 / client_values.shape[0])
    else:
        weights /= total
    return weights, evidence, baseline


class CELMContributionEstimator:
    """Estimate and retain contribution weights during a bounded warm-up."""

    def __init__(
        self, model_name, num_classes, total_clients, rounds, device,
        warmup_rounds=None, warmup_fraction=0.05, steps=25, lr=1e-2,
        l2_penalty=1e-3, ema_beta=0.5, seed=42,
        initialization_std=0.05, l2_reduction="mean",
    ):
        self.model_name = model_name
        self.num_classes = int(num_classes)
        self.total_clients = int(total_clients)
        self.rounds = int(rounds)
        self.device = device
        self.warmup_rounds = (
            int(warmup_rounds) if warmup_rounds is not None
            else max(1, int(math.ceil(rounds * warmup_fraction)))
        )
        self.steps = int(steps)
        self.lr = float(lr)
        self.l2_penalty = float(l2_penalty)
        self.ema_beta = float(ema_beta)
        self.seed = int(seed)
        self.initialization_std = float(initialization_std)
        if l2_reduction not in {"mean", "sum_per_probe"}:
            raise ValueError("l2_reduction must be 'mean' or 'sum_per_probe'")
        self.l2_reduction = l2_reduction
        self.client_weights = np.full(total_clients, 1.0 / total_clients, dtype=np.float64)
        self.synthetic_inputs = {}

    def _maximize_logits(self, state, identity):
        height, width = resolve_timm_input_hw(self.model_name)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(identity))
        initial = self.synthetic_inputs.get(identity)
        if initial is None:
            initial = torch.randn(
                self.num_classes, 3, height, width, generator=generator,
            ) * self.initialization_std
        inputs = initial.to(self.device).detach().requires_grad_(True)
        model = make_dl_model(
            self.model_name, self.num_classes, self.device, pretrained=False
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam([inputs], lr=self.lr)
        targets = torch.arange(self.num_classes, device=self.device)
        try:
            for _ in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                selected_logits = logits[targets, targets]
                if self.l2_reduction == "sum_per_probe":
                    regularization = inputs.square().flatten(1).sum(dim=1).mean()
                else:
                    regularization = inputs.square().mean()
                loss = -selected_logits.mean() + self.l2_penalty * regularization
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite CELM logit-maximization loss")
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                values = model(inputs)[targets, targets].detach().cpu().numpy()
            self.synthetic_inputs[identity] = inputs.detach().cpu()
            return values
        finally:
            del optimizer, model, inputs

    def _convert_evidence(self, global_logits, client_logits):
        return evidence_to_weights(global_logits, client_logits)

    def weights_for_round(self, round_index, global_state, client_states, client_ids):
        started = perf_counter()
        selected_ids = [int(value) for value in client_ids]
        if len(client_states) != len(selected_ids):
            raise CELMEstimationError("client state and identifier counts do not match")
        try:
            if round_index < self.warmup_rounds:
                global_logits = self._maximize_logits(global_state, self.total_clients)
                client_logits = np.stack([
                    self._maximize_logits(state, client_id)
                    for state, client_id in zip(client_states, selected_ids)
                ])
                instantaneous, evidence, baseline = self._convert_evidence(
                    global_logits, client_logits
                )
                previous = self.client_weights[selected_ids]
                previous /= previous.sum()
                smoothed = self.ema_beta * previous + (1.0 - self.ema_beta) * instantaneous
                smoothed /= smoothed.sum()
                self.client_weights[selected_ids] = smoothed
                selected_weights = smoothed
                phase = "warmup"
            else:
                selected_weights = self.client_weights[selected_ids].copy()
                selected_weights /= selected_weights.sum()
                global_logits = client_logits = evidence = None
                baseline = None
                phase = "frozen"
            return selected_weights, {
                "status": "success", "fallback_used": False,
                "round": int(round_index + 1), "phase": phase,
                "warmup_rounds": self.warmup_rounds,
                "selected_client_ids": [value + 1 for value in selected_ids],
                "client_weights": selected_weights.tolist(),
                "global_logits": None if global_logits is None else global_logits.tolist(),
                "client_logits": None if client_logits is None else client_logits.tolist(),
                "evidence": None if evidence is None else evidence.tolist(),
                "global_logit_baseline": baseline,
                "estimation_seconds": perf_counter() - started,
            }
        except Exception as error:
            raise CELMEstimationError(
                f"CELM contribution estimation failed in round {round_index + 1}"
            ) from error


class PaperCELMContributionEstimator(CELMContributionEstimator):
    """Paper-aligned CELM probe settings kept separate from the prototype."""

    def __init__(
        self, model_name, num_classes, total_clients, rounds, device,
        warmup_rounds=None, warmup_fraction=0.05, steps=200, lr=1e-2,
        l2_penalty=1e-3, ema_beta=0.5, seed=42,
    ):
        super().__init__(
            model_name, num_classes, total_clients, rounds, device,
            warmup_rounds=warmup_rounds, warmup_fraction=warmup_fraction,
            steps=steps, lr=lr, l2_penalty=l2_penalty,
            ema_beta=ema_beta, seed=seed, initialization_std=1.0,
            l2_reduction="sum_per_probe",
        )

    def _convert_evidence(self, global_logits, client_logits):
        return paper_evidence_to_weights(global_logits, client_logits)
