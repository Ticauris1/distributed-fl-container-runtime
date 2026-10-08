"""CELM-style data-free contribution weighting for binary segmentation."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import torch

from models.segmentation import make_segmentation_model
from training.celm import (
    CELMEstimationError,
    evidence_to_weights,
    paper_evidence_to_weights,
)


class SegmentationCELMContributionEstimator:
    """Adapt CELM logit maximization to a spatial binary crack-logit map."""

    def __init__(self, model_name, encoder_name, image_size, total_clients, rounds,
                 device, warmup_rounds=None, warmup_fraction=0.05, steps=25,
                 lr=1e-2, l2_penalty=1e-3, ema_beta=0.5, seed=42):
        self.model_name, self.encoder_name = model_name, encoder_name
        self.image_size, self.total_clients = int(image_size), int(total_clients)
        self.device = device
        self.warmup_rounds = int(warmup_rounds) if warmup_rounds is not None else max(1, math.ceil(rounds * warmup_fraction))
        self.steps, self.lr = int(steps), float(lr)
        self.l2_penalty, self.ema_beta, self.seed = float(l2_penalty), float(ema_beta), int(seed)
        self.client_weights = np.full(total_clients, 1.0 / total_clients, dtype=np.float64)
        self.synthetic_inputs = {}

    def _maximize_crack_logit(self, state, identity):
        generator = torch.Generator(device="cpu").manual_seed(self.seed + int(identity))
        initial = self.synthetic_inputs.get(identity)
        if initial is None:
            initial = torch.randn(1, 3, self.image_size, self.image_size, generator=generator) * 0.05
        inputs = initial.to(self.device).detach().requires_grad_(True)
        model = make_segmentation_model(self.model_name, self.device, self.encoder_name, encoder_weights=None)
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam([inputs], lr=self.lr)
        try:
            for _ in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                mean_crack_logit = model(inputs).mean()
                loss = -mean_crack_logit + self.l2_penalty * inputs.square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite segmentation CELM loss")
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                value = float(model(inputs).mean().detach().cpu())
            self.synthetic_inputs[identity] = inputs.detach().cpu()
            return np.asarray([value], dtype=np.float64)
        finally:
            del optimizer, model, inputs

    def weights_for_round(self, round_index, global_state, client_states, client_ids):
        started = perf_counter()
        selected_ids = [int(value) for value in client_ids]
        if len(client_states) != len(selected_ids):
            raise CELMEstimationError("client state and identifier counts do not match")
        try:
            if round_index < self.warmup_rounds:
                global_logits = self._maximize_crack_logit(global_state, self.total_clients)
                client_logits = np.stack([self._maximize_crack_logit(state, client_id) for state, client_id in zip(client_states, selected_ids)])
                instantaneous, evidence, baseline = evidence_to_weights(global_logits, client_logits)
                previous = self.client_weights[selected_ids]
                previous /= previous.sum()
                selected_weights = self.ema_beta * previous + (1.0 - self.ema_beta) * instantaneous
                selected_weights /= selected_weights.sum()
                self.client_weights[selected_ids] = selected_weights
                phase = "warmup"
            else:
                selected_weights = self.client_weights[selected_ids].copy()
                selected_weights /= selected_weights.sum()
                global_logits = client_logits = evidence = None
                baseline, phase = None, "frozen"
            return selected_weights, {
                "phase": phase, "warmup_rounds": self.warmup_rounds,
                "client_weights": selected_weights.tolist(),
                "global_logits": None if global_logits is None else global_logits.tolist(),
                "client_logits": None if client_logits is None else client_logits.tolist(),
                "evidence": None if evidence is None else evidence.tolist(),
                "global_logit_baseline": baseline,
                "estimation_seconds": perf_counter() - started,
            }
        except Exception as error:
            raise CELMEstimationError(f"Segmentation CELM failed in round {round_index + 1}") from error


class PaperSegmentationCELMContributionEstimator:
    """Paper-aligned binary segmentation CELM using crack/background evidence."""

    def __init__(self, model_name, encoder_name, image_size, total_clients, rounds,
                 device, warmup_rounds=None, warmup_fraction=0.05, steps=200,
                 lr=1e-2, l2_penalty=1e-3, ema_beta=0.5, seed=42):
        self.model_name, self.encoder_name = model_name, encoder_name
        self.image_size, self.total_clients = int(image_size), int(total_clients)
        self.device = device
        self.warmup_rounds = (
            int(warmup_rounds) if warmup_rounds is not None
            else max(1, math.ceil(rounds * warmup_fraction))
        )
        self.steps, self.lr = int(steps), float(lr)
        self.l2_penalty = float(l2_penalty)
        self.ema_beta, self.seed = float(ema_beta), int(seed)
        self.client_weights = np.full(
            total_clients, 1.0 / total_clients, dtype=np.float64
        )
        self.synthetic_inputs = {}

    def _maximize_binary_evidence(self, state, identity):
        generator = torch.Generator(device="cpu").manual_seed(self.seed + int(identity))
        initial = self.synthetic_inputs.get(identity)
        if initial is None:
            # Probe 0 maximizes crack evidence; probe 1 maximizes background
            # evidence by maximizing the negative crack logit.
            initial = torch.randn(
                2, 3, self.image_size, self.image_size, generator=generator
            )
        inputs = initial.to(self.device).detach().requires_grad_(True)
        model = make_segmentation_model(
            self.model_name, self.device, self.encoder_name, encoder_weights=None
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam([inputs], lr=self.lr)
        try:
            for _ in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                evidence_scores = torch.stack((logits[0].mean(), -logits[1].mean()))
                regularization = inputs.square().flatten(1).sum(dim=1)
                loss = -(evidence_scores - self.l2_penalty * regularization).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "non-finite paper segmentation CELM loss"
                    )
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                logits = model(inputs)
                values = torch.stack(
                    (logits[0].mean(), -logits[1].mean())
                ).detach().cpu().numpy()
            self.synthetic_inputs[identity] = inputs.detach().cpu()
            return values
        finally:
            del optimizer, model, inputs

    def weights_for_round(self, round_index, global_state, client_states, client_ids):
        started = perf_counter()
        selected_ids = [int(value) for value in client_ids]
        if len(client_states) != len(selected_ids):
            raise CELMEstimationError("client state and identifier counts do not match")
        try:
            if round_index < self.warmup_rounds:
                global_logits = self._maximize_binary_evidence(
                    global_state, self.total_clients
                )
                client_logits = np.stack([
                    self._maximize_binary_evidence(state, client_id)
                    for state, client_id in zip(client_states, selected_ids)
                ])
                instantaneous, evidence, baseline = paper_evidence_to_weights(
                    global_logits, client_logits
                )
                previous = self.client_weights[selected_ids]
                previous /= previous.sum()
                selected_weights = (
                    self.ema_beta * previous
                    + (1.0 - self.ema_beta) * instantaneous
                )
                selected_weights /= selected_weights.sum()
                self.client_weights[selected_ids] = selected_weights
                phase = "warmup"
            else:
                selected_weights = self.client_weights[selected_ids].copy()
                selected_weights /= selected_weights.sum()
                global_logits = client_logits = evidence = None
                baseline, phase = None, "frozen"
            return selected_weights, {
                "status": "success", "fallback_used": False,
                "phase": phase, "warmup_rounds": self.warmup_rounds,
                "evidence_channels": ["crack", "background"],
                "client_weights": selected_weights.tolist(),
                "global_logits": None if global_logits is None else global_logits.tolist(),
                "client_logits": None if client_logits is None else client_logits.tolist(),
                "evidence": None if evidence is None else evidence.tolist(),
                "global_logit_baseline": baseline,
                "estimation_seconds": perf_counter() - started,
            }
        except Exception as error:
            raise CELMEstimationError(
                f"Paper segmentation CELM failed in round {round_index + 1}"
            ) from error
