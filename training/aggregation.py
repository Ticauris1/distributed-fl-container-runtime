"""Model-state and prediction aggregation."""

import copy
import numpy as np
import torch


def weighted_average_weights(weights, client_weights):
    """Average model states with validated, normalized client coefficients."""
    if not weights:
        raise ValueError("weights must contain at least one client state")
    coefficients = np.asarray(client_weights, dtype=np.float64).reshape(-1)
    if coefficients.size != len(weights):
        raise ValueError(
            f"client_weights length {coefficients.size} != number of clients {len(weights)}"
        )
    if not np.isfinite(coefficients).all() or np.any(coefficients < 0):
        raise ValueError("client_weights must be finite and non-negative")
    total_weight = float(coefficients.sum())
    if total_weight <= 0:
        raise ValueError("client_weights must have a positive sum")
    coefficients /= total_weight
    average = copy.deepcopy(weights[0])
    for key in average:
        if average[key].dtype in (torch.float16, torch.float32, torch.float64):
            average[key] = torch.zeros_like(weights[0][key], dtype=torch.float32)
            for index, state in enumerate(weights):
                average[key] += state[key] * float(coefficients[index])
        else:
            average[key] = weights[0][key]
    return average


def average_weights(weights, data_lengths):
    """Backward-compatible sample-size-weighted model averaging."""
    return weighted_average_weights(weights, data_lengths)


def aggregate_predictions_avg(probs_by_client, client_weights=None):
    stacked = np.stack(probs_by_client, axis=0)
    client_count = stacked.shape[0]
    if client_weights is None:
        weights = np.full(client_count, 1.0 / client_count, dtype=np.float64)
    else:
        weights = np.asarray(client_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != client_count:
            raise ValueError(f"client_weights length {weights.shape[0]} != number of clients {client_count}")
        weights /= np.clip(weights.sum(), 1e-12, None)
    probabilities = np.tensordot(weights, stacked, axes=(0, 0))
    return np.argmax(probabilities, axis=1), probabilities


def aggregate_predictions_confidence_select(probs_by_client):
    stacked = np.stack(probs_by_client, axis=0)
    chosen_clients = np.argmax(stacked.max(axis=2), axis=0)
    sample_count, class_count = stacked.shape[1:]
    chosen_probabilities = np.zeros((sample_count, class_count), dtype=stacked.dtype)
    for sample in range(sample_count):
        chosen_probabilities[sample] = stacked[chosen_clients[sample], sample]
    return np.argmax(chosen_probabilities, axis=1), chosen_probabilities, chosen_clients


class FedAdamServer:
    """Server Adam update using the existing aggregated-model delta equations."""

    def __init__(self, model, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.model = model
        self.m = [torch.zeros_like(parameter, memory_format=torch.preserve_format) for parameter in model.parameters()]
        self.v = [torch.zeros_like(parameter, memory_format=torch.preserve_format) for parameter in model.parameters()]

    @torch.no_grad()
    def step(self, deltas):
        if len(deltas) != len(self.m):
            raise ValueError(f"Expected {len(self.m)} deltas, received {len(deltas)}.")
        self.t += 1
        for index, (parameter, delta) in enumerate(zip(self.model.parameters(), deltas)):
            delta = delta.to(device=parameter.device, dtype=parameter.dtype)
            self.m[index].mul_(self.beta1).add_(delta, alpha=1.0 - self.beta1)
            self.v[index].mul_(self.beta2).addcmul_(delta, delta, value=1.0 - self.beta2)
            moment = self.m[index] / (1.0 - self.beta1 ** self.t)
            variance = self.v[index] / (1.0 - self.beta2 ** self.t)
            parameter.addcdiv_(moment, torch.sqrt(variance).add_(self.eps), value=self.lr)
