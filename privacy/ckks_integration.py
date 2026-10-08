"""Selected-parameter CKKS aggregation for classification and segmentation."""

from __future__ import annotations

import numpy as np
import torch

from privacy.ckks import CKKSAggregator
from privacy.parameter_utils import get_classifier_param_keys


def _keys(state, task, target):
    if task == "classification":
        return get_classifier_param_keys(state, target=target)
    if task == "segmentation":
        head_keys = [
            key for key, value in state.items()
            if "segmentation_head" in key and torch.is_floating_point(value)
        ]
        if not head_keys:
            raise ValueError("Could not identify floating-point segmentation-head parameters")
        if target == "segmentation_head_bias":
            candidates = [
                key for key in head_keys
                if key.endswith(".bias") and state[key].ndim == 1
            ]
            if not candidates:
                raise ValueError("Could not identify segmentation-head bias")
            return [candidates[-1]]
        if target == "segmentation_head":
            # Preserve state_dict order so packing and reconstruction are stable.
            return head_keys
    raise ValueError(f"Unsupported CKKS task/target: {task}/{target}")


def ckks_aggregate_selected_params(plaintext_average, client_states, weights,
                                   task, target, poly_modulus_degree=8192,
                                   coeff_mod_bit_sizes=(60, 40, 40, 60),
                                   global_scale=2**40):
    if not client_states:
        raise ValueError("CKKS aggregation requires client states")
    keys = _keys(plaintext_average, task, target)
    metadata, vectors = [], []
    for state in client_states:
        parts = []
        for key in keys:
            array = state[key].detach().cpu().numpy()
            if not metadata:
                metadata.append((key, array.shape, array.size))
            parts.append(array.reshape(-1))
        vectors.append(np.concatenate(parts).astype(np.float64))
    aggregator = CKKSAggregator(poly_modulus_degree, coeff_mod_bit_sizes, global_scale)
    decoded, stats = aggregator.aggregate(vectors, weights)
    updated = {key: value.clone() for key, value in plaintext_average.items()}
    offset = 0
    for key, shape, count in metadata:
        chunk = decoded[offset:offset + count].reshape(shape)
        updated[key] = torch.as_tensor(chunk, dtype=updated[key].dtype)
        offset += count
    stats.update({
        "enabled": True,
        "task": task,
        "target": target,
        "param_keys": keys,
        "protected_tensor_count": len(keys),
        "protected_parameter_count": int(decoded.size),
        "vector_dim": int(decoded.size),
    })
    return updated, stats
