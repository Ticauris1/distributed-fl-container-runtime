"""Privacy-aware aggregation integration for federated training."""

from time import perf_counter

import numpy as np

from config import PRIVACY_AGG_MODULUS, PRIVACY_AGG_SCALE, PRIVACY_AGG_SEED
from .aggregation import PrivacyAwareAggregation
from .parameter_utils import (
    fixed_point_decode, fixed_point_encode,
    flatten_selected_params, get_classifier_param_keys, restore_selected_params_from_flat,
)


class PrivacyAggregationError(RuntimeError):
    """Raised when an enabled privacy aggregation round cannot complete safely."""


def _validated_client_weights(client_states, client_weights, weighting_mode=None):
    if client_weights is None:
        weights = np.ones(len(client_states), dtype=np.float64)
        mode = "uniform"
    else:
        weights = np.asarray(client_weights, dtype=np.float64).reshape(-1)
        mode = weighting_mode or "sample_size"
    if len(weights) != len(client_states):
        raise ValueError(
            f"client_weights length {len(weights)} does not match "
            f"client_states length {len(client_states)}"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("client_weights must be finite, non-negative, and have a positive sum")
    return weights, mode


def privacy_aware_aggregate_selected_params(
    global_state, client_states, model_name, device, target="classifier_bias",
    client_weights=None, weighting_mode=None,
):
    total_started = perf_counter()
    if not client_states:
        raise ValueError("client_states is empty")
    weights, weighting_mode = _validated_client_weights(
        client_states, client_weights, weighting_mode
    )
    total_weight = float(weights.sum())

    extraction_started = perf_counter()
    param_keys = get_classifier_param_keys(global_state, model_name, target)
    client_flats, metadata = [], None
    for state in client_states:
        flat, current_metadata = flatten_selected_params(state, param_keys)
        client_flats.append(flat)
        metadata = metadata or current_metadata
    extraction_seconds = perf_counter() - extraction_started
    vector_dim = len(client_flats[0])
    if vector_dim > 256:
        raise ValueError(f"Selected vector_dim={vector_dim} is too large for this matrix-based scheme. Use classifier_bias or a smaller subset.")

    weighted_flats = [vector * weight for vector, weight in zip(client_flats, weights)]
    scaled_abs_bound = np.sum(
        np.abs(np.rint(np.stack(weighted_flats) * PRIVACY_AGG_SCALE)), axis=0
    )
    if scaled_abs_bound.size and scaled_abs_bound.max() >= PRIVACY_AGG_MODULUS // 2:
        raise OverflowError(
            "Weighted fixed-point aggregate may wrap modulo PRIVACY_AGG_MODULUS; "
            "increase the modulus or reduce the scale/weights."
        )

    setup_started = perf_counter()
    scheme = PrivacyAwareAggregation(vector_dim, PRIVACY_AGG_MODULUS, PRIVACY_AGG_SEED)
    keys = [scheme.generate_user_keys(index) for index in range(len(client_flats))]
    setup_seconds = perf_counter() - setup_started

    encode_encrypt_started = perf_counter()
    ciphertexts = [
        scheme.encrypt_reading(
            fixed_point_encode(vector, PRIVACY_AGG_SCALE, PRIVACY_AGG_MODULUS), key
        )
        for vector, key in zip(weighted_flats, keys)
    ]
    encode_encrypt_seconds = perf_counter() - encode_encrypt_started

    aggregation_started = perf_counter()
    encrypted_aggregate = scheme.aggregate_ciphertexts(ciphertexts)
    aggregation_seconds = perf_counter() - aggregation_started

    decrypt_started = perf_counter()
    decoded_sum = fixed_point_decode(
        scheme.decrypt_full(encrypted_aggregate),
        PRIVACY_AGG_SCALE,
        PRIVACY_AGG_MODULUS,
    )
    average = decoded_sum / total_weight
    decrypt_decode_seconds = perf_counter() - decrypt_started

    expected_average = np.average(np.stack(client_flats), axis=0, weights=weights)
    error = average - expected_average
    error_norm = float(np.linalg.norm(error))
    reference_norm = float(np.linalg.norm(expected_average))

    restore_started = perf_counter()
    updated = restore_selected_params_from_flat(global_state, average, metadata, device)
    restore_seconds = perf_counter() - restore_started

    plaintext_bytes_per_client = [
        int(sum(state[key].numel() * state[key].element_size() for key in param_keys))
        for state in client_states
    ]
    protected_bytes_per_client = [int(ciphertext.nbytes) for ciphertext in ciphertexts]
    plaintext_bytes = int(sum(plaintext_bytes_per_client))
    protected_bytes = int(sum(protected_bytes_per_client))

    return updated, {
        "enabled": True,
        "status": "success",
        "fallback_used": False,
        "privacy_agg_target": target,
        "num_clients": len(client_flats),
        "vector_dim": vector_dim,
        "param_keys": param_keys,
        "weighting_mode": weighting_mode,
        "client_weights": [float(value) for value in weights],
        "total_weight": total_weight,
        "scale": int(PRIVACY_AGG_SCALE),
        "modulus": int(PRIVACY_AGG_MODULUS),
        "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
        "mean_abs_error": float(np.mean(np.abs(error))) if error.size else 0.0,
        "rmse": float(np.sqrt(np.mean(error ** 2))) if error.size else 0.0,
        "relative_l2_error": error_norm / reference_norm if reference_norm else error_norm,
        "extraction_seconds": extraction_seconds,
        "setup_keygen_seconds": setup_seconds,
        "encode_encrypt_seconds": encode_encrypt_seconds,
        "aggregation_seconds": aggregation_seconds,
        "decrypt_decode_seconds": decrypt_decode_seconds,
        "restore_seconds": restore_seconds,
        "total_seconds": perf_counter() - total_started,
        "plaintext_bytes_per_client": plaintext_bytes_per_client,
        "protected_bytes_per_client": protected_bytes_per_client,
        "plaintext_bytes": plaintext_bytes,
        "protected_bytes": protected_bytes,
        "size_expansion_ratio": protected_bytes / plaintext_bytes if plaintext_bytes else None,
        "size_measurement": "raw selected tensors versus raw NumPy ciphertext arrays",
    }
