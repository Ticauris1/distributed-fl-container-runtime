"""Optional, fail-closed CKKS vector aggregation shared by FL task types."""

from __future__ import annotations

from time import perf_counter
import numpy as np


class CKKSUnavailableError(RuntimeError):
    pass


def _tenseal():
    try:
        import tenseal as ts
    except ImportError as error:
        raise CKKSUnavailableError(
            "CKKS requires TenSEAL. Install it in the active environment with "
            "`pip install tenseal`; plaintext fallback is disabled."
        ) from error
    return ts


class CKKSAggregator:
    """Coordinator-key CKKS encryption, weighted addition, and decryption."""

    def __init__(self, poly_modulus_degree=8192,
                 coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2**40):
        ts = _tenseal()
        started = perf_counter()
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree,
                                  -1, list(coeff_mod_bit_sizes))
        self.context.global_scale = float(global_scale)
        self.context.generate_galois_keys()
        self.keygen_seconds = perf_counter() - started

    def aggregate(self, vectors, weights):
        vectors = [np.asarray(value, dtype=np.float64).reshape(-1) for value in vectors]
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        if not vectors or len(vectors) != len(weights):
            raise ValueError("CKKS vectors and weights must be non-empty and aligned")
        if len({value.size for value in vectors}) != 1:
            raise ValueError("CKKS vectors must have equal length")
        ts = _tenseal()
        encrypt_started = perf_counter()
        ciphertexts = [ts.ckks_vector(self.context, value.tolist()) for value in vectors]
        encrypt_seconds = perf_counter() - encrypt_started
        aggregation_started = perf_counter()
        aggregate = ciphertexts[0] * float(weights[0])
        for ciphertext, weight in zip(ciphertexts[1:], weights[1:]):
            aggregate += ciphertext * float(weight)
        aggregation_seconds = perf_counter() - aggregation_started
        decrypt_started = perf_counter()
        decoded = np.asarray(aggregate.decrypt(), dtype=np.float64)
        decrypt_seconds = perf_counter() - decrypt_started
        expected = np.average(np.stack(vectors), axis=0, weights=weights)
        error = decoded - expected
        return decoded, {
            "status": "success", "fallback_used": False,
            "keygen_seconds": self.keygen_seconds,
            "encrypt_seconds": encrypt_seconds,
            "aggregation_seconds": aggregation_seconds,
            "decrypt_seconds": decrypt_seconds,
            "max_abs_error": float(np.max(np.abs(error))),
            "mean_abs_error": float(np.mean(np.abs(error))),
            "plaintext_bytes": int(sum(value.nbytes for value in vectors)),
            "ciphertext_bytes": int(sum(len(value.serialize()) for value in ciphertexts)),
        }
