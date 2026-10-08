"""Tests for CKKS parameter selection and reconstructed aggregation."""

import numpy as np
import torch

import privacy.ckks_integration as integration


class _PlaintextTestAggregator:
    """Deterministic stand-in that exercises packing and reconstruction."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def aggregate(self, vectors, weights):
        decoded = np.average(np.stack(vectors), axis=0, weights=weights)
        return decoded, {
            "status": "success",
            "fallback_used": False,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "plaintext_bytes": sum(value.nbytes for value in vectors),
            "ciphertext_bytes": 0,
        }


def _state(weight, bias, encoder):
    return {
        "encoder.block.weight": torch.tensor([encoder], dtype=torch.float32),
        "segmentation_head.0.weight": torch.tensor(weight, dtype=torch.float32).reshape(1, 1, 1, 2),
        "segmentation_head.0.bias": torch.tensor([bias], dtype=torch.float32),
    }


def test_full_segmentation_head_protects_weight_and_bias(monkeypatch):
    monkeypatch.setattr(integration, "CKKSAggregator", _PlaintextTestAggregator)
    clients = [_state([1.0, 3.0], 5.0, 10.0), _state([5.0, 7.0], 9.0, 20.0)]
    plaintext_average = _state([3.0, 5.0], 7.0, 15.0)

    updated, stats = integration.ckks_aggregate_selected_params(
        plaintext_average, clients, [1.0, 3.0], "segmentation", "segmentation_head"
    )

    assert torch.allclose(updated["segmentation_head.0.weight"], torch.tensor([[[[4.0, 6.0]]]]))
    assert torch.allclose(updated["segmentation_head.0.bias"], torch.tensor([8.0]))
    # Non-head tensors retain the existing plaintext aggregate at this stage.
    assert torch.equal(updated["encoder.block.weight"], plaintext_average["encoder.block.weight"])
    assert stats["protected_tensor_count"] == 2
    assert stats["protected_parameter_count"] == 3
    assert stats["param_keys"] == [
        "segmentation_head.0.weight",
        "segmentation_head.0.bias",
    ]


def test_bias_only_mode_remains_available(monkeypatch):
    monkeypatch.setattr(integration, "CKKSAggregator", _PlaintextTestAggregator)
    clients = [_state([1.0, 3.0], 5.0, 10.0), _state([5.0, 7.0], 9.0, 20.0)]
    plaintext_average = _state([3.0, 5.0], 7.0, 15.0)

    updated, stats = integration.ckks_aggregate_selected_params(
        plaintext_average, clients, [1.0, 3.0], "segmentation", "segmentation_head_bias"
    )

    assert torch.equal(updated["segmentation_head.0.weight"], plaintext_average["segmentation_head.0.weight"])
    assert torch.allclose(updated["segmentation_head.0.bias"], torch.tensor([8.0]))
    assert stats["protected_parameter_count"] == 1
