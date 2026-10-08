import unittest
from unittest.mock import patch

import numpy as np
import torch

from config import PRIVACY_AGG_SCALE
from privacy.integration import (
    PrivacyAggregationError,
    privacy_aware_aggregate_selected_params,
)
from training.federated import _apply_privacy_if_enabled
from training.fedprox import _privacy_average


class PrivacyAggregationTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.client_states = [
            {"head.bias": torch.tensor([0.12525, -0.50025])},
            {"head.bias": torch.tensor([0.87575, 0.25075])},
        ]
        self.averaged_state = {"head.bias": torch.zeros(2)}
        self.client_sizes = [3, 1]

    def test_protected_aggregate_uses_client_sample_weights(self):
        updated, stats = privacy_aware_aggregate_selected_params(
            self.averaged_state,
            self.client_states,
            "test-model",
            self.device,
            "classifier_bias",
            client_weights=self.client_sizes,
        )
        vectors = np.stack([
            state["head.bias"].numpy() for state in self.client_states
        ])
        expected = np.average(vectors, axis=0, weights=self.client_sizes)
        tolerance = 1.0 / (PRIVACY_AGG_SCALE * sum(self.client_sizes))

        np.testing.assert_allclose(
            updated["head.bias"].numpy(), expected, atol=tolerance, rtol=0
        )
        self.assertEqual(stats["status"], "success")
        self.assertEqual(stats["weighting_mode"], "sample_size")
        self.assertFalse(stats["fallback_used"])
        self.assertLessEqual(stats["max_abs_error"], tolerance)
        self.assertGreater(stats["total_seconds"], 0)
        self.assertGreater(stats["protected_bytes"], 0)
        self.assertGreater(stats["size_expansion_ratio"], 0)

    def test_fedavg_privacy_failure_is_fail_closed(self):
        with patch(
            "training.federated.privacy_aware_aggregate_selected_params",
            side_effect=RuntimeError("intentional test failure"),
        ):
            with self.assertRaises(PrivacyAggregationError):
                _apply_privacy_if_enabled(
                    self.averaged_state,
                    self.client_states,
                    "test-model",
                    self.device,
                    privacy_agg_enabled=True,
                    privacy_agg_target="classifier_bias",
                    client_weights=self.client_sizes,
                )

    def test_fedprox_privacy_failure_is_fail_closed(self):
        with patch(
            "training.fedprox.privacy_aware_aggregate_selected_params",
            side_effect=RuntimeError("intentional test failure"),
        ):
            with self.assertRaises(PrivacyAggregationError):
                _privacy_average(
                    self.averaged_state,
                    self.client_states,
                    "test-model",
                    self.device,
                    privacy_agg_enabled=True,
                    privacy_agg_target="classifier_bias",
                    client_weights=self.client_sizes,
                )


if __name__ == "__main__":
    unittest.main()
