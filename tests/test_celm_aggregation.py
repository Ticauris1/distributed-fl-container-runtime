import unittest

import numpy as np
import torch
import torch.nn as nn

from training.aggregation import weighted_average_weights
from training.celm import (
    PaperCELMContributionEstimator,
    evidence_to_weights,
    paper_evidence_to_weights,
)
from training.federated import _classifier_state_keys, _state_with_local_classifier
from training.segmentation_celm import PaperSegmentationCELMContributionEstimator


class CELMAggregationTests(unittest.TestCase):
    def test_evidence_is_normalized_per_class_then_across_clients(self):
        global_logits = np.array([1.0, 1.0])
        client_logits = np.array([[3.0, 1.0], [1.0, 5.0]])
        weights, evidence, baseline = evidence_to_weights(
            global_logits, client_logits
        )

        self.assertAlmostEqual(baseline, 1.0)
        np.testing.assert_allclose(evidence, [[2.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-10)
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_zero_evidence_produces_uniform_weights(self):
        weights, _, _ = evidence_to_weights(
            np.array([2.0, 2.0]), np.array([[1.0, 1.0], [0.0, 0.0]])
        )
        np.testing.assert_allclose(weights, [0.5, 0.5])

    def test_paper_normalization_does_not_credit_a_zero_evidence_class(self):
        weights, _, _ = paper_evidence_to_weights(
            np.array([1.0, 1.0]), np.array([[3.0, 1.0], [1.0, 1.0]])
        )
        np.testing.assert_allclose(weights, [1.0, 0.0], atol=1e-10)

    def test_weighted_model_average_accepts_celm_coefficients(self):
        states = [
            {"weight": torch.tensor([1.0, 3.0]), "counter": torch.tensor(1)},
            {"weight": torch.tensor([5.0, 7.0]), "counter": torch.tensor(2)},
        ]
        result = weighted_average_weights(states, [0.25, 0.75])
        torch.testing.assert_close(result["weight"], torch.tensor([4.0, 6.0]))
        self.assertEqual(result["counter"].item(), 1)

    def test_invalid_weights_fail_closed(self):
        states = [{"weight": torch.tensor([1.0])}, {"weight": torch.tensor([2.0])}]
        with self.assertRaises(ValueError):
            weighted_average_weights(states, [float("nan"), 1.0])

    def test_paper_celm_defaults_match_reported_probe_settings(self):
        estimator = PaperCELMContributionEstimator(
            "densenet121", 4, 5, 100, torch.device("cpu"), seed=42
        )
        self.assertEqual(estimator.steps, 200)
        self.assertEqual(estimator.warmup_rounds, 5)
        self.assertEqual(estimator.initialization_std, 1.0)
        self.assertEqual(estimator.l2_reduction, "sum_per_probe")
        self.assertAlmostEqual(estimator.lr, 1e-2)
        self.assertAlmostEqual(estimator.l2_penalty, 1e-3)
        self.assertAlmostEqual(estimator.ema_beta, 0.5)

    def test_local_classifier_overlay_preserves_global_backbone(self):
        model = nn.Module()
        model.backbone = nn.Linear(2, 2)
        model.head = nn.Linear(2, 2)
        keys = _classifier_state_keys(model)
        base = {key: value.detach().clone() for key, value in model.state_dict().items()}
        retained = {key: torch.full_like(base[key], 7) for key in keys}
        merged = _state_with_local_classifier(base, retained, keys)
        for key in base:
            if key in keys:
                torch.testing.assert_close(merged[key], retained[key])
            else:
                torch.testing.assert_close(merged[key], base[key])

    def test_paper_segmentation_celm_uses_binary_evidence_and_five_percent_warmup(self):
        estimator = PaperSegmentationCELMContributionEstimator(
            "unetplusplus", "efficientnet_b3", 384, 2, 100,
            torch.device("cpu"), seed=42,
        )
        self.assertEqual(estimator.steps, 200)
        self.assertEqual(estimator.warmup_rounds, 5)
        self.assertAlmostEqual(estimator.lr, 1e-2)
        self.assertAlmostEqual(estimator.l2_penalty, 1e-3)


if __name__ == "__main__":
    unittest.main()
