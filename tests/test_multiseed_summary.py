import tempfile
import unittest
from pathlib import Path

import pandas as pd

from reporting.multiseed import write_multiseed_summary


class MultiSeedSummaryTests(unittest.TestCase):
    def test_writes_mean_std_and_unique_seed_count(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "summary.csv"
            output = Path(directory) / "multiseed.csv"
            pd.DataFrame([
                {"experiment": 1, "source": "x", "comparison_variant": "a",
                 "client_weight_method": "celm", "experiment_seed": 42,
                 "accuracy": 0.8},
                {"experiment": 1, "source": "x", "comparison_variant": "a",
                 "client_weight_method": "celm", "experiment_seed": 52,
                 "accuracy": 1.0},
            ]).to_csv(source, index=False)

            write_multiseed_summary(source, output)
            result = pd.read_csv(output).iloc[0]
            self.assertEqual(result["seed_count"], 2)
            self.assertAlmostEqual(result["accuracy_mean"], 0.9)
            self.assertAlmostEqual(result["accuracy_std"], 2 ** 0.5 / 10)


if __name__ == "__main__":
    unittest.main()
