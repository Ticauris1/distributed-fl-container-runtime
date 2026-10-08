import csv
import json
from pathlib import Path
import tempfile
import unittest

from reporting.tableau_export import export_tableau_data
from tableau.tabpy_functions import confidence_interval_95, paired_delta, rolling_mean


class TableauExportTests(unittest.TestCase):
    def test_exports_metrics_rounds_and_client_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            run = root / "segmentation" / "federated" / "study" / "variant" / "seed_42"
            run.mkdir(parents=True)
            (run / "test_metrics.json").write_text(json.dumps({
                "training_type": "federated", "variant_name": "variant",
                "model": "unetplusplus", "encoder": "efficientnet-b3",
                "seed": 42, "dice": 0.8, "client_distribution": "iid",
                "clients": 2, "method": "FedAvg", "weighting": "paper_celm",
            }))
            with (run / "round_history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "round", "selected_clients", "selected_client_samples",
                    "aggregation_weights", "validation_dice",
                ])
                writer.writeheader()
                writer.writerow({
                    "round": 1, "selected_clients": "1;2",
                    "selected_client_samples": "10;10",
                    "aggregation_weights": "0.4;0.6", "validation_dice": 0.7,
                })
            output = root / "tableau"
            counts = export_tableau_data(root, output)
            self.assertEqual(counts["segmentation_experiments.csv"], 1)
            self.assertEqual(counts["federated_rounds.csv"], 1)
            self.assertEqual(counts["client_weights.csv"], 2)
            self.assertTrue((output / "data_quality.csv").is_file())

    def test_tabpy_calculations_preserve_tableau_row_counts(self):
        self.assertEqual(paired_delta([1, 2], [2, 5]), [1.0, 3.0])
        self.assertEqual(rolling_mean([1, 2, 3], 2), [1.0, 1.5, 2.5])
        self.assertEqual(len(confidence_interval_95([1, 2, 3])), 3)


if __name__ == "__main__":
    unittest.main()
