"""Rights and claim gates for the validation-only neural comparison publisher."""

import tempfile
import unittest
from pathlib import Path

from research.benchmark import pilot24_neural_comparison as publisher


class Pilot24NeuralComparisonTests(unittest.TestCase):
    def test_sealed_test_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "prepared-v2/worker/Adult/test.csv"
            target.parent.mkdir(parents=True)
            target.write_text("restricted")
            with self.assertRaises(ValueError):
                publisher.within(root, "prepared-v2/worker", str(target))

    def test_gated_score_cannot_enter_validation_publication(self):
        metric = {"mfs_v2": 0.99, "gate_profile_complete": False,
                  "task": "regression"}
        with self.assertRaises(ValueError):
            publisher.check_metric(metric)

    def test_missing_sample_seed_cannot_form_summary(self):
        rows = [{"dataset": "Adult", "method": "DOPE",
                 "configuration": "q8_symbolic", "size_multiplier": 1,
                 "sample_seed": seed, "catboost_retention": 0.8,
                 "artifact_bytes": 6000, "within_l3_bytes": True,
                 "exact_row_copies": 0} for seed in (101, 211)]
        with self.assertRaises(ValueError):
            publisher.summarize(rows)


if __name__ == "__main__":
    unittest.main()
