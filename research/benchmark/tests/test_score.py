import json
import tempfile
import unittest
from pathlib import Path

from research.benchmark.score import CONTRACT, evaluate, validate_contract


ROOT = Path(__file__).resolve().parents[3]


class ScoreTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "target").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "target")
        self.root = Path(self.directory.name)
        (self.root / "model.json").write_text(json.dumps({"mean": 0.5}))
        self.bytes = (self.root / "model.json").stat().st_size
        self.evidence = {key: value for key, value in CONTRACT["generator_gates"].items()
                         if key != "feature_importance_min_informative_features"}
        self.evidence.update({
            "informative_feature_count": 3, "artifact_bytes": self.bytes,
            "source_rows_required": False, "artifact_sampling_verified": True,
            "privacy_attack_complete": True, "real_vs_real_control_complete": True,
            "feature_importance_complete": True,
            "required_size_cells_complete": True, "metric_implementations_locked": True,
            "mfs_components": {key: 0.8 for key in CONTRACT["mfs_v2"]["weights"]},
        })

    def tearDown(self):
        self.directory.cleanup()

    def score(self):
        return evaluate(self.evidence, self.root, ["model.json"], "l3")

    def test_contract_matches_production(self):
        validate_contract(ROOT)

    def test_complete_evidence_scores(self):
        report = self.score()
        self.assertTrue(report["eligible"])
        self.assertAlmostEqual(report["score"], 80.0001, places=4)

    def test_strong_but_leaky_is_null(self):
        self.evidence["mfs_components"] = {key: 1 for key in CONTRACT["mfs_v2"]["weights"]}
        self.evidence["membership_auc_max"] = 0.9
        self.assertIsNone(self.score()["score"])

    def test_row_copy_and_missing_evidence_are_null(self):
        self.evidence["exact_copies_max"] = 1
        self.assertIn("exact_copies_max", self.score()["failed_gates"])
        self.evidence["exact_copies_max"] = 0
        del self.evidence["privacy_attack_complete"]
        self.assertIsNone(self.score()["score"])

    def test_byte_reconciliation_and_source_rows(self):
        self.evidence["artifact_bytes"] -= 1
        self.assertIn("artifact_inventory", self.score()["failed_gates"])
        self.evidence["artifact_bytes"] += 1
        self.evidence["source_rows_required"] = True
        self.assertIn("source_rows_absent", self.score()["failed_gates"])
        (self.root / "unlisted.bin").write_bytes(b"hidden")
        self.assertIn("artifact_inventory", self.score()["failed_gates"])


if __name__ == "__main__":
    unittest.main()
