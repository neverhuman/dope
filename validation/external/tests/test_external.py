from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run as external
import external_learners as learners


class ExternalEvidenceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[3] / "target" / "external-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.real_train = self.root / "real-train.csv"
        self.holdout = self.root / "sealed-holdout.csv"
        self.synthetic = self.root / "synthetic-train.csv"
        for path, offset in (
            (self.real_train, 0),
            (self.holdout, 13),
            (self.synthetic, 5),
        ):
            with path.open("w", encoding="utf-8") as destination:
                for row in range(48):
                    x0 = ((row + offset) % 47) / 46
                    x1 = ((row * 7 + offset) % 43) / 42
                    x2 = ((row * 11 + offset) % 41) / 40
                    target = int(x0 + 0.4 * x1 + 0.2 * x2 > 0.75)
                    destination.write(f"{x0:.8f},{x1:.8f},{x2:.8f},{target}\n")
        self.artifact = self.root / "kernel.dpk"
        self.artifact.write_bytes(b"fixture artifact")
        self.manifest = self.root / "split.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "format": "dope-external-split",
                    "version": 1,
                    "real_train_sha256": external.sha256_file(self.real_train),
                    "real_holdout_sha256": external.sha256_file(self.holdout),
                }
            ),
            encoding="utf-8",
        )

    def arguments(self, *models):
        return external.parse_args(
            [
                "--real-train",
                str(self.real_train),
                "--real-holdout",
                str(self.holdout),
                "--synthetic-train",
                str(self.synthetic),
                "--artifact",
                str(self.artifact),
                "--split-manifest",
                str(self.manifest),
                "--task",
                "binary",
                "--seed",
                "57721",
                "--models",
                *models,
                "--out",
                str(self.root / "evidence.json"),
            ]
        )

    def test_fixture_measures_same_holdout_and_is_byte_reproducible(self):
        arguments = self.arguments("sklearn_linear", "random_forest")
        self.assertEqual(
            external.main(
                [
                    "--real-train",
                    str(self.real_train),
                    "--real-holdout",
                    str(self.holdout),
                    "--synthetic-train",
                    str(self.synthetic),
                    "--artifact",
                    str(self.artifact),
                    "--split-manifest",
                    str(self.manifest),
                    "--task",
                    "binary",
                    "--seed",
                    "57721",
                    "--models",
                    "sklearn_linear",
                    "random_forest",
                    "--out",
                    str(arguments.out),
                ]
            ),
            0,
        )
        first = arguments.out.read_bytes()
        self.assertEqual(external.run(arguments), json.loads(first))
        evidence = json.loads(first)
        self.assertEqual(evidence["rows"]["sealed_holdout"], 48)
        self.assertEqual(
            evidence["input_hashes"]["artifact_sha256"],
            external.sha256_file(self.artifact),
        )
        self.assertEqual(
            [model["status"] for model in evidence["models"]], ["measured", "measured"]
        )
        self.assertEqual(evidence["supported_exact_library_claims"], ["scikit-learn"])
        for model in evidence["models"]:
            self.assertEqual(model["feature_importance"]["feature_count"], 3)
            self.assertAlmostEqual(
                sum(model["feature_importance"]["real_normalized_shares"]), 1.0
            )
            self.assertEqual(len(model["evidence_sha256"]), 64)
        self.artifact.write_bytes(b"different fixture artifact")
        changed_artifact = external.run(arguments)
        self.assertNotEqual(
            evidence["evidence_sha256"], changed_artifact["evidence_sha256"]
        )
        self.assertNotEqual(
            evidence["models"][0]["evidence_sha256"],
            changed_artifact["models"][0]["evidence_sha256"],
        )

    def test_missing_optional_backend_withholds_claim(self):
        with patch.dict(sys.modules, {"xgboost": None}):
            evidence = external.run(self.arguments("xgboost"))
        self.assertEqual(evidence["models"][0]["status"], "missing_library")
        self.assertEqual(evidence["supported_exact_library_claims"], [])

    def test_failed_forest_withholds_sklearn_library_claim(self):
        original_evaluate = external.evaluate_model

        def evaluate(model_id, *arguments):
            if model_id == "random_forest":
                raise RuntimeError("fixture failure")
            return original_evaluate(model_id, *arguments)

        with patch.object(external, "evaluate_model", side_effect=evaluate):
            evidence = external.run(self.arguments("sklearn_linear", "random_forest"))
        self.assertEqual(evidence["supported_exact_model_claims"], ["sklearn_linear"])
        self.assertEqual(evidence["supported_exact_library_claims"], [])
        self.assertEqual(evidence["withheld_model_claims"], ["random_forest"])

    def test_nonfinite_model_evidence_withholds_claim(self):
        with patch.object(
            external,
            "evaluate_model",
            return_value={
                "status": "measured",
                "model_id": "sklearn_linear",
                "library": "scikit-learn",
                "metrics": {"trtr_loss": float("nan")},
            },
        ):
            evidence = external.run(self.arguments("sklearn_linear"))
        self.assertEqual(evidence["models"][0]["status"], "failed")
        self.assertEqual(evidence["supported_exact_library_claims"], [])

    def test_importance_ties_use_feature_index_and_undefined_rank_stays_missing(self):
        real = external.np.asarray([0.3, 0.3, 0.1])
        synthetic = external.np.asarray([0.3, 0.1, 0.3])
        comparison = learners.compare_importance(real, synthetic, top_k=1)
        self.assertEqual(comparison["real_top_indices"], [0])
        self.assertEqual(comparison["synthetic_top_indices"], [0])
        self.assertEqual(comparison["top_k_jaccard"], 1.0)
        undefined = learners.compare_importance(
            external.np.asarray([0.2, 0.2, 0.2]),
            external.np.asarray([0.3, 0.2, 0.1]),
            top_k=1,
        )
        self.assertIsNone(undefined["spearman"])

    def test_installed_optional_backends_smoke_or_withhold_claim(self):
        evidence = external.run(self.arguments("xgboost", "lightgbm"))
        for model in evidence["models"]:
            self.assertIn(model["status"], ("measured", "missing_library"))
            if model["status"] == "measured":
                self.assertIn(
                    model["library"], evidence["supported_exact_library_claims"]
                )
                self.assertIsNotNone(model["feature_importance"]["top_k_jaccard"])

    def test_regression_ridge_and_forest_use_rmse(self):
        for path, offset in (
            (self.real_train, 0),
            (self.holdout, 13),
            (self.synthetic, 5),
        ):
            with path.open("w", encoding="utf-8") as destination:
                for row in range(48):
                    x0 = ((row + offset) % 47) / 46
                    x1 = ((row * 7 + offset) % 43) / 42
                    x2 = ((row * 11 + offset) % 41) / 40
                    target = 0.1 + 0.5 * x0 + 0.2 * x1 + 0.1 * x2
                    destination.write(f"{x0:.8f},{x1:.8f},{x2:.8f},{target:.8f}\n")
        self.manifest.write_text(
            json.dumps(
                {
                    "format": "dope-external-split",
                    "version": 1,
                    "real_train_sha256": external.sha256_file(self.real_train),
                    "real_holdout_sha256": external.sha256_file(self.holdout),
                }
            ),
            encoding="utf-8",
        )
        arguments = self.arguments("sklearn_linear", "random_forest")
        arguments.task = "regression"
        evidence = external.run(arguments)
        self.assertEqual(evidence["models"][0]["estimator"], "Ridge")
        self.assertEqual(evidence["models"][1]["estimator"], "RandomForestRegressor")
        for model in evidence["models"]:
            self.assertGreaterEqual(model["metrics"]["trtr_loss"], 0.0)
            self.assertIn("regret", model["metrics"])

    def test_split_hash_mismatch_and_header_fail_without_echoing_input(self):
        self.holdout.write_text("source_name,another_name,target\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "split manifest does not match"):
            external.run(self.arguments("sklearn_linear"))
        with self.assertRaises(ValueError) as caught:
            external.load_numeric_csv(self.holdout, "binary")
        self.assertNotIn("source_name", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
