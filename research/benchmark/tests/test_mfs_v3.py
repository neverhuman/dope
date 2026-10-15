import json
import shutil
import stat
import tempfile
import time
import unittest
from pathlib import Path

from research.benchmark.representation import (
    artifact_has_cleartext,
    category_code,
    hash_manifest,
    midrank_quantile,
    write_local_lookup,
)
from research.benchmark.mfs_v3_score import PREREQUISITES, evaluate_v3
from research.benchmark.score import CONTRACT, validate_contract


ROOT = Path(__file__).resolve().parents[3]


def _directory_outside_repo(repo: Path) -> Path:
    """A directory the lookup guard accepts. Runner temp can sit inside the checkout."""
    stamp = time.time_ns()
    refusal = None
    for base in (Path("/dev/shm"), Path("/var/tmp"), Path("/tmp"), Path(tempfile.gettempdir())):
        candidate = base / f"dope-mfs-v3-lookup-{stamp}"
        try:
            written = write_local_lookup(candidate, repo, b"probe", [("c", 1, 1)], [])
        except ValueError as exc:
            refusal = exc
            continue
        written.unlink()
        return candidate
    raise AssertionError(refusal or "no directory outside the repository")


def passing():
    return {
        **{key: True for key in PREREQUISITES},
        "encoder": "kumo_tabular_l",
        "distance": 0.2,
        "d_null": 0.5,
        "d_match": 0.4,
        "gap_mitra": -0.1,
        "gap_tabicl": -0.2,
        "exact_row_matches": 0,
        "near_copy_ok": True,
        "cleartext_absent": True,
        "membership_auc": 0.5,
        "attribute_inference_advantage": 0.0,
        "artifact_bytes": 1000,
        "normalizer": "empirical_midrank_quantile_v1",
        "utility_protocol": "train_on_synthetic_score_on_real",
        "utility_auditors": ["kumo_tabular_l", "mitra_v2", "tabicl2"],
        "mfs_components": {
            "utility_transfer": 0.5,
            "driver_fidelity": 0.5,
            "distribution_fidelity": 0.5,
            "structure_fidelity": 0.5,
            "coverage_realism": 0.5,
            "compactness": 0.5,
        },
    }


class MfsV3Tests(unittest.TestCase):
    def test_v2_contract_pin_is_unchanged(self):
        validate_contract(ROOT)
        self.assertEqual(CONTRACT["mfs_v2"]["weights"]["utility_transfer"], 0.30)
        self.assertNotIn("representation_closeness", CONTRACT["mfs_v2"]["weights"])

    def test_geometric_mean_of_a_complete_vector(self):
        report = evaluate_v3(passing())
        self.assertTrue(report["eligible"])
        self.assertAlmostEqual(report["representation_closeness"], 0.5)
        self.assertAlmostEqual(report["score"], 50.0)

    def test_perfect_profile_is_exactly_100(self):
        evidence = passing()
        evidence["distance"] = 0.0
        evidence["mfs_components"] = {key: 1.0 for key in evidence["mfs_components"]}
        self.assertEqual(evaluate_v3(evidence)["score"], 100.0)

    def test_invalid_bytes_fail_closed(self):
        for value in (-5, 0, 1.5, 1000.0, True, None, 10241):
            with self.subTest(value=value):
                evidence = passing()
                evidence["artifact_bytes"] = value
                report = evaluate_v3(evidence)
                self.assertIsNone(report["score"])
                self.assertIn("tier_bytes", report["failed_gates"])
        for value in (1, 10240):
            evidence = passing()
            evidence["artifact_bytes"] = value
            self.assertTrue(evaluate_v3(evidence)["eligible"])

    def test_missing_or_false_prerequisites_fail_closed(self):
        for key in PREREQUISITES:
            for value in (None, False, 1):
                with self.subTest(key=key, value=value):
                    evidence = passing()
                    if value is None:
                        del evidence[key]
                    else:
                        evidence[key] = value
                    report = evaluate_v3(evidence)
                    self.assertIsNone(report["score"])
                    self.assertIn(key, report["failed_gates"])

    def test_attack_metrics_require_valid_probability_domains(self):
        for field, gate, limit in (("membership_auc", "membership_auc", 0.55),
                                   ("attribute_inference_advantage", "attribute_inference", 0.05)):
            for value in (-1.0, limit + 0.001, 1.1, True, None, float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    report = evaluate_v3(dict(passing(), **{field: value}))
                    self.assertIsNone(report["score"])
                    self.assertIn(gate, report["failed_gates"])
            for value in (0.0, limit):
                self.assertTrue(evaluate_v3(dict(passing(), **{field: value}))["eligible"])

    def test_missing_or_invalid_components_are_reported(self):
        for value in (None, -0.1, 1.1, True, float("nan")):
            evidence = passing()
            evidence["mfs_components"]["compactness"] = value
            report = evaluate_v3(evidence)
            self.assertIsNone(report["score"])
            self.assertIn("mfs_components_complete", report["failed_gates"])

    def test_catboost_retention_cannot_fill_tabular_transfer(self):
        evidence = passing()
        evidence["utility_protocol"] = "auditor_retention"
        evidence["utility_auditors"] = ["catboost"]
        report = evaluate_v3(evidence)
        self.assertIsNone(report["score"])
        self.assertIn("tabular_transfer", report["failed_gates"])

    def test_one_exact_row_match_is_null(self):
        evidence = passing()
        evidence["exact_row_matches"] = 1
        report = evaluate_v3(evidence)
        self.assertIsNone(report["score"])
        self.assertIn("exact_row_match", report["failed_gates"])

    def test_cleartext_header_is_null(self):
        evidence = passing()
        evidence["cleartext_absent"] = not artifact_has_cleartext(b"header,age\n1", [b"age"])
        report = evaluate_v3(evidence)
        self.assertIsNone(report["score"])
        self.assertIn("cleartext_absent", report["failed_gates"])

    def test_refused_encoder_is_null(self):
        evidence = passing()
        evidence["encoder"] = "foundation"
        report = evaluate_v3(evidence)
        self.assertIsNone(report["score"])
        self.assertIn("clean_encoder", report["failed_gates"])
        evidence["encoder"] = "tabpfn35"
        self.assertIsNone(evaluate_v3(evidence)["score"])

    def test_zero_match_distance_and_mitra_disagreement_are_null(self):
        evidence = passing()
        evidence["d_match"] = 0.0
        self.assertIsNone(evaluate_v3(evidence)["score"])
        evidence["d_match"] = 0.4
        evidence["gap_mitra"] = 0.2
        report = evaluate_v3(evidence)
        self.assertIsNone(report["representation_closeness"])
        self.assertIsNone(report["score"])

    def test_supplied_closeness_cannot_replace_the_measurement(self):
        evidence = passing()
        evidence["mfs_components"]["representation_closeness"] = 1.0
        evidence["d_match"] = 0.0
        self.assertIsNone(evaluate_v3(evidence)["score"])

    def test_manifest_and_quantile_drop_source_text(self):
        manifest = hash_manifest(b"salt", [("age", 1, 1)], "kumo_tabular_l", 100)
        encoded = json.dumps(manifest)
        self.assertNotIn("age", encoded)
        self.assertEqual(len(manifest["columns"][0]["hash"]), 16)
        real = [1.0, 2.0, 4.0]
        frozen = midrank_quantile(real, [0.0, 1.0, 3.0, 4.0, 9.0])
        self.assertAlmostEqual(frozen[2], 2.0 / 3.0)
        synthetic = [10.0, 11.0, 12.0]
        frozen_synth = midrank_quantile(real, synthetic)
        self.assertNotEqual(frozen_synth, midrank_quantile(synthetic, synthetic))
        self.assertTrue(all(abs(value - (2.5 / 3.0)) < 1e-12 for value in frozen_synth))
        self.assertNotIn("red", f"{category_code(b'salt', 'red')}")

    def test_lookup_file_stays_outside_the_repo(self):
        with self.assertRaises(ValueError):
            write_local_lookup(ROOT / "production", ROOT, b"salt", [("age", 1, 1)], ["red"])
        outside = _directory_outside_repo(ROOT)
        try:
            path = write_local_lookup(outside, ROOT, b"salt", [("age", 1, 1)], ["red"])
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertTrue(artifact_has_cleartext(path.read_bytes(), [b"age", b"red"]))
            fixture = outside / "artifact.csv"
            fixture.write_bytes(b"header,age\n1")
            self.assertTrue(artifact_has_cleartext(fixture.read_bytes(), [b"age"]))
            encoded = json.dumps(hash_manifest(b"salt", [("age", 1, 1)], "kumo_tabular_l", 100))
            self.assertNotIn("age", encoded)
            self.assertNotIn("red", encoded)
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_mfs_v4_evaluation_and_gates(self):
        from research.benchmark.mfs_v3_score import V4_PREREQUISITES, evaluate_v4

        obs = {
            **{key: True for key in V4_PREREQUISITES},
            "encoder": "kumo_tabular_l",
            "distance": 0.2,
            "d_null": 0.5,
            "d_match": 0.4,
            "gap_mitra": -0.1,
            "gap_tabicl": -0.2,
            "exact_row_matches": 0,
            "near_copy_ok": True,
            "cleartext_absent": True,
            "membership_auc": 0.51,
            "attribute_inference_advantage": 0.02,
            "artifact_bytes": 1024,
            "encoded_numbers": 64,
            "real_rows": 1024,
            "real_columns": 8,
            "normalizer": "empirical_midrank_quantile_v1",
            "utility_protocol": "train_on_synthetic_score_on_real",
            "utility_auditors": ["kumo_tabular_l", "catboost", "mitra_v2", "tabicl2"],
            "retention": 0.90,
            "predictor_only_retention": 0.70,
            "independent_marginals_retention": 0.65,
            "real_bootstrap_4n_retention": 0.85,
            "real_importances": [0.40, 0.30, 0.20, 0.10],
            "synthetic_importances": [0.45, 0.25, 0.20, 0.10],
            "real_signs": [1, -1, 1, -1],
            "synthetic_signs": [1, -1, 1, 1],
            "mfs_components": {
                "distribution_fidelity": 0.85,
                "structure_fidelity": 0.80,
                "coverage_realism": 0.75,
            },
        }
        report = evaluate_v4(obs)
        self.assertTrue(report["eligible"])
        self.assertEqual(report["unavailable_terms"], [])
        self.assertAlmostEqual(report["utility_transfer"], 0.935)
        self.assertAlmostEqual(report["driver_fidelity"], 0.95)
        self.assertAlmostEqual(report["control_contrastive_lift"]["contrastive_utility_score"], 0.935)
        self.assertAlmostEqual(report["driver_attribution"]["composite_driver_fidelity"], 0.95)
        self.assertAlmostEqual(report["representation_closeness"], 0.5)
        self.assertIsInstance(report["knowledge_compression"], float)
        self.assertAlmostEqual(
            report["knowledge_compression"],
            report["compression_metrics"]["knowledge_compression_score"],
        )
        self.assertIsNotNone(report["score"])
        self.assertGreater(report["score"], 70.0)

        no_lift = dict(obs, predictor_only_retention=0.92)
        failed = evaluate_v4(no_lift)
        self.assertFalse(failed["eligible"])
        self.assertIsNone(failed["score"])
        self.assertIn("positive_predictor_lift", failed["failed_gates"])

        missing_retention = dict(obs)
        del missing_retention["retention"]
        no_ret_report = evaluate_v4(missing_retention)
        self.assertFalse(no_ret_report["eligible"])
        self.assertIsNone(no_ret_report["score"])
        self.assertIsNone(no_ret_report["knowledge_compression"])
        self.assertIsNone(no_ret_report["compression_metrics"])
        self.assertIn("encoded_numbers", no_ret_report["failed_gates"])
        self.assertIn("control_contrastive_baselines_complete", no_ret_report["failed_gates"])
        self.assertIn("utility_transfer", no_ret_report["unavailable_terms"])
        self.assertIn("knowledge_compression", no_ret_report["unavailable_terms"])

        bad_baseline = dict(obs, predictor_only_retention=float("nan"))
        bad_base_report = evaluate_v4(bad_baseline)
        self.assertFalse(bad_base_report["eligible"])
        self.assertIsNone(bad_base_report["utility_transfer"])
        self.assertIn("control_contrastive_baselines_complete", bad_base_report["failed_gates"])
        self.assertIn("utility_transfer", bad_base_report["unavailable_terms"])

        no_signs = dict(obs, real_signs=None, synthetic_signs=None)
        no_signs_report = evaluate_v4(no_signs)
        self.assertFalse(no_signs_report["eligible"])
        self.assertIsNone(no_signs_report["driver_fidelity"])
        self.assertIn("model_driver_attribution_measured", no_signs_report["failed_gates"])
        self.assertIn("driver_fidelity", no_signs_report["unavailable_terms"])

    def test_representation_refuses_overflow_and_booleans(self):
        from research.benchmark.representation import representation_closeness, unit_half_distance

        self.assertIsNone(midrank_quantile([-1e308, 1e308], [0.0]))
        self.assertIsNone(midrank_quantile([-1e308, 1e308], [0.5e308]))
        self.assertIsNone(midrank_quantile([True, 2.0], [1.0]))
        self.assertIsNone(midrank_quantile([1.0, 2.0], [False]))
        self.assertIsNone(unit_half_distance([1e200, 0.0], [0.0, 1e200]))
        self.assertIsNone(unit_half_distance([True, 0.0], [0.0, 1.0]))
        self.assertIsNone(representation_closeness(True, 0.5, 0.4, -0.1, -0.2))


if __name__ == "__main__":
    unittest.main()
