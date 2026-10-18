"""End-to-end CPU surrogate campaign dry-run tests for W01 and E03-E07 (0% GPU)."""
from __future__ import annotations

from pathlib import Path
import unittest

from research.transfer.preflight_readiness import (
    CLASSICAL_BASELINE_SLOTS,
    MODERN_AUDITOR_SLOTS,
    run_e03_e04_cpu_calibration_and_m1_dry_run,
    verify_e05_e06_e07_manifest_replay_and_sealed_locks,
    verify_w01_auditor_contracts,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestCampaignReadinessPreflight(unittest.TestCase):
    def test_w01_modern_and_classical_auditor_contracts_and_absent_weights_fail_closed(self):
        report = verify_w01_auditor_contracts(REPO_ROOT)
        self.assertEqual(report["task_id"], "W01")
        self.assertEqual(report["modern_auditors"], MODERN_AUDITOR_SLOTS)
        self.assertEqual(report["classical_baselines"], CLASSICAL_BASELINE_SLOTS)
        self.assertTrue(report["fail_closed_without_external_weights"])
        self.assertEqual(report["fail_closed_absent_weights_slots"], ("kumo-tabular", "tabicl", "mitra"))
        self.assertEqual(report["gpu_utilization_percent"], 0)
        for slot in MODERN_AUDITOR_SLOTS:
            contract = report["modern_contracts"][slot]
            self.assertEqual(contract["lock_status"], "unqualified")
            self.assertEqual(len(contract["recipe_sha256"]), 64)
        for slot in CLASSICAL_BASELINE_SLOTS:
            contract = report["classical_contracts"][slot]
            self.assertEqual(len(contract["recipe_sha256"]), 64)

    def test_e03_and_e04_synthetic_1table_2seed_cpu_calibration_and_m1_confidence_intervals(self):
        report = run_e03_e04_cpu_calibration_and_m1_dry_run(REPO_ROOT)
        self.assertEqual(report["tasks"], ("E03", "E04"))
        self.assertEqual(report["gpu_utilization_percent"], 0)
        self.assertTrue(report["deterministic_repeat_verified"])
        self.assertEqual(len(report["e03_calibration_sha256"]), 64)
        self.assertEqual(len(report["e03_evidence_sha256"]), 64)
        self.assertGreaterEqual(report["e03_oriented_auc"], 0.9)
        self.assertLessEqual(report["e03_auc_bounds"][0], report["e03_auc_bounds"][1])
        self.assertFalse(report["e03_release_authorized"])
        self.assertTrue(report["e03_receipt_authenticates"])
        self.assertFalse(report["e03_receipt_qualifies"])
        self.assertLess(
            report["e04_point_estimates"]["dope"],
            report["e04_point_estimates"]["gaussian_copula_control"],
        )
        dope_lb, dope_est, dope_ub = report["e04_ucb_bounds"]["dope"]
        self.assertLessEqual(dope_lb, dope_est)
        self.assertLessEqual(dope_est, dope_ub)
        self.assertGreaterEqual(report["e04_precommit_pin_count"], 20)

    def test_e05_e06_e07_manifest_hashes_independent_replay_and_sealed_test_lock(self):
        report = verify_e05_e06_e07_manifest_replay_and_sealed_locks(REPO_ROOT)
        self.assertEqual(report["tasks"], ("E05", "E06", "E07"))
        self.assertEqual(report["gpu_utilization_percent"], 0)
        self.assertEqual(len(report["e05_datasets_manifest_sha256"]), 64)
        self.assertGreaterEqual(report["e05_datasets_entry_count"], 100)
        self.assertEqual(report["e05_classical_100_table_lineages"], 100)
        self.assertTrue(report["e05_custody_hold_enforced"])
        self.assertEqual(len(report["e06_independent_replay_receipt_sha256"]), 64)
        self.assertTrue(report["e06_tamper_rejected"])
        self.assertIs(report["e07_official_tests_opened"], False)
        self.assertEqual(
            report["e07_observational_status"],
            "frozen_before_source_acquisition_and_scoring",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
