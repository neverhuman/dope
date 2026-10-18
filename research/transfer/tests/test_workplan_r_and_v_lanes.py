"""Deterministic tracked acceptance checks for the R and V workplan lanes."""
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class RustPilotAcceptanceTests(unittest.TestCase):
    def _cargo_test(self, name: str) -> None:
        subprocess.run(
            ["cargo", "test", "--locked", "--offline", "--lib", name, "--", "--nocapture"],
            cwd=ROOT,
            check=True,
        )

    def test_r01_full_artifact_bitpacking_pilot(self):
        self._cargo_test("preregistered_full_artifact_bitpacking_pilot")

    def test_r02_precision_allocation_pilot(self):
        self._cargo_test("preregistered_precision_pilot_compares_fixed_and_equal_nominal_cost_allocations")

    def test_r05_residual_profiles_and_fit_support(self):
        self._cargo_test("all_preregistered_residual_profiles_round_trip_and_enforce_fit_support")

    def test_r09_residual_study_fixtures(self):
        self._cargo_test("compiler::residual_study_fixtures")


class V00RegressionGateTests(unittest.TestCase):
    def test_v00_frozen_evidence_and_cpu_readiness_gates(self):
        subprocess.run(
            ["python3", "-B", "ops/ci/check-frozen-evidence.py"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            ["bash", "ops/ci/test-cpu-tool-readiness.sh"],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
