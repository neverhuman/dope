"""predeclare_v2 pins and the v2 runner contract. No benchmark data is read."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.review_fixes import fit_seeds as v1
from research.benchmark.review_fixes import fit_seeds_v2 as v2
from research.benchmark.review_fixes import sealed_once

HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PredeclareV2Tests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads((HERE / "predeclare_v2.json").read_text())

    def test_v1_stays_frozen_at_its_recorded_digest(self):
        supersedes = self.document["supersedes"]
        self.assertEqual(sha(HERE / "predeclare.json"), supersedes["v1_sha256"])
        self.assertEqual(supersedes["v1_sha256"], sealed_once.FREEZE_SHA256)
        self.assertEqual(supersedes["v1_git_blob"], sealed_once.FREEZE_BLOB)
        self.assertEqual(supersedes["v1_commit"], sealed_once.FREEZE_COMMIT)

    def test_evaluator_digest_matches_the_code(self):
        evaluator = self.document["evaluator"]
        self.assertEqual(sha(v1.PILOT_PATH), evaluator["utility_sha256"])
        self.assertEqual(sha(HERE.parent / "expanded_validation_metrics.py"), evaluator["diagnostics_sha256"])
        self.assertEqual(evaluator["auditor_seed"], 1729)

    def test_runner_contract_matches_the_historical_command(self):
        contract = v2.load_contract()
        self.assertEqual(contract["fit_seeds"], (11, 23, 37, 53, 71))
        self.assertEqual(contract["seed_flag"], "--seed")
        spec = contract["compile"]
        self.assertEqual(
            (spec["candidate"], spec["tier"], spec["neural_target_weight"],
             spec["neural_structural_penalty"], spec["deadline_seconds"]),
            (v1.CANDIDATE, v1.TIER, v1.NEURAL_TARGET_WEIGHT, v1.NEURAL_STRUCTURAL_PENALTY, v1.DEADLINE_SECONDS))
        self.assertEqual(self.document["methods"]["primary"]["TabSyn"]["config_sha256"],
                         json.loads((HERE / "predeclare.json").read_text())["tabsyn"]["config_sha256"])

    def test_margins_and_families_are_declared(self):
        self.assertEqual(self.document["equivalence"]["margin_retention"], 0.02)
        families = self.document["tests"]["holm_families_per_size"]
        self.assertTrue(families["primary"].startswith("12 tests"))
        self.assertTrue(self.document["tests"]["sizes_are_separate_families"])
        self.assertFalse(self.document["claims"]["formal_dp"])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.contract = v2.load_contract()

    def test_compile_uses_the_historical_seed_flag(self):
        command = v2.compile_command(Path("/bin/k"), Path("/w"), "regression", Path("/m.dpk"), 37, self.contract)
        self.assertIn("--seed", command)
        self.assertNotIn("--fit-seed", command)
        self.assertEqual(command[command.index("--seed") + 1], "37")

    def test_product_default_drops_the_research_profile(self):
        headline = v2.child_env("/lib", self.contract["arms"]["headline"])
        default = v2.child_env("/lib", self.contract["arms"]["product_default"])
        self.assertEqual(headline["DOPE_RESEARCH_TARGET_PROFILE"], "features12_steps2048")
        self.assertNotIn("DOPE_RESEARCH_TARGET_PROFILE", default)
        self.assertEqual(default["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_plan_is_seed_major(self):
        plan = v2.seed_major([("a", "A"), ("b", "B")], (11, 23))
        self.assertEqual([item[:2] for item in plan], [(11, "a"), (11, "b"), (23, "a"), (23, "b")])

    def test_a_changed_seed_list_refuses(self):
        document = json.loads((HERE / "predeclare_v2.json").read_text())
        document["dope"]["fit_seeds"] = [23, 37, 53, 71]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predeclare_v2.json"
            path.write_text(json.dumps(document))
            with self.assertRaises(SystemExit):
                v2.load_contract(path)

    def test_wrong_host_refuses_before_any_read(self):
        args = type("Args", (), {"host": "not-this-host"})()
        with patch.object(v2, "load_contract") as load:
            with self.assertRaises(SystemExit):
                v2.run(args)
            load.assert_not_called()

    def test_v1_sealed_run_refuses_once_v2_exists(self):
        with self.assertRaisesRegex(SystemExit, "supersedes"):
            sealed_once._refuse_superseded()
        with tempfile.TemporaryDirectory() as tmp:
            sealed_once._refuse_superseded(Path(tmp) / "absent.json")


if __name__ == "__main__":
    unittest.main()
