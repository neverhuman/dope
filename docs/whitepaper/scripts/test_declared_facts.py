"""Figure scripts read the interval level, byte cap and full-panel floor from their sources."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import declared_facts as D  # noqa: E402
import v2_figures  # noqa: E402
from research.benchmark.review_fixes.v2_panel import STRONGEST_MIN_LINEAGES  # noqa: E402


class DeclaredFactsTests(unittest.TestCase):
    def setUp(self):
        self.predeclare = D.read(D.PREDECLARE)
        self.contract = D.read(D.CONTRACT)

    def test_values_come_from_the_predeclaration_and_contract(self):
        level = D.interval_level(self.predeclare)
        self.assertIn(f"percentile {level}% interval", self.predeclare["estimands"]["uncertainty"])
        self.assertEqual(D.byte_cap(self.contract), self.contract["tier_byte_limits"]["l3"])

    def test_missing_level_and_invalid_cap_are_refused(self):
        unstated = copy.deepcopy(self.predeclare)
        unstated["estimands"]["uncertainty"] = "Nested bootstrap."
        with self.assertRaisesRegex(ValueError, "interval level"):
            D.interval_level(unstated)
        for cap in (0, -1, "10240", 10240.0, True, None):
            contract = copy.deepcopy(self.contract)
            contract["tier_byte_limits"]["l3"] = cap
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "byte cap"):
                D.byte_cap(contract)

    def test_read_uses_the_supplied_root(self):
        local = D.REPO / "target" / "declared-facts-tests"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            root = Path(directory)
            drifted = copy.deepcopy(self.predeclare)
            drifted["estimands"]["uncertainty"] = drifted["estimands"]["uncertainty"].replace(
                f"percentile {D.interval_level(self.predeclare)}% interval", "percentile 90% interval")
            (root / D.PREDECLARE).parent.mkdir(parents=True)
            (root / D.PREDECLARE).write_text(json.dumps(drifted))
            self.assertEqual(D.interval_level(D.read(D.PREDECLARE, root)), 90)

    def test_v2_figures_read_instead_of_typing(self):
        self.assertEqual(v2_figures.CAP, D.byte_cap(self.contract))
        self.assertEqual(v2_figures.INTERVAL_LEVEL, D.interval_level(self.predeclare))
        self.assertEqual(v2_figures.FULL_PANEL, STRONGEST_MIN_LINEAGES)
        source = Path(v2_figures.__file__).read_text()
        for literal in ("10240", "10,240", "95%", "FULL_PANEL = 90"):
            self.assertNotIn(literal, source, f"v2_figures types {literal!r} instead of reading it")


if __name__ == "__main__":
    unittest.main()
