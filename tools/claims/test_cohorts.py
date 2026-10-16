"""FOLLOW-L1b/U3: regenerated public claims must retain ledger-derived cohorts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/whitepaper/scripts"))
import claim_registry as C
from tools.claims import cohorts, render


class CohortProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "paper/generated/headline-results.json").read_text())
        cls.status = json.loads((ROOT / "evidence/status.json").read_text())

    def public_failures(self, candidate):
        # Regenerate the card too: output drift alone cannot prove provenance.
        readme = (ROOT / "README.md").read_text()
        start, end = readme.index(render.BEGIN), readme.index(render.END) + len(render.END)
        card = render.text_outputs(candidate, self.status)["paper/generated/readme-result-card.md"].decode().strip()
        readme = readme[:start] + card + readme[end:]
        original = Path.read_text
        def read(path, *args, **kwargs):
            if path == ROOT / "paper/generated/headline-results.json":
                return json.dumps(candidate)
            if path == ROOT / "README.md":
                return readme
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", read):
            return C.public_failures(check_render=False)

    def test_prepared_denominator_drift_is_refused_even_after_card_regeneration(self):
        candidate = copy.deepcopy(self.data)
        candidate["results"][0]["cohort"]["prepared_lineages"] += 1
        self.assertIn("Dope: cohort prepared_lineages differs from v2 ledger",
                      self.public_failures(candidate))

    def test_false_full_cohort_flag_is_refused(self):
        candidate = copy.deepcopy(self.data)
        candidate["results"][0]["cohort"]["complete"] = False
        self.assertIn("Dope: cohort complete differs from v2 lineage rule",
                      self.public_failures(candidate))

    def test_small_cohort_cannot_be_relabelled_complete(self):
        candidate = copy.deepcopy(self.data)
        row = next(row for row in candidate["results"] if row["id"] == "Forest")
        row["cohort"]["complete"] = True
        self.assertIn("Forest: cohort complete differs from v2 lineage rule",
                      self.public_failures(candidate))

    def test_renderer_refuses_free_denominators_and_complete_flags(self):
        target = ROOT / "target/claims-tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            path = Path(directory) / "headline-results.json"
            for field, value in (("prepared_lineages", 101), ("complete", False)):
                candidate = copy.deepcopy(self.data)
                candidate["results"][0]["cohort"][field] = value
                path.write_text(json.dumps(candidate))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "cohort " + field):
                    render.load_results(path)

    def test_cohort_field_types_cannot_exploit_numeric_bool_equality(self):
        for field, value in (("prepared_lineages", 100.0), ("complete", 1)):
            candidate = copy.deepcopy(self.data)
            candidate["results"][0]["cohort"][field] = value
            with self.subTest(field=field):
                self.assertTrue(any("cohort " + field in failure for failure in self.public_failures(candidate)))

    def test_refresh_derives_fields_and_preserves_every_other_input(self):
        target = ROOT / "target/claims-tests"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            path = Path(directory) / "headline-results.json"
            candidate = copy.deepcopy(self.data)
            candidate["results"][0]["cohort"].update(prepared_lineages=1, complete=False)
            path.write_text(json.dumps(candidate))
            self.assertEqual(render.load_results(path, derive_cohorts=True), self.data)

    def test_derivation_uses_unique_ledger_datasets_and_pinned_threshold_boundary(self):
        target = ROOT / "target/claims-tests"
        target.mkdir(parents=True, exist_ok=True)
        floor = C.STRONGEST_MIN_LINEAGES
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory)
            ledger, source = root / cohorts.LEDGER, root / cohorts.SOURCE
            ledger.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            rows = [{"dataset": f"fixture-{i}"} for i in range(floor + 2)]
            ledger.write_text("\n".join(json.dumps(row) for row in rows + rows))
            source.write_text(json.dumps({"arms": [
                {"code": "full", "level": {"n": floor}},
                {"code": "small", "level": {"n": floor - 1}},
            ]}))
            data = {"citable": False, "fixture": True, "results": [
                {"source": {"path": cohorts.SOURCE, "arm": code},
                 "cohort": {"observed_lineages": floor + 2, "prepared_lineages": 0, "complete": None}}
                for code in ("full", "small")]}
            cohorts.refresh_cohorts(data, root)
            self.assertEqual([row["cohort"]["prepared_lineages"] for row in data["results"]], [floor + 2] * 2)
            self.assertEqual([row["cohort"]["complete"] for row in data["results"]], [True, False])


if __name__ == "__main__":
    unittest.main()
