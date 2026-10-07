"""Generated metadata controls for the committed linear-only scalar panel."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import render_matched_linear_fit11 as renderer


class LinearFit11Retention(unittest.TestCase):
    def setUp(self):
        self.panel = renderer.load(renderer.RESULTS)

    def test_complete_scalar_projection_and_alias_counts(self):
        summary = renderer.recompute(self.panel)
        self.assertEqual(len(summary), 6)
        self.assertEqual(self.panel["coverage"]["logical_comparator_cells"], 210)
        self.assertEqual(self.panel["coverage"]["distinct_peer_metric_receipts"], 204)

    def test_exact_public_csv_and_markdown(self):
        summary = renderer.recompute(self.panel)
        self.assertEqual(renderer.table(summary), (renderer.RESULTS / "figure-kpis.csv").read_bytes())
        self.assertEqual(renderer.markdown(summary), (renderer.RESULTS / "README.md").read_bytes())

    def test_hash_rejection_precedes_JSON_decode(self):
        raw = b'{"not":"trusted"}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_bytes(raw)
            with patch.object(renderer.json, "loads", side_effect=AssertionError("JSON decoding reached")):
                with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
                    renderer.read_public(path, len(raw), "0" * 64)

    def test_missing_sample_rejected(self):
        self.panel["cells"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete_sample_group"):
            renderer.recompute(self.panel)

    def test_duplicate_logical_cell_rejected(self):
        self.panel["cells"].append(copy.deepcopy(self.panel["cells"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate_cell"):
            renderer.recompute(self.panel)

    def test_mixed_task_and_boolean_seed_rejected(self):
        for field, value in (("task", "binary"), ("sample_seed", True)):
            changed = copy.deepcopy(self.panel)
            changed["cells"][0][field] = value
            with self.assertRaises(ValueError):
                renderer.recompute(changed)

    def test_alias_payload_drift_rejected(self):
        seen = {}
        for cell in self.panel["cells"]:
            pin = cell["peer_metric_sha256"]
            if pin in seen:
                cell["peer_retention"] += .125
                break
            seen[pin] = cell
        else:
            self.fail("frozen default/native physical alias missing")
        with self.assertRaisesRegex(ValueError, "inconsistent_physical_alias"):
            renderer.recompute(self.panel)

    def test_gated_scores_and_physical_cost_remain_unknown(self):
        for parent, key, value in (("gated_scores", "mfs_v2", 0), ("cost_scope", "wall_seconds", 1.0), ("cost_scope", "logical_aliases_are_independent", True)):
            changed = copy.deepcopy(self.panel)
            changed[parent][key] = value
            with self.assertRaises(ValueError):
                renderer.recompute(changed)

    def test_nested_boolean_numeric_substitution_rejected(self):
        self.panel["cost_scope"]["physical_cost_estimated"] = 0
        with self.assertRaisesRegex(ValueError, "cost_or_artifact_scope"):
            renderer.recompute(self.panel)

    def test_missing_task_source_rejected(self):
        proof = renderer.read_public(renderer.RESULTS / "source-proof.json", *renderer.PINS["source-proof.json"])
        proof["projection_task_refs"].pop()
        with self.assertRaisesRegex(ValueError, "projection_task_coverage"):
            renderer.verify_source_proof(self.panel, proof)

    def test_low_signal_exclusions_are_not_imputed(self):
        excluded = {r["dataset"] for r in self.panel["exclusions"]}
        self.assertEqual(excluded, {"19f4780b53b3fa41", "a04f964bc53281e0"})
        cell = next(c for c in self.panel["cells"] if not c["informative"])
        cell["DOPE_retention"] = 0.0
        with self.assertRaisesRegex(ValueError, "retention_applicability"):
            renderer.recompute(self.panel)

    def test_stored_aggregate_and_nonfinite_drift_rejected(self):
        self.panel["summary"][0]["median_paired_difference"] += .1
        with self.assertRaisesRegex(ValueError, "stored_aggregate_mismatch"):
            renderer.recompute(self.panel)
        self.panel["cells"][0]["peer_retention"] = float("inf")
        with self.assertRaisesRegex(ValueError, "retention_applicability"):
            renderer.recompute(self.panel)

    def test_paired_median_difference_is_not_difference_of_medians(self):
        row = next(r for r in renderer.recompute(self.panel) if r["peer_method"] == "Forest-Flow" and r["size_multiplier"] == 4)
        self.assertNotEqual(row["median_paired_difference"], row["DOPE_median_of_same_cohort_three_sample_means"] - row["peer_median_of_same_cohort_three_sample_means"])

    def test_public_payload_has_no_private_absolute_paths(self):
        for filename in ("panel.json", "source-proof.json", "figure-kpis.csv", "README.md"):
            raw = (renderer.RESULTS / filename).read_bytes()
            self.assertNotIn(b"/home/", raw)
            self.assertNotIn(b"/mnt/", raw)


if __name__ == "__main__":
    unittest.main()
