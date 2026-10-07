"""Stdlib-only custody and input controls; these tests never import plotting packages."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import render_matched_linear_fit11_figure as renderer

BENCHMARK = Path(__file__).resolve().parents[1]
RESULTS = BENCHMARK / "results" / "matched-linear-fit11-figure-v1"
PANEL = BENCHMARK / "results" / "matched-linear-fit11-retention-v1" / "panel.json"
ARTIFACT_PINS = {
    "matched-linear-fit11.svg": (22434, "5d0dbbea790ac3ad92006ac40643699df94c092f1ec2b52407f024a71254a526"),
    "matched-linear-fit11.pdf": (18304, "00608f323e2b5472ed75bc89ee27a953e0f3301abe85b79b143f7e51c21521db"),
}


def same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


class LinearFit11Figure(unittest.TestCase):
    def setUp(self):
        raw = PANEL.read_bytes()
        self.assertEqual(len(raw), renderer.PANEL_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), renderer.PANEL_SHA256)
        panel = json.loads(raw)
        self.fixture = {k: copy.deepcopy(panel[k]) for k in (
            "format", "status", "fit_seed", "sample_seeds", "gated_scores",
            "artifact_scope", "official_tests_opened", "production_certified", "summary")}

    def read_fixture(self, fixture):
        raw = json.dumps(fixture, sort_keys=True, allow_nan=True).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.json"
            path.write_bytes(raw)
            with patch.object(renderer, "PANEL_BYTES", len(raw)), patch.object(
                    renderer, "PANEL_SHA256", hashlib.sha256(raw).hexdigest()):
                return renderer.read_summary(path)

    def test_frozen_six_summary_rows_and_cohorts(self):
        rows = renderer.read_summary(PANEL)
        self.assertEqual(len(rows), 6)
        for method, role, _, count in renderer.PANELS:
            selected = [r for r in rows if (r["peer_method"], r["peer_configuration"]) == (method, role)]
            self.assertEqual([r["size_multiplier"] for r in selected], [1, 4])
            self.assertEqual([r["paired_complete_informative_lineages"] for r in selected], [count, count])

    def test_hash_rejected_before_JSON_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.json"
            path.write_bytes(b"x" * renderer.PANEL_BYTES)
            with patch.object(renderer.json, "loads", side_effect=AssertionError("decode reached")):
                with self.assertRaisesRegex(ValueError, "public_input_pin_mismatch"):
                    renderer.read_summary(path)

    def test_missing_duplicate_and_mixed_summary_rejected(self):
        for change in ("missing", "duplicate", "mixed"):
            fixture = copy.deepcopy(self.fixture)
            if change == "missing":
                fixture["summary"].pop()
            elif change == "duplicate":
                fixture["summary"][0] = copy.deepcopy(fixture["summary"][1])
            else:
                fixture["summary"][0]["peer_configuration"] = "new_search"
            with self.assertRaises(ValueError):
                self.read_fixture(fixture)

    def test_boolean_identity_substitutions_rejected(self):
        for level, field in (("root", "fit_seed"), ("row", "size_multiplier"),
                             ("row", "paired_complete_informative_lineages")):
            fixture = copy.deepcopy(self.fixture)
            target = fixture if level == "root" else fixture["summary"][0]
            target[field] = True
            with self.assertRaises(ValueError):
                self.read_fixture(fixture)

    def test_nonfinite_summary_rejected(self):
        for value in (float("nan"), float("inf")):
            fixture = copy.deepcopy(self.fixture)
            fixture["summary"][0]["median_paired_difference"] = value
            with self.assertRaisesRegex(ValueError, "nonfinite_JSON"):
                self.read_fixture(fixture)

    def test_gated_scores_and_TEST_scope_rejected(self):
        for key in ("gated_scores", "official_tests_opened", "production_certified"):
            fixture = copy.deepcopy(self.fixture)
            if key == "gated_scores":
                fixture[key]["mfs_v2"] = 0
            else:
                fixture[key] = True
            with self.assertRaises(ValueError):
                self.read_fixture(fixture)

    def test_frozen_vector_and_renderer_byte_custody(self):
        for name, (length, digest) in ARTIFACT_PINS.items():
            raw = (RESULTS / name).read_bytes()
            self.assertEqual(len(raw), length)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        raw = Path(renderer.__file__).read_bytes()
        self.assertEqual(len(raw), 9036)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "52dcd1dafd424783c7e009ce1040520d89a9f44bb00519a801af8aa0645cbc51")

    def test_provenance_matches_closed_schema(self):
        proof = json.loads((RESULTS / "source-proof.json").read_bytes())
        schema = json.loads((RESULTS / "source-proof.schema.json").read_bytes())
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(set(proof), set(schema["required"]))
        self.assertEqual(set(proof), set(schema["properties"]))
        for key, value in proof.items():
            self.assertTrue(same(value, schema["properties"][key]["const"]))
        self.assertTrue(same(proof["gated_scores"], dict(mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None)))
        self.assertFalse(same(False, 0))

    def test_paired_difference_is_distinct_from_displayed_medians(self):
        rows = renderer.read_summary(PANEL)
        row = next(r for r in rows if r["peer_method"] == "Forest-Flow" and r["size_multiplier"] == 4)
        self.assertNotEqual(row["median_paired_difference"],
                            row["DOPE_median_of_same_cohort_three_sample_means"] - row["peer_median_of_same_cohort_three_sample_means"])
        svg = (RESULTS / "matched-linear-fit11.svg").read_text()
        self.assertIn("Median paired difference is separate", svg)

    def test_plotting_dependencies_are_lazy(self):
        tree = ast.parse(Path(renderer.__file__).read_bytes())
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith(("matplotlib", "numpy")) for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith(("matplotlib", "numpy")))

    def test_public_leafs_have_no_private_absolute_paths(self):
        for name in ("source-proof.json", "source-proof.schema.json", "README.md", *ARTIFACT_PINS):
            raw = (RESULTS / name).read_bytes()
            self.assertNotIn(b"/home/", raw)
            self.assertNotIn(b"/mnt/", raw)
            self.assertNotIn(b"xbabe", raw)


if __name__ == "__main__":
    unittest.main()
