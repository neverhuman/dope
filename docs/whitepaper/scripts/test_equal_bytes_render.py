"""The equal-byte renderer's text outputs equal the committed files and stay digit-free where linted."""
import sys
import hashlib
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claim_registry  # noqa: E402
import equal_bytes_text as T  # noqa: E402


class EqualBytesRenderTests(unittest.TestCase):
    def test_committed_text_outputs_are_current(self):
        for name, expected in T.text_outputs().items():
            self.assertEqual((T.REPO / name).read_text(), expected, name)

    def test_readme_carries_the_generated_block_once(self):
        readme = (T.REPO / "README.md").read_text()
        block = (T.REPO / "paper/generated/readme-equal-bytes.md").read_text().strip()
        self.assertEqual(readme.count(T.README_BEGIN), 1)
        self.assertIn(block, readme)

    def test_readme_block_is_the_only_place_with_study_numbers(self):
        readme = (T.REPO / "README.md").read_text()
        start, end = readme.index(T.README_BEGIN), readme.index(T.README_END) + len(T.README_END)
        failures = claim_registry.lint_public_text(readme[:start] + readme[end:].split("<details>")[0], "README")
        self.assertEqual([f for f in failures if "hand-typed" in f], [])

    def test_origin_correction_preserves_historical_membership(self):
        origin = json.loads((T.REPO / "docs/evidence/pmlb-pollen-origin-20261021.json").read_text())
        self.assertEqual(hashlib.sha256(origin["metadata_utf8"].encode()).hexdigest(), origin["metadata_sha256"])
        self.assertIn("Synthetic pollen dataset", origin["metadata_utf8"])
        summary = json.loads((T.RESULTS / "summary.json").read_text())
        self.assertIn(origin["dataset"], {row["dataset"] for row in summary["lineages"]})
        block = T.text_outputs()["paper/generated/readme-equal-bytes.md"]
        self.assertIn("mixed/unknown-origin", block)
        self.assertIn("simulated pollen", block)
        self.assertNotIn("real-world tables that reach", block)

    def test_frontier_coverage_counts_failures_in_the_denominator(self):
        lineages = [[{"bytes": 500, "robust": 0.9}], [{"bytes": 500, "robust": 0.5}], []]
        self.assertEqual(T.coverage(lineages, 0.8, [100, 1000]), [0.0, 1 / 3])


if __name__ == "__main__":
    unittest.main()
