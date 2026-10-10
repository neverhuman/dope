"""Cross-file count checks on fixture artifacts."""

from __future__ import annotations

import unittest

import paper_consistency as consistency

FOREST = ("ForestDiffusion/Forest-Flow", "native_selected")
DOPE = ("DOPE", "features12_steps2048")
ARF = ("ARF", "author_default")


def row(arm, n, size=4, auditor="catboost", **extra):
    return {"method": arm[0], "configuration": arm[1], "size": size, "auditor": auditor, "n": n, **extra}


def informative(arm, n, size=4, auditor="catboost"):
    return {"method": arm[0], "configuration": arm[1], "size": size, "auditor": auditor, "n_informative": n}


# Shaped like the review panel: the levels and the corrected denominator agree.
ARTIFACTS = {
    "informative": [informative(DOPE, 97), informative(ARF, 99), informative(FOREST, 6),
                    informative(FOREST, 5, auditor="linear"), informative(DOPE, 92, auditor="linear")],
    "level": [row(DOPE, 97), row(ARF, 99), row(FOREST, 6), row(FOREST, 5, auditor="linear")],
    "paired": [row(ARF, 97), row(FOREST, 6), row(FOREST, 5, auditor="linear")],
    "printed": [
        {"file": "numbers.tex", "quantity": "NDopeCb", "n": 97, "cohort": None},
        {"file": "headline-table.tex", "quantity": "NDopeCb", "n": 97, "cohort": None},
    ],
}


class ConsistencyTests(unittest.TestCase):
    def test_consistent_artifacts_pass(self):
        self.assertEqual(consistency.check_artifacts(ARTIFACTS), [])
        self.assertEqual(consistency.check_artifacts({}), [])

    def test_paired_n_cannot_exceed_either_informative_side(self):
        index = consistency.informative_index(ARTIFACTS["informative"])
        self.assertEqual(consistency.paired_n_violations([row(ARF, 98)], index), [
            "paired ARF author_default 4n catboost: n=98 exceeds the left informative n=97 "
            "(DOPE features12_steps2048 4n catboost)",
        ])
        found = consistency.paired_n_violations([row(FOREST, 7)], index)
        self.assertEqual(len(found), 1)
        self.assertIn("exceeds the right informative n=6", found[0])
        found = consistency.paired_n_violations([row(FOREST, 4, auditor="mlp")], index)
        self.assertEqual(len(found), 2)
        self.assertIn("no informative count for the left side DOPE features12_steps2048 4n mlp", found[0])
        other = row(ARF, 99, left_method="ARF", left_configuration="native_selected")
        self.assertIn("left side ARF native_selected", consistency.paired_n_violations([other], index)[0])

    def test_level_n_must_equal_informative_n(self):
        # The pre-fix denominator printed 0 informative Forest-Flow lineages beside a level n of 6.
        stale = [informative(DOPE, 97), informative(FOREST, 0)]
        found = consistency.level_n_violations([row(DOPE, 97), row(FOREST, 6), row(ARF, 99)],
                                               consistency.informative_index(stale))
        self.assertEqual(found, [
            "level ForestDiffusion/Forest-Flow native_selected 4n catboost: n=6 but informative n=0",
            "level ARF author_default 4n catboost: no informative count",
        ])
        with self.assertRaisesRegex(ValueError, "two informative counts"):
            consistency.informative_index([informative(FOREST, 6), informative(FOREST, 0)])

    def test_a_quantity_printed_twice_needs_distinct_cohort_labels(self):
        def printed(file, n, cohort=None, quantity="MedDopeCb"):
            return {"file": file, "quantity": quantity, "n": n, "cohort": cohort}

        self.assertEqual(consistency.cohort_violations([printed("a.tex", 97), printed("b.tex", 97)]), [])
        self.assertEqual(consistency.cohort_violations([printed("a.tex", 97), printed("b.tex", 100)]), [
            "MedDopeCb: n=97 in a.tex and n=100 in b.tex without two distinct cohort labels",
        ])
        self.assertEqual(len(consistency.cohort_violations(
            [printed("a.tex", 97, "density"), printed("b.tex", 100)])), 1)
        self.assertEqual(len(consistency.cohort_violations(
            [printed("a.tex", 97, "density"), printed("b.tex", 100, "density")])), 1)
        self.assertEqual(consistency.cohort_violations(
            [printed("a.tex", 97, "density"), printed("b.tex", 100, "threshold set")]), [])
        self.assertEqual(consistency.cohort_violations(
            [printed("a.tex", 97), printed("b.tex", 100, quantity="NArfCb")]), [])
        artifacts = dict(ARTIFACTS, printed=[printed("a.tex", 97), printed("b.tex", 98)])
        self.assertEqual(len(consistency.check_artifacts(artifacts)), 1)

    def test_strongest_comparator_needs_enough_paired_lineages(self):
        arms = [
            row(ARF, 97, median=0.830),
            row(("ARF", "native_selected"), 97, median=0.815),
            row(("GaussianCopula", "native_selected"), 97, median=0.656),
            row(("TVAE", "native_selected"), 21, median=0.669),
            row(FOREST, 6, median=1.003),
            row(DOPE, 97, median=0.940),
            row(ARF, 97, size=1, median=0.95),
            row(ARF, 92, auditor="linear", median=0.99),
            row(("Broken", "native_selected"), 97, median=float("nan")),
        ]
        self.assertEqual(consistency.strongest_comparator(arms)["configuration"], "author_default")
        self.assertEqual(consistency.strongest_comparator(arms, min_lineages=5)["method"], FOREST[0])
        self.assertEqual(consistency.strongest_comparator(arms, min_lineages=20)["method"], "ARF")
        self.assertIsNone(consistency.strongest_comparator(arms, min_lineages=98))
        paired_n = [dict(item, paired_n=item["n"], n=1) for item in arms]
        self.assertEqual(consistency.strongest_comparator(paired_n)["configuration"], "author_default")
        tied = [row(("B", "x"), 95, median=0.5), row(("A", "y"), 95, median=0.5)]
        self.assertEqual(consistency.strongest_comparator(tied)["method"], "A")

    def test_v2_panel_wiring_checks_counts_and_the_strongest_arm(self):
        import json
        import tempfile
        from pathlib import Path

        def level(method, configuration, n, median, size=4, auditor="catboost"):
            return {"method": method, "configuration": configuration, "size": size, "auditor": auditor,
                    "n": n, "median": median}

        panel = {
            "levels": [level("DOPE", "headline", 97, 0.9), level("TabSyn", "scaled", 96, 0.88),
                       level("ARF", "author_default", 97, 0.85), level("Forest", "historical", 6, 0.99),
                       level("real_bootstrap_4n", "control", 97, 1.0)],
            "contrasts": [dict(level("TabSyn", "scaled", 95, None)), dict(level("ARF", "author_default", 97, None)),
                          dict(level("Forest", "historical", 6, None)),
                          dict(level("real_bootstrap_4n", "control", 97, None))],
            "strongest": {"method": "TabSyn", "configuration": "scaled", "median": 0.88, "paired_n": 95},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "panel.json"
            path.write_text(json.dumps(panel))
            self.assertIsNone(consistency.load_generated_artifacts(root / "absent.json", root))
            artifacts = consistency.load_generated_artifacts(path, root)
            self.assertEqual(consistency.check_artifacts(
                {key: artifacts[key] for key in ("informative", "paired")}), [])
            self.assertEqual(consistency.strongest_violations(artifacts), [])
            panel["strongest"] = {"method": "ARF", "configuration": "author_default"}
            path.write_text(json.dumps(panel))
            self.assertIn("rule gives ('TabSyn', 'scaled')",
                          consistency.strongest_violations(consistency.load_generated_artifacts(path, root))[0])
            panel["contrasts"][1]["n"] = 98
            path.write_text(json.dumps(panel))
            artifacts = consistency.load_generated_artifacts(path, root)
            self.assertTrue(consistency.check_artifacts({key: artifacts[key] for key in ("informative", "paired")}))


if __name__ == "__main__":
    unittest.main()
