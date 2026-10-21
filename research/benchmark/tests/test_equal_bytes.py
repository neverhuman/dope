"""Equal-byte study: real-row codec arm, frontier, TDL and summary logic."""
import json
import unittest
from pathlib import Path

import numpy as np

from research.benchmark.equal_bytes import prepare, run, summarize

RESULTS = Path(__file__).resolve().parents[1] / "equal_bytes" / "results"


def record(real_loss: float = 0.2, null: float = 1.0) -> dict:
    return {"dataset": "fixture", "n_fit": 1000, "d": 3, "null": null,
            "real": {a: real_loss for a in summarize.AUDITORS}, "projection_bytes": 32,
            "projection_json_bytes": 900, "full_table_bytes": {"8": 4000, "16": 7000}}


class CodecArmTests(unittest.TestCase):
    def test_encoding_is_monotone_and_eight_bits_is_smaller(self):
        rows = np.random.default_rng(0).random((400, 5))
        sizes = [run.encoded_rows(rows[:k], 8) for k in (10, 50, 200, 400)]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(run.encoded_rows(rows, 8), run.encoded_rows(rows, 16))

    def test_largest_subset_respects_the_budget(self):
        rows = np.random.default_rng(1).random((300, 4))
        order = np.arange(300)
        for budget in (128, 512, 2048):
            k = run.largest_subset(order, rows, 8, budget, 40)
            if k:
                self.assertLessEqual(run.encoded_rows(rows[:k], 8) + 40, budget)
            if k < 300:
                self.assertGreater(run.encoded_rows(rows[:k + 1], 8) + 40, budget)

    def test_program_budgets_are_each_rungs_median_bytes(self):
        record = {"dope": [{"rung": "L4", "bytes": b} for b in (600, 640, 620)]
                  + [{"rung": "L2", "bytes": 400}, {"rung": "L2", "error": "x"}]}
        self.assertEqual(run.program_budgets(record), [400, 620])

    def test_only_the_mlp_is_resampled_and_it_keeps_its_better_loss(self):
        rng = np.random.default_rng(3)
        x = rng.random((40, 3))
        y = x @ np.array([1.0, -2.0, 0.5]) + 0.05 * rng.standard_normal(40)
        x_test = rng.random((200, 3))
        y_test = x_test @ np.array([1.0, -2.0, 0.5])
        best, plain = run.matched_losses(x, y, x_test, y_test, 400, 0)
        self.assertEqual(best["linear"], plain["linear"])
        self.assertEqual(best["catboost"], plain["catboost"])
        self.assertLessEqual(best["mlp"], plain["mlp"])
        self.assertEqual(run.matched_losses(x, y, x_test, y_test, 40, 0)[0], plain)

    def test_ladder_names_are_unique_and_the_headline_is_present(self):
        names = [rung[0] for rung in run.LADDER + run.ABLATIONS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn(run.HEADLINE_RUNG, names)


class SummaryTests(unittest.TestCase):
    def test_retention_is_clipped_and_uninformative_learners_are_ignored(self):
        rec = record()
        self.assertEqual(summarize.retention(rec, "mlp", -5.0), 1.0)
        self.assertEqual(summarize.retention(rec, "mlp", 5.0), 0.0)
        rec["real"]["linear"] = 0.999
        self.assertEqual(summarize.informative(rec), ["mlp", "catboost"])
        self.assertEqual(summarize.robust(rec, {"linear": 0.1, "mlp": 0.7, "catboost": 0.9}), 0.7)

    def test_frontier_and_tdl_use_only_configurations_within_budget(self):
        points = [{"bytes": 300, "robust": 0.5}, {"bytes": 900, "robust": 0.85}, {"bytes": 5000, "robust": 0.95}]
        self.assertIsNone(summarize.frontier(points, 200))
        self.assertEqual(summarize.frontier(points, 1000), 0.85)
        self.assertEqual(summarize.tdl(points, 0.8), 1024)
        self.assertEqual(summarize.tdl(points, 0.9), 8192)
        self.assertIsNone(summarize.tdl(points, 0.99))
        # Interpolated in log bytes between the bracketing cells (300 B at 0.5, 900 B at 0.85).
        self.assertAlmostEqual(summarize.bytes_to_reach(points, 0.8), 300 * 3 ** (0.3 / 0.35), places=6)
        self.assertEqual(summarize.bytes_to_reach(points, 0.5), 300)
        self.assertIsNone(summarize.bytes_to_reach(points, 0.99))

    def test_matched_reference_and_the_as_given_sensitivity(self):
        rec = record(real_loss=0.4)
        rec["real_matched"] = {"linear": 0.4, "mlp": 0.2, "catboost": 0.4}
        row = {"loss": {"linear": 0.7, "mlp": 0.6, "catboost": 0.7},
               "loss_as_given": {"linear": 0.7, "mlp": 0.9, "catboost": 0.7}}
        self.assertAlmostEqual(summarize.retention(rec, "mlp", 0.6), 0.5)
        self.assertEqual(summarize.subset_loss(rec, row), row["loss"])
        summarize.AS_GIVEN = True
        try:
            self.assertAlmostEqual(summarize.retention(rec, "mlp", 0.6), 2 / 3)
            self.assertEqual(summarize.subset_loss(rec, row), row["loss_as_given"])
            self.assertEqual(summarize.subset_loss(rec, {"loss": {}, "anchored": True}), rec["real"])
        finally:
            summarize.AS_GIVEN = False

    def test_two_part_proxy_grows_with_noise_and_retention(self):
        rec = record(real_loss=0.2)
        low = summarize.two_part_proxy(rec, 0.8, 600)
        self.assertGreater(summarize.two_part_proxy(rec, 0.9, 600), low)
        self.assertGreater(summarize.two_part_proxy(record(real_loss=0.5), 0.8, 600), low)

    def test_committed_summary_reproduces_from_committed_records(self):
        records = [json.loads(p.read_text()) for p in sorted((RESULTS / "lineages").glob("*.json"))]
        lineages = [summarize.lineage_summary(r) for r in records]
        committed = json.loads((RESULTS / "summary.json").read_text())
        self.assertEqual(json.loads(json.dumps(summarize.aggregate(lineages), default=lambda v: None)),
                         committed["aggregate"])
        self.assertEqual(json.loads(json.dumps(summarize.sensitivity(records), default=lambda v: None)),
                         committed["sensitivity"])

    def test_committed_panel_summary_reproduces_on_the_two_rung_ladder(self):
        paths = sorted([*(RESULTS / "lineages").glob("*.json"), *(RESULTS / "panel").glob("*.json")],
                       key=lambda p: p.name)
        self.assertEqual(len(paths), 97)
        previous, summarize.RUNGS = summarize.RUNGS, ("L2", "L4")
        try:
            records = [json.loads(p.read_text()) for p in paths]
            aggregate = summarize.aggregate([summarize.lineage_summary(r) for r in records])
            sensitivity = summarize.sensitivity(records)
        finally:
            summarize.RUNGS = previous
        committed = json.loads((RESULTS / "panel-summary.json").read_text())
        self.assertEqual(json.loads(json.dumps(aggregate, default=lambda v: None)), committed["aggregate"])
        self.assertEqual(json.loads(json.dumps(sensitivity, default=lambda v: None)), committed["sensitivity"])


class PrepareTests(unittest.TestCase):
    def test_real_world_selection_drops_simulated_and_duplicate_lineages(self):
        names = {entry["pmlb_id"] for entry in prepare.real_world_entries()}
        self.assertEqual(len(names), 26)
        self.assertFalse(any(prepare.is_simulated(name) for name in names))
        self.assertFalse(set(prepare.DUPLICATES) & names)
        self.assertEqual(len(prepare.real_world_entries(include_simulated=True)), 97)


if __name__ == "__main__":
    unittest.main()
