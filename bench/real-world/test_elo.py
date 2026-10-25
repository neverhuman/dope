"""Deterministic fixtures for ranking integrity; no scientific qualification."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import elo
from elo_math import leaderboard, ratings


def cell(ds="a", method="a", budget="1k", seed=0, retention=.8, size=1024, status="ok"):
    return {"dataset": ds, "method": method, "budget": budget, "seed": seed,
            "status": status, "robust": retention, "bytes": size}


class Curves(unittest.TestCase):
    def test_byte_eligibility_log_area_and_t90(self):
        c = elo.curve([cell(size=2048, retention=.95)], [("1k", 0)], [1024, 2048, 4096])
        self.assertEqual(c["retention_curve"], [0., .95, .95])
        self.assertAlmostEqual(c["btr_auc"], 71.25)
        self.assertEqual(c["grid_t90_bytes"], 2048)

    def test_failures_in_seed_median_are_not_dropped(self):
        rows = [cell(seed=s, retention=1., status="ok" if s == 0 else "failed") for s in range(3)]
        c = elo.curve(rows, [("1k", s) for s in range(3)], [1024, 4096])
        self.assertEqual(c["btr_auc"], 0.)
        self.assertEqual(c["failed_cells"], 2)

    def test_missing_failed_and_unqualified_cells(self):
        self.assertEqual(elo.curve([], [("1k", 0)], [1024, 4096])["status"], "incomplete")
        self.assertEqual(elo.curve([cell(status="failed")], [("1k", 0)], [1024, 4096])["btr_auc"], 0.)
        for invalid in [None, float("nan"), float("inf")]:
            self.assertEqual(elo.curve([cell(retention=invalid)], [("1k", 0)], [1024, 4096])["status"], "unqualified")

    def test_clipping_and_measured_envelope(self):
        rows = [cell(budget="1k", retention=-4.), cell(budget="4k", retention=3., size=4096)]
        c = elo.curve(rows, [("1k", 0), ("4k", 0)], [1024, 2048, 4096])
        self.assertEqual(c["retention_curve"], [0., 0., 1.])
        self.assertAlmostEqual(c["btr_auc"], 25.)
        self.assertEqual(c["negative_retention_cells"], 1)

    def test_oversized_seeds_count_zero_even_if_seed_median_size_fits(self):
        c = elo.curve([cell(seed=0, size=500), cell(seed=1, size=1500)],
                      [("1k", 0), ("1k", 1)], [1024, 2048])
        self.assertEqual(c["retention_curve"], [.4, .8])

    def test_duplicate_receipts_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "records.jsonl"
            p.write_text(json.dumps(cell()) + "\n" + json.dumps(cell()) + "\n")
            self.assertEqual(len(elo.load(p)), 1)
            p.write_text(json.dumps(cell()) + "\n" + json.dumps(cell(retention=.5)) + "\n")
            with self.assertRaises(ValueError):
                elo.load(p)


class Elo(unittest.TestCase):
    def test_equal_scores_center_and_order_invariance(self):
        np.testing.assert_allclose(ratings(np.ones((5, 3))), [1500.] * 3)
        s = np.array([[90., 70., 20.], [60., 80., 30.], [90., 50., 15.]])
        e = ratings(s)
        self.assertGreater(e[0], e[1])
        self.assertGreater(e[1], e[2])
        self.assertAlmostEqual(e.mean(), 1500.)
        np.testing.assert_allclose(e, ratings(s[::-1]))
        np.testing.assert_allclose(e, ratings(s[:, ::-1])[::-1])

    def test_tie_margin_and_perfect_winner_finite(self):
        np.testing.assert_allclose(ratings(np.array([[90., 89.5]]), 1.), [1500.] * 2)
        e = ratings(np.tile([100., 0.], (1000, 1)))
        self.assertTrue(np.isfinite(e).all())
        self.assertGreater(e[0], 1500.)

    def test_bootstrap_reproducible_and_counts(self):
        a = leaderboard(np.array([[90., 80.], [60., 70.]]), ["a", "b"], 1., 20, 7)
        self.assertEqual(a, leaderboard(np.array([[90., 80.], [60., 70.]]), ["a", "b"], 1., 20, 7))
        self.assertEqual(a[0]["games"], 2)
        self.assertEqual((a[0]["wins"], a[0]["losses"]), (1, 1))

    def test_common_cohort_and_local_missing(self):
        rows = [cell(ds=d, method=m, retention=v) for d in ["x", "y"] for m, v in [("a", .9), ("b", .6)]]
        rows += [cell(ds="x", method="c", retention=.4)]
        s = elo.summarize(rows, ["a", "b", "c"], [1024, 4096], 1., 10, 1)
        self.assertEqual(s["common_datasets"], ["x"])
        self.assertEqual(s["per_dataset"]["y"]["c"]["status"], "incomplete")
        self.assertNotIn("elo", s["per_dataset"]["y"]["c"])
        self.assertEqual({r["datasets"] for r in s["leaderboard"]}, {1})
        self.assertEqual(set(elo.outputs(s)), {"elo.json", "elo.md", "datasets.csv", "elo.svg", "datasets.svg"})

    def test_last_bit_float_noise_does_not_drift_surfaces(self):
        rows = [cell(ds=d, method=m, retention=v) for d in ["x", "y"] for m, v in [("a", .9), ("b", .6)]]
        s = elo.summarize(rows, ["a", "b"], [1024, 4096], 1., 10, 1)
        noisy = json.loads(json.dumps(s))
        noisy["per_dataset"]["x"]["a"]["elo"] = float(np.nextafter(noisy["per_dataset"]["x"]["a"]["elo"], 0))
        self.assertEqual(elo.outputs(s), elo.outputs(noisy))

    def test_predeclared_schedule_preserves_absent_cells(self):
        rows = [cell(method=m) for m in ["a", "b"]]
        schedule = {m: [["1k", 0], ["1k", 1]] for m in ["a", "b"]}
        with self.assertRaisesRegex(ValueError, "no common"):
            elo.summarize(rows, ["a", "b"], [1024, 2048], 1., 10, 1, schedule)
        for budgets in ([1024, 1024], [2048, 1024], [0, 2048]):
            with self.assertRaises(ValueError):
                elo.summarize(rows, ["a", "b"], budgets, 1., 10, 1)

    def test_dataset_manifest_keeps_entirely_absent_jobs_visible(self):
        rows = [cell(method=m) for m in ["a", "b"]]
        s = elo.summarize(rows, ["a", "b"], [1024, 2048], 1., 10, 1,
                          dataset_ids=["a", "never-ran"])
        self.assertEqual(s["all_datasets"], ["a", "never-ran"])
        self.assertEqual(s["common_datasets"], ["a"])
        self.assertEqual(s["per_dataset"]["never-ran"]["a"]["status"], "incomplete")

    def test_committed_input_covers_seven_and_full_cpu_sixteen(self):
        rows = elo.load(elo.HERE / "results/confirm/records.jsonl")
        all_methods = elo.summarize(rows, elo.METHODS, elo.BUDGETS, 1., 10, 1)
        cpu = elo.summarize(rows, elo.METHODS[:-2], elo.BUDGETS, 1., 10, 1)
        self.assertEqual(len(all_methods["common_datasets"]), 7)
        self.assertEqual(len(cpu["common_datasets"]), 16)

    def test_multilevel_convergence_snapshot_matches_live_receipts(self):
        from render_tex import ROOT, _multilevel_macro_items, multilevel_convergence_data
        snap = multilevel_convergence_data()
        t1 = snap["v2_full_corpus_maxtrain97_val3pct"]
        t2 = snap["v2_disjoint1000_maxtrain97_devval345"]
        t3 = snap["v3_maxtrain97_val3pct"]
        self.assertLess(t1["val_dataset_pct"], 5.0)
        self.assertLess(t1["val_split_pct"], 5.0)
        self.assertLess(t2["val_dataset_pct"], 5.0)
        self.assertLess(t2["val_split_pct"], 5.0)
        self.assertLess(t3["loaded_episode_cells_val_pct"], 5.0)
        self.assertLess(t3["catalog_manifest_val_dataset_pct"], 5.0)
        self.assertEqual((t1["train_population_splits"], t1["val_population_splits"], t1["attempted_visits_cumulative_total"]), (18395, 684, 24527))
        self.assertAlmostEqual(t1["effective_full_population_passes_total"], 24527 / 18395, places=6)
        self.assertEqual((t2["train_population_splits"], t2["val_population_splits"], t2["attempted_visits_cumulative_total"]), (11976, 345, 15968))
        self.assertAlmostEqual(t2["effective_full_population_passes_total"], 15968 / 11976, places=6)
        self.assertEqual(
            (t3["loaded_episode_cells_train_total"], t3["loaded_episode_cells_val_total"], t3["loaded_episode_cells_pool_total"]),
            (2229, 75, 2304),
        )
        self.assertAlmostEqual(t3["loaded_episode_cells_val_pct"], 100 * 75 / 2304, places=4)
        self.assertEqual(t3["attempted_visits_cumulative_total_per_variant"], 4457)
        self.assertAlmostEqual(t3["effective_full_population_passes_total_per_variant"], 4457 / 2229, places=6)
        macros = dict(_multilevel_macro_items())
        self.assertEqual(macros["MlcMaxTrainLossInit"], "0.40988")
        self.assertEqual(macros["MlcMaxTrainLossBest"], "0.34216")
        self.assertEqual(macros["MlcDisjointLossBest"], "0.57363")
        self.assertEqual(macros["MlcVThreeMaxTrainLossBest"], "0.68492")
        self.assertEqual(macros["MlcVThreeMaxTrainLoadedPool"], "2{,}304")
        readme_block = (ROOT / "bench/real-world/results/readme-block.md").read_text(encoding="utf-8")
        for needle in ("0.34216 ± 0.01621", "1.333351×", "0.57363 ± 0.02131", "0.68492 ± 0.03877", "1.999551×", "3.2552% < 5%"):
            self.assertIn(needle, readme_block)


if __name__ == "__main__":
    unittest.main()
