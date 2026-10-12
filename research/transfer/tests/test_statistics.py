from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest

import numpy as np

from research.transfer.losses import PairedLosses, TrainingScale
from research.transfer.metrics import Margin
from research.transfer.power import PowerScenario, power_simulation
from research.transfer.statistics import (
    CrossedDesign, EvaluationRun, bootstrap_effects, interval_decision,
    leave_one_global_seed_out, paired_family_effect, simultaneous_ucb, task_substream,
)


PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "protocols/statistics-v1.json").read_text())


def scheduled(runs, methods):
    runs = tuple(runs)
    return CrossedDesign(runs, methods, tuple(r.key for r in runs))


def run(family="f", task="t", method="a", fit="0", values=(0., 0.), namespace=None, seed=None,
        sample="0", learner="0", clusters=None):
    n = len(values)
    pairs = PairedLosses(tuple(f"q-{i}" for i in range(n)), clusters or tuple(f"e-{i}" for i in range(n)),
                         (10.,) * n, tuple(10 + v for v in values), (20.,) * n)
    return EvaluationRun(family, task, method, fit, sample, learner, pairs, TrainingScale(1.), Margin(0., 0.),
                         global_seed_namespace=namespace, global_seed_id=seed)


def seed_panel(nfamilies=8, *, methods=("a",), namespaces=None):
    runs = []
    for f in range(nfamilies):
        for method in methods:
            for s, value in enumerate((-3., -1., 1., 3.)):
                namespace = namespaces[method] if namespaces else "panel"
                runs.append(run(f"f-{f}", f"t-{f}", method, str(s), (value, value), namespace, str(s)))
    return scheduled(tuple(runs), methods)


class StatisticsTests(unittest.TestCase):
    def test_family_balancing_and_duplicate_alias_refusal(self):
        runs = [run("f", f"variant-{i}", values=(0., 0.)) for i in range(10)]
        runs += [run("other", "other", values=(2., 2.))]
        design = scheduled(tuple(runs), ("a",))
        point = paired_family_effect(design)
        self.assertEqual(point["families"], 2)
        self.assertEqual(point["estimates"]["a"], 1.)
        with self.assertRaises(ValueError):
            scheduled(tuple(runs + [runs[0]]), ("a",))
        with self.assertRaises(ValueError):
            scheduled((runs[0], replace(runs[0], family_id="wrong")), ("a",))

    def test_global_seed_axis_survives_source_replication(self):
        small = bootstrap_effects(seed_panel(4), draws=999, seed=12)
        large = bootstrap_effects(seed_panel(40), draws=999, seed=12)
        self.assertEqual(small["status"], "complete")
        # Identical family copies cannot wash out shared panel-wide fit shocks.
        np.testing.assert_allclose(small["samples"]["a"], large["samples"]["a"])
        self.assertGreater(np.std(large["samples"]["a"]), 0.9)
        sensitivity = leave_one_global_seed_out(seed_panel(4))
        self.assertAlmostEqual(sensitivity["panel"]["0"]["a"], 1.)

    def test_equal_integer_seeds_do_not_pair_unrelated_methods(self):
        independent = bootstrap_effects(seed_panel(methods=("a", "b"), namespaces={"a": "algorithm-a", "b": "algorithm-b"}), draws=999, seed=4)
        shared = bootstrap_effects(seed_panel(methods=("a", "b")), draws=999, seed=4)
        self.assertLess(abs(np.corrcoef(independent["samples"]["a"], independent["samples"]["b"])[0, 1]), 0.15)
        np.testing.assert_allclose(shared["samples"]["a"], shared["samples"]["b"])
        a = task_substream(11, "task", "algorithm-a/fit")
        self.assertEqual(a, task_substream(11, "task", "algorithm-a/fit"))
        self.assertNotEqual(a, task_substream(11, "task", "algorithm-b/fit"))
        self.assertNotEqual(a, task_substream(11, "other", "algorithm-a/fit"))

    def test_fixed_artifact_clusters_retain_all_rows_and_pairs(self):
        clusters = ("household-a", "household-a", "household-b", "household-b")
        design = scheduled((run(values=(-1., -1., 1., 1.), clusters=clusters),), ("a",))
        result = bootstrap_effects(design, estimand="fixed_artifact", draws=999, seed=1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["samples"]["a"]), {-1., 0., 1.})
        self.assertAlmostEqual(np.var(result["samples"]["a"]), 0.5, delta=0.08)
        p = PairedLosses(("q-0", "q-1"), ("e-0", "e-1"), (10., 20.), (11., 22.), (30., 30.))
        paired = replace(run(), pairs=p, margin=Margin(0.1, 0.))
        result = bootstrap_effects(scheduled((paired,), ("a",)), estimand="fixed_artifact", draws=199, seed=9)
        np.testing.assert_allclose(result["samples"]["a"], 0., atol=1e-12)

    def test_nested_fits_samples_and_learners_are_not_collapsed(self):
        runs = []
        for family in range(3):
            for fit in range(3):
                for sample in range(2):
                    for learner in range(2):
                        value = fit + 2 * sample + 4 * learner
                        runs.append(run(f"f-{family}", f"t-{family}", fit=str(fit), sample=str(sample), learner=str(learner), values=(value, value)))
        design = scheduled(tuple(runs), ("a",))
        self.assertEqual(paired_family_effect(design)["estimates"]["a"], 4.)
        result = bootstrap_effects(design, draws=999, seed=41)
        self.assertGreater(np.std(result["samples"]["a"]), 0.2)
        with self.assertRaises(ValueError):
            bootstrap_effects(design, estimand="fixed_artifact", draws=199)

    def test_missing_unbalanced_and_small_axes_fail_closed(self):
        panel = seed_panel(3)
        with self.assertRaises(ValueError):
            CrossedDesign(panel.runs[4:], panel.required_methods, panel.scheduled_keys)
        runs = list(panel.runs)
        runs[0] = replace(runs[0], pairs=None, status="technical_failure")
        incomplete = scheduled(tuple(runs), ("a",))
        result = bootstrap_effects(incomplete, draws=199)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(simultaneous_ucb(result)["bounds"])
        self.assertEqual(paired_family_effect(incomplete)["estimand"], "complete_case_sensitivity")
        self.assertEqual(incomplete.counts["scheduled_runs"], 12)
        missing_seed = scheduled(panel.runs[1:], ("a",))
        self.assertEqual(bootstrap_effects(missing_seed, draws=199)["status"], "incomplete")
        self.assertEqual(bootstrap_effects(seed_panel(1), draws=199)["status"], "insufficient_families")
        tiny = scheduled((run(values=(1.,)),), ("a",))
        self.assertEqual(bootstrap_effects(tiny, estimand="fixed_artifact", draws=199)["status"], "insufficient_query_clusters")

    def test_query_alignment_and_coupling_validation(self):
        panel = seed_panel(3)
        bad = replace(panel.runs[1], pairs=replace(panel.runs[1].pairs, query_ids=("other", "q-1")))
        with self.assertRaises(ValueError):
            scheduled((panel.runs[0], bad), ("a",))
        with self.assertRaises(ValueError):
            replace(run(), global_seed_id="11")

    def test_simultaneous_bounds_and_equivalence(self):
        single = bootstrap_effects(seed_panel(4), draws=999, seed=18)
        joint = {**single, "samples": {"a": single["samples"]["a"], "b": single["samples"]["a"]},
                 "estimates": {"a": 0., "b": 0.}}
        self.assertGreaterEqual(simultaneous_ucb(joint)["bounds"]["a"]["upper"], simultaneous_ucb(single)["bounds"]["a"]["upper"])
        self.assertFalse(interval_decision(-1., 1., equivalence_margin=0.1)["equivalent"])
        self.assertTrue(interval_decision(-0.01, 0.01, equivalence_margin=0.1)["equivalent"])
        self.assertFalse(interval_decision(-0.1, 0.1, equivalence_margin=0.1)["equivalent"])
        self.assertTrue(interval_decision(0.2, 0.3)["harmful"])
        self.assertTrue(interval_decision(-0.3, -0.2)["superior"])
        too_many = {**joint, "draws": 99}
        self.assertEqual(simultaneous_ucb(too_many)["status"], "insufficient_tail_draws")

    def test_development_power_null_inferior_equivalent_heavy_tail(self):
        policy = PROTOCOL["development_validation"]
        evidence = {}
        for index, (name, scenario) in enumerate((
            ("null", PowerScenario(0.)),
            ("inferior", PowerScenario(0.5)),
            ("equivalent", PowerScenario(0., family_sd=0.1, seed_sd=0.1, query_sd=0.1)),
            ("heavy_tail", PowerScenario(0., heavy_tail_df=5., unequal_family_sizes=True)),
            ("independent_task_fits", PowerScenario(0., shared_global_seeds=False)),
        )):
            with self.subTest(scenario=name):
                result = power_simulation(scenario, repetitions=policy["repetitions"], bootstrap_draws=policy["bootstrap_draws"],
                                          coverage_tolerance=policy["absolute_coverage_tolerance"],
                                          false_positive_tolerance=policy["absolute_false_positive_tolerance"], seed=4200+index)
                evidence[name] = {"scenario": asdict(scenario), "simulation_seed": 4200 + index, "result": result}
                self.assertTrue(result["validation_passed"], result)
                self.assertEqual(result["valid_repetitions"], policy["repetitions"])
                self.assertIsNotNone(result["metrics"]["upper_covered"]["mc95"])
                if name == "inferior":
                    self.assertGreater(result["metrics"]["harmful"]["rate"], 0.8)
                if name == "equivalent":
                    self.assertGreater(result["metrics"]["equivalent"]["rate"], 0.8)
        output = Path(__file__).resolve().parents[3] / "target/vnext/power-development-results.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")

    def test_power_missing_fits_retains_attempts_and_refuses(self):
        result = power_simulation(PowerScenario(0., families=4, fits=3, query_clusters=3, missing_fit_probability=0.5),
                                  repetitions=20, bootstrap_draws=199, seed=220)
        self.assertEqual(result["valid_repetitions"], 0)
        self.assertEqual(sum(result["failed_repetitions"].values()), 20)
        self.assertFalse(result["validation_passed"])
        self.assertIsNone(result["metrics"]["equivalent"]["rate"])
        with self.assertRaises(ValueError):
            PowerScenario(0., heavy_tail_df=1.5)

    def test_power_small_query_sets_refuse_precision(self):
        result = power_simulation(PowerScenario(0., families=4, fits=3, query_clusters=1),
                                  repetitions=20, bootstrap_draws=199, seed=221)
        self.assertEqual(result["valid_repetitions"], 0)
        self.assertEqual(result["failed_repetitions"], {"insufficient_query_clusters": 20})


if __name__ == "__main__":
    unittest.main()
