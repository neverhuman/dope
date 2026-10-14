from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import unittest

import numpy as np

from research.transfer.losses import PairedLosses, TrainingScale, training_variance
from research.transfer.metrics import Margin
from research.transfer.power import PowerScenario, power_simulation
from research.transfer.statistics import (
    CrossedDesign, EvaluationRun, bootstrap_effects, cross_task_endpoint, interval_decision,
    leave_one_global_seed_out, paired_family_effect, require_separate_cell_bounds,
    simultaneous_ucb, task_substream,
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
        self.assertEqual(result["estimand"], "method_new_source")
        self.assertIsNone(result["estimates"])
        self.assertIsNone(result["samples"])
        self.assertEqual(result["sensitivity"]["estimand"], "complete_case_sensitivity")
        self.assertFalse(result["sensitivity"]["qualification"])
        self.assertIsNone(simultaneous_ucb(result)["bounds"])
        self.assertEqual(paired_family_effect(incomplete)["estimand"], "complete_case_sensitivity")
        self.assertEqual(incomplete.counts["scheduled_runs"], 12)
        missing_seed = scheduled(panel.runs[1:], ("a",))
        self.assertEqual(bootstrap_effects(missing_seed, draws=199)["status"], "incomplete")
        self.assertEqual(bootstrap_effects(seed_panel(1), draws=199)["status"], "insufficient_families")
        tiny = scheduled((run(values=(1.,)),), ("a",))
        self.assertEqual(bootstrap_effects(tiny, estimand="fixed_artifact", draws=199)["status"], "insufficient_query_clusters")

    def test_crossed_cell_and_missing_fit_do_not_publish_the_complete_case_win(self):
        # Every observed cell is a large negative gap. Dropping or failing one
        # cell must not turn that complete-case mean into the requested estimand.
        def grid(fail=None, drop=None):
            runs = []
            for family in (0, 1):
                for fit in (0, 1):
                    for sample in (0, 1):
                        if drop == (family, fit, sample):
                            continue
                        item = run(f"f-{family}", f"t-{family}", fit=str(fit), sample=str(sample), values=(-5., -5.))
                        if fail == (family, fit, sample):
                            item = replace(item, pairs=None, status="technical_failure")
                        runs.append(item)
            return scheduled(tuple(runs), ("a",))

        complete = bootstrap_effects(grid(), draws=199, seed=3)
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["estimand"], "method_new_source")
        self.assertAlmostEqual(complete["estimates"]["a"], -5.)

        def refused(result, reason):
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["reason"], reason)
            self.assertEqual(result["estimand"], "method_new_source")
            self.assertIsNone(result["estimates"])
            self.assertIsNone(result["family_effects"])
            self.assertIsNone(result["samples"])
            self.assertEqual(result["sensitivity"]["estimand"], "complete_case_sensitivity")
            self.assertAlmostEqual(result["sensitivity"]["estimates"]["a"], -5.)
            self.assertFalse(result["sensitivity"]["qualification"])
            self.assertIsNone(simultaneous_ucb(result)["bounds"])

        refused(bootstrap_effects(grid(drop=(0, 1, 1)), draws=199, seed=3), "unbalanced scheduled evidence")
        missing = bootstrap_effects(grid(fail=(0, 1, 1)), draws=199, seed=3)
        refused(missing, "incomplete scheduled evidence")
        self.assertEqual(missing["scheduled_runs"], 8)
        self.assertEqual(missing["dispositions"], {"measured": 7, "technical_failure": 1})

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
        within_margin_worse = interval_decision(0.02, 0.05, endpoint="gap", equivalence_margin=0.1)
        self.assertTrue(within_margin_worse["equivalent"])
        self.assertFalse(within_margin_worse["superior"])
        adjusted = interval_decision(-0.3, -0.2, endpoint="z")
        self.assertFalse(adjusted["superior"])
        self.assertTrue(adjusted["noninferior"])
        shared = {"cell_id": "cell-a", "bound_kind": "task_cell", "lower": -0.2, "upper": -0.1, "provenance": "prov-a"}
        with self.assertRaises(ValueError):
            require_separate_cell_bounds({"cell-a": shared, "cell-b": shared})
        copied = {**shared}
        with self.assertRaises(ValueError):
            require_separate_cell_bounds({"cell-a": shared, "cell-b": copied})
        relabeled = {**copied, "cell_id": "cell-b"}
        with self.assertRaises(ValueError):
            require_separate_cell_bounds({"cell-a": shared, "cell-b": relabeled})
        require_separate_cell_bounds({
            "cell-a": shared,
            "cell-b": {"cell_id": "cell-b", "bound_kind": "task_cell", "lower": -0.2, "upper": -0.1, "provenance": "prov-b"},
        })
        population = {**shared, "bound_kind": "method_population"}
        with self.assertRaises(ValueError):
            require_separate_cell_bounds({"cell-a": population})
        too_many = {**joint, "draws": 99}
        self.assertEqual(simultaneous_ucb(too_many)["status"], "insufficient_tail_draws")

    def test_cross_task_endpoint_is_dimensionless_and_omits_normalized_gap(self):
        targets, real, synthetic, null = [0., 1., 4.], [0., 2., 3.], [1., 3., 6.], [1., 1., 1.]
        margin = Margin(0.1, 0.02)

        def squared(left, right):
            return tuple((y - p) ** 2 for y, p in zip(left, right))

        def cell(name, y, r, s, n, scale, *, loss="mse", deviation=None):
            item = {
                "task_id": name,
                "loss": loss,
                "pairs": PairedLosses(("a", "b", "c"), ("a", "b", "c"), squared(y, r), squared(y, s), squared(y, n)),
                "margin": margin,
                "fit_scale": scale,
            }
            if deviation is not None:
                item["fit_absolute_deviation"] = deviation
            return item

        def independent_mse(y, r, s, scale):
            total = 0.0
            for real_loss, synthetic_loss in zip(squared(y, r), squared(y, s)):
                tolerance = margin.relative * real_loss + margin.absolute_scaled * scale.value
                total += synthetic_loss - real_loss - tolerance
            return (total / 3) / scale.value

        original_scale = training_variance(targets)
        changed = [7 * v + 30 for v in targets]
        changed_scale = training_variance(changed)
        original = cross_task_endpoint([cell("dollars", targets, real, synthetic, null, original_scale)])
        converted = cross_task_endpoint([
            cell("dollars", changed, [7 * v + 30 for v in real], [7 * v + 30 for v in synthetic],
                 [7 * v + 30 for v in null], changed_scale),
        ])
        self.assertEqual(original["format"], "dope-cross-task-endpoint")
        self.assertEqual(original["version"], 1)
        self.assertEqual(original["campaign"], "unmeasured")
        self.assertEqual(original["calibration"], "unmeasured")
        self.assertIs(original["qualification"], False)
        self.assertAlmostEqual(original["value"], independent_mse(targets, real, synthetic, original_scale))
        self.assertAlmostEqual(original["value"], converted["value"])
        self.assertAlmostEqual(converted["tasks"][0]["raw_z"], 49 * original["tasks"][0]["raw_z"])
        gap = sum(squared(targets, synthetic)) / 3 - sum(squared(targets, real)) / 3
        normalized = gap / original_scale.value
        self.assertAlmostEqual(original["tasks"][0]["normalized_gap"], normalized)
        self.assertNotAlmostEqual(original["value"], normalized)
        self.assertGreater(abs(original["value"] - normalized), 1e-8)
        degenerate = cross_task_endpoint([
            cell("flat", targets, real, synthetic, null, training_variance([5., 5.])),
            cell("dollars", targets, real, synthetic, null, original_scale),
        ])
        self.assertEqual(degenerate["tasks"][0]["status"], "degenerate_scale")
        self.assertIsNone(degenerate["tasks"][0]["dimensionless"])
        self.assertAlmostEqual(degenerate["value"], original["value"])
        alone = cross_task_endpoint([cell("flat", targets, real, synthetic, null, training_variance([5., 5.]))])
        self.assertIsNone(alone["value"])
        self.assertEqual(alone["status"], "unpooled_degenerate_scale")
        mad = sum(abs(v - (sum(targets) / 3)) for v in targets) / 3
        mae = cross_task_endpoint([
            cell("dollars", targets, real, synthetic, null, original_scale, loss="mae", deviation=mad),
        ])
        mae_converted = cross_task_endpoint([
            cell("dollars", changed, [7 * v + 30 for v in real], [7 * v + 30 for v in synthetic],
                 [7 * v + 30 for v in null], changed_scale, loss="mae", deviation=7 * mad),
        ])
        self.assertAlmostEqual(mae["value"], mae_converted["value"])
        self.assertNotAlmostEqual(
            mae["value"],
            cross_task_endpoint([
                cell("dollars", targets, real, synthetic, null, original_scale, loss="mae", deviation=mad * 2),
            ])["value"],
        )
        rmse = cross_task_endpoint([cell("dollars", targets, real, synthetic, null, original_scale, loss="rmse")])
        real_rmse = math.sqrt(sum(squared(targets, real)) / 3)
        synthetic_rmse = math.sqrt(sum(squared(targets, synthetic)) / 3)
        width = math.sqrt(original_scale.value)
        rmse_z = (synthetic_rmse - real_rmse - (margin.relative * real_rmse + margin.absolute_scaled * width)) / width
        self.assertAlmostEqual(rmse["value"], rmse_z)
        with self.assertRaises(ValueError):
            cross_task_endpoint([cell("dollars", targets, real, synthetic, null, original_scale)], version=2)
        with self.assertRaises(ValueError):
            cross_task_endpoint([cell("dollars", targets, real, synthetic, null, original_scale, loss="mae")])

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
        self.assertEqual(result["failed_repetitions"], {"incomplete": 20})
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
