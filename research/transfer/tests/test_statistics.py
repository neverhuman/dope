from dataclasses import asdict, replace
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from research.transfer.losses import PairedLosses, TrainingScale, training_variance
from research.transfer.metrics import Margin
from research.transfer.power import PowerScenario, heavy_tail_sensitivity, power_simulation, seeded_stress_cases
from research.transfer.inference import design_from_payload
from research.transfer.tdl.aggregate import evaluate as evaluate_aggregate
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
    def test_crossed_stress_goldens_and_robust_sensitivity(self):
        cases = seeded_stress_cases()
        self.assertEqual(set(cases), {"near_threshold", "low_gain", "heavy_tailed", "correlated", "few_source"})
        for name, payload in cases.items():
            with self.subTest(name=name):
                path = Path(__file__).resolve().parents[3] / f"tests/golden/aggregate/c02_crossed_{name}.json"
                case = json.loads(path.read_text())
                self.assertEqual(payload, case["input"])
                self.assertEqual(evaluate_aggregate(payload), case["expected"])
                self.assertFalse(case["expected"]["qualification"])
                if name == "correlated":
                    bounds = case["expected"]["bounds"]
                    self.assertEqual(bounds["gap/1024/tree"], bounds["gap/2048/neural"])
                elif name == "few_source":
                    self.assertEqual(case["expected"]["status"], "insufficient_families")
        design = design_from_payload(cases["heavy_tailed"])
        sensitivity = heavy_tail_sensitivity(design)
        self.assertEqual(sensitivity["status"], "sensitivity_only")
        self.assertFalse(sensitivity["primary_losses_clipped"])
        self.assertIsNone(sensitivity["bounds"])
        self.assertEqual(sensitivity, json.loads((Path(__file__).resolve().parents[3] / "tests/golden/aggregate/c02_crossed_heavy_tailed.json").read_text())["robust_expected"])
        self.assertNotAlmostEqual(sensitivity["estimates"][design.required_methods[0]], paired_family_effect(design)["estimates"][design.required_methods[0]])

    def test_wrong_cluster_learner_source_and_fit_seed_alias_refusal(self):
        design = design_from_payload(seeded_stress_cases()["correlated"])
        for unit in ("query_row", "learner_seed", "fit_seed"):
            with self.subTest(unit=unit), self.assertRaisesRegex(ValueError, "invalid_cluster_unit"):
                replace(design, cluster_unit=unit)
        alias = replace(design.runs[0], fit_id="aliased-fit")
        with self.assertRaisesRegex(ValueError, "duplicate_fit_seed"):
            scheduled((*design.runs, alias), design.required_methods)
        for field, value in (("cluster_unit", "PRIVATE-source-header-extrema"), ("required_methods", []), ("bootstrap_draws", True)):
            payload = deepcopy(seeded_stress_cases()["near_threshold"])
            payload[field] = value
            self.assertEqual(evaluate_aggregate(payload), {"status": "failed", "reason": "invalid_crossed_design"})
        self.assertEqual(task_substream(11, "task", "fit"), 9525942615105419655)
        self.assertEqual(task_substream(11, "tâche/水", "fit"), 9973167365048566836)
        payload = deepcopy(seeded_stress_cases()["correlated"])
        payload["bootstrap_seed"] = task_substream(11, "tâche/水", "fit")
        case = json.loads((Path(__file__).resolve().parents[3] / "tests/golden/aggregate/c02_crossed_uint64_substream.json").read_text())
        self.assertEqual(payload, case["input"])
        self.assertEqual(evaluate_aggregate(payload), case["expected"])

    def test_crossed_tasks_and_nested_paired_seeds_remain_random(self):
        design = design_from_payload(seeded_stress_cases()["correlated"])
        runs = []
        for item in design.runs:
            for sample in (0, 1):
                for learner in (0, 1):
                    values = tuple(v + sample + 2 * learner for v in item.pairs.synthetic)
                    runs.append(replace(item, sample_id=str(sample), learner_seed_id=str(learner), pairs=replace(item.pairs, synthetic=values)))
        nested = bootstrap_effects(scheduled(runs, design.required_methods), draws=399, seed=31)
        np.testing.assert_array_equal(nested["samples"][design.required_methods[0]], nested["samples"][design.required_methods[1]])
        # Source replication cannot turn a shared task shock into precision.
        runs = []
        for source in range(20):
            for task in range(2):
                for fit in range(2):
                    item = run(str(source), f"s{source}/t{task}", fit=str(fit), values=((-3. if task == 0 else 3.),) * 2,
                               namespace="shared", seed=str(fit))
                    runs.append(replace(item, task_axis_id=str(task)))
        boot = bootstrap_effects(scheduled(runs, ("a",)), draws=399, seed=31)
        self.assertGreater(np.var(boot["samples"]["a"]), 3.)
        incomplete = scheduled(runs[2:], ("a",))
        refused = bootstrap_effects(incomplete, draws=199)
        self.assertEqual(refused["sensitivity"]["estimand"], "complete_case_sensitivity")
        self.assertIsNone(simultaneous_ucb(refused)["bounds"])

    def test_max_t_retains_family_and_stepdown_p_order(self):
        base = bootstrap_effects(seed_panel(4), draws=399, seed=19)
        sample = np.asarray(base["samples"]["a"])
        joint = {**base, "estimates": {"strong": 5., "weak": 0.}, "samples": {"strong": sample + 5, "weak": sample}}
        report = simultaneous_ucb(joint)
        self.assertEqual(report["bound_method"], "full_family_max_t")
        self.assertLessEqual(report["bounds"]["strong"]["lower_stepdown_p"], report["bounds"]["weak"]["lower_stepdown_p"])
        single = simultaneous_ucb(base)["bounds"]["a"]
        self.assertAlmostEqual(report["bounds"]["weak"]["lower"], single["lower"])
        self.assertAlmostEqual(report["bounds"]["strong"]["lower"] - 5., single["lower"])
        flat = {**base, "samples": {"a": [0.] * 399}}
        self.assertEqual(simultaneous_ucb(flat)["status"], "insufficient_variation")
        rng = np.random.default_rng(36)
        two = {**base, "estimates": {"a": 0., "b": 0.}, "samples": {"a": rng.normal(size=399), "b": rng.normal(size=399)}}
        one = {**two, "estimates": {"a": 0.}, "samples": {"a": two["samples"]["a"]}}
        self.assertLess(simultaneous_ucb(two)["bounds"]["a"]["lower"], simultaneous_ucb(one)["bounds"]["a"]["lower"])

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

    def test_inferential_outputs_carry_finite_variance_assumptions(self):
        def check(payload):
            self.assertEqual(payload["moment_assumption"], "finite_variance")
            self.assertIs(payload["distribution_free"], False)

        complete = bootstrap_effects(seed_panel(4), draws=199, seed=1)
        check(complete)
        refused = bootstrap_effects(seed_panel(1), draws=199)
        self.assertEqual(refused["status"], "insufficient_families")
        check(refused)
        check(simultaneous_ucb(complete))
        check(simultaneous_ucb(refused))
        check(simultaneous_ucb({**complete, "draws": 99}))
        self.assertEqual(simultaneous_ucb({**complete, "draws": 99})["status"], "insufficient_tail_draws")
        narrow = interval_decision(-0.01, 0.01, equivalence_margin=0.1)
        wide = interval_decision(-0.1, 0.1, equivalence_margin=0.1)
        check(narrow)
        check(wide)
        self.assertTrue(narrow["equivalent"])
        self.assertFalse(wide["equivalent"])


if __name__ == "__main__":
    unittest.main()
