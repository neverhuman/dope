import unittest

from research.transfer.frontier import (
    ScheduledTask, b_star_tested, deployed_artifact, qualified_learning_density, tested_budget_frontier,
)
from research.transfer.geometry import held_out_status


def task(name, family="f", status="measured", bytes_=40, qualified=True, attempted=True, weight=1.0, **flags):
    return ScheduledTask(family, name, weight, status, bytes_, qualified, attempted, **flags)


class LearningDensityTests(unittest.TestCase):
    def test_states_use_the_frozen_names(self):
        measured = b_star_tested([task("fit", bytes_=40)], (16, 64, 256), quality=1)
        self.assertEqual(measured["state"], "measured")
        self.assertEqual(measured["budget"], 64)
        self.assertEqual(measured["minimum_lies_in"], {"lower_exclusive": 16, "upper_inclusive": 64})
        self.assertIsNone(measured["at_most_budget"])
        self.assertFalse(measured["interpolation"])
        self.assertFalse(measured["lower_bound_claim"])
        censored = b_star_tested([task("only", bytes_=8)], (64,), quality=1)
        self.assertEqual(censored["state"], "censored_below_grid")
        self.assertNotEqual(censored["state"], "measured")
        self.assertEqual(censored["at_most_budget"], 64)
        self.assertIsNone(censored["budget"])
        none = b_star_tested(
            [task("ok"), task("fail", "g", status="technical_failure", bytes_=None, qualified=False)],
            (64, 128), quality=0.75,
        )
        self.assertEqual(none["state"], "no_feasible_tested_artifact")
        self.assertEqual(none["through_budget"], 128)
        self.assertEqual(none["denominator_weight"], 2)
        self.assertEqual(none["points"][0]["coverage"], 0.5)
        self.assertIsNone(none["budget"])
        bent = task("bend", outcomes=((64, True), (128, False)))
        non_monotone = b_star_tested([bent], (64, 128), quality=0.5)
        self.assertEqual(non_monotone["state"], "non_monotone")
        self.assertIsNone(non_monotone["budget"])
        self.assertEqual(non_monotone["first_crossing"], 64)
        self.assertEqual(tuple(point["coverage"] for point in non_monotone["points"]), (1.0, 0.0))
        self.assertIsNone(non_monotone["minimum_lies_in"])
        pending = b_star_tested(
            [task("wait", status="pending", bytes_=None, qualified=False, attempted=False)],
            (64, 128), quality=0.5,
        )
        self.assertEqual(pending["state"], "pending")
        self.assertIsNone(pending["points"])
        self.assertIsNone(pending["budget"])
        self.assertIsNone(pending["first_crossing"])
        self.assertEqual(pending["denominator_weight"], 1)
        with self.assertRaises(ValueError):
            b_star_tested([task("fit")], (64,), quality=0)
        with self.assertRaises(ValueError):
            b_star_tested([task("fit")], (64,), quality=True)

    def test_padding_moves_the_budget_and_not_a_density(self):
        plain = b_star_tested([task("fit", bytes_=40)], (16, 64, 256), quality=1)
        padded = b_star_tested([task("fit", bytes_=200)], (16, 64, 256), quality=1)
        self.assertEqual(plain["budget"], 64)
        self.assertEqual(padded["state"], "measured")
        self.assertEqual(padded["budget"], 256)
        self.assertIsNone(plain["qld"]["value"])
        self.assertIsNone(padded["qld"]["value"])
        self.assertFalse(plain["qld"]["adopted"])
        self.assertFalse(padded["qld"]["release"])

    def test_duplicate_tasks_do_not_add_family_weight(self):
        duplicated = [
            task("a", "f"), task("b", "f"),
            task("c", "g", status="technical_failure", bytes_=None, qualified=False),
        ]
        single = [
            task("a", "f"),
            task("c", "g", status="technical_failure", bytes_=None, qualified=False),
        ]
        duplicated_report = b_star_tested(duplicated, (64,), quality=1)
        single_report = b_star_tested(single, (64,), quality=1)
        self.assertEqual(duplicated_report["points"][0]["coverage"], 0.5)
        self.assertEqual(duplicated_report["points"][0]["coverage"], single_report["points"][0]["coverage"])
        self.assertEqual(duplicated_report["denominator_weight"], single_report["denominator_weight"])
        alone = b_star_tested([task("a", "f"), task("b", "f")], (64,), quality=1)
        self.assertEqual(alone["points"][0]["coverage"], 1.0)
        self.assertEqual(alone["denominator_weight"], 1)

    def test_empty_trivial_and_hidden_bytes_stay_unqualified(self):
        empty = task("e", bytes_=8, empty_generator=True)
        empty_report = b_star_tested([empty], (64, 128), quality=1)
        self.assertEqual(empty_report["state"], "no_feasible_tested_artifact")
        self.assertEqual(empty_report["points"][0]["coverage"], 0.0)
        self.assertEqual(empty_report["qld"]["status"], "ineligible")
        self.assertIsNone(empty_report["qld"]["value"])
        old = tested_budget_frontier([empty], (64, 128))
        self.assertEqual(old["tested"]["64"]["rate"], 0.0)
        self.assertTrue(old["censored_below_grid"])
        trivial = task("t", bytes_=4, trivial_target=True, outcomes=((64, True),))
        trivial_report = b_star_tested([trivial], (64,), quality=1)
        self.assertEqual(trivial_report["points"][0]["coverage"], 0.0)
        self.assertEqual(trivial_report["qld"]["status"], "ineligible")
        hidden = task("h", bytes_=40, external_checkpoint=True)
        hidden_report = b_star_tested([hidden], (64, 128), quality=1)
        self.assertEqual(hidden_report["state"], "no_feasible_tested_artifact")
        self.assertEqual(hidden_report["qld"]["status"], "ineligible")
        charged = b_star_tested([task("d", bytes_=80)], (64, 128), quality=1)
        self.assertEqual(charged["state"], "measured")
        self.assertEqual(charged["budget"], 128)
        self.assertEqual(charged["points"][0]["coverage"], 0.0)
        incomplete = b_star_tested([task("i", complete=False)], (64,), quality=1)
        self.assertEqual(incomplete["points"][0]["coverage"], 0.0)
        self.assertEqual(incomplete["qld"]["status"], "ineligible")

    def test_partial_coverage_keeps_the_full_denominator(self):
        report = b_star_tested(
            [task("ok"), task("fail", "g", status="unfitted", bytes_=None, qualified=False)],
            (64, 128), quality=1,
        )
        self.assertEqual(report["denominator_weight"], 2)
        self.assertEqual(report["points"][1]["coverage"], 0.5)
        self.assertEqual(report["state"], "no_feasible_tested_artifact")
        self.assertNotEqual(report["points"][1]["coverage"], 1)

    def test_one_deployed_seed_and_no_best_of_k(self):
        refused = deployed_artifact((
            {"seed": "s-low", "loss": 0.01},
            {"seed": "s-high", "loss": 0.9},
        ))
        self.assertEqual(refused["status"], "refused")
        self.assertEqual(refused["reason"], "best_of_k_forbidden")
        self.assertIsNone(refused["selected_seed"])
        self.assertEqual(refused["seeds"], ("s-high", "s-low"))
        self.assertNotIn("loss", refused)
        self.assertFalse(refused["selected_by_loss"])
        deployed = deployed_artifact(({"seed": "s-only", "loss": 0.4},))
        self.assertEqual(deployed["status"], "deployed")
        self.assertEqual(deployed["selected_seed"], "s-only")
        self.assertNotIn("loss", deployed)
        with self.assertRaises(ValueError) as duplicate:
            deployed_artifact((
                {"seed": "s-same", "loss": 0.1},
                {"seed": "s-same", "loss": 0.2},
            ))
        self.assertEqual(str(duplicate.exception), "duplicate deployed seed")

    def test_old_censor_flag_is_not_the_b_star_state(self):
        tasks = [task("exact", bytes_=64)]
        old = tested_budget_frontier(tasks, (64, 128))
        new = b_star_tested(tasks, (64, 128), quality=1)
        self.assertFalse(old["censored_below_grid"])
        self.assertEqual(old["tested"]["64"]["rate"], 1)
        self.assertNotIn("state", old)
        self.assertEqual(new["state"], "censored_below_grid")
        self.assertEqual(new["at_most_budget"], 64)
        self.assertIsNone(new["budget"])
        self.assertFalse(new["lower_bound_claim"])
        below = tested_budget_frontier([task("small", bytes_=8)], (64,))
        self.assertTrue(below["censored_below_grid"])
        self.assertEqual(below["tested"]["64"]["rate"], 1)

    def test_qld_has_no_ratio_even_when_coverage_is_verified(self):
        density = qualified_learning_density(
            single_artifact=True, fixed_workload=True, verified_coverage=True, eligible=True,
        )
        self.assertFalse(density["adopted"])
        self.assertEqual(density["status"], "unmeasured")
        self.assertEqual(density["reason"], "unadopted")
        self.assertIsNone(density["value"])
        self.assertFalse(density["aggregate"])
        self.assertFalse(density["headline"])
        self.assertFalse(density["release"])
        ineligible = qualified_learning_density(
            single_artifact=False, fixed_workload=True, verified_coverage=True, eligible=True,
        )
        self.assertEqual(ineligible["status"], "ineligible")
        self.assertIsNone(ineligible["value"])

    def test_overfit_selection_stays_exposed(self):
        exposed = held_out_status("learner", {"losses"})
        self.assertFalse(exposed["held_out"])
        self.assertEqual(exposed["reason"], "selection_exposed")
        held = held_out_status("learner", ())
        self.assertTrue(held["held_out"])


if __name__ == "__main__":
    unittest.main()
