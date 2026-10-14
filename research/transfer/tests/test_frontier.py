import unittest

from research.transfer.frontier import ScheduledTask, tested_budget_frontier


def task(name, family, status="measured", bytes_=40, qualified=True, attempted=True, weight=1.0):
    return ScheduledTask(family, name, weight, status, bytes_, qualified, attempted)


class FrontierTests(unittest.TestCase):
    def test_full_denominator_and_frontier_states(self):
        tasks = [task("ok", "f"), task("fail", "g", status="technical_failure", bytes_=None, qualified=False)]
        report = tested_budget_frontier(tasks, (64, 128))
        self.assertEqual(report["tested"]["64"]["denominator_weight"], 2)
        self.assertEqual(report["tested"]["64"]["rate"], 0.5)
        self.assertEqual(report["tested"]["128"]["rate"], 0.5)
        self.assertFalse(report["no_feasible_tested_artifact"])
        self.assertFalse(report["interpolation"])
        self.assertFalse(report["lower_bound_claim"])
        over = tested_budget_frontier([task("big", "f", bytes_=200)], (64, 128))
        self.assertEqual(over["tested"]["128"]["rate"], 0)
        self.assertTrue(over["no_feasible_tested_artifact"])
        censored = tested_budget_frontier([task("small", "f", bytes_=8)], (64,))
        self.assertTrue(censored["censored_below_grid"])
        self.assertEqual(censored["tested"]["64"]["rate"], 1)
        pending = tested_budget_frontier([task("wait", "f", status="pending", bytes_=None, qualified=False, attempted=False)], (64,))
        self.assertEqual(pending["status"], "refused")
        self.assertIsNone(pending["tested"])
        bent = tested_budget_frontier(tasks, (64, 128), measured_points=((64, 0.4), (128, 0.2)))
        self.assertTrue(bent["non_monotone"])
        self.assertFalse(bent["interpolation"])


if __name__ == "__main__":
    unittest.main()
