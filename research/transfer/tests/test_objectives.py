import unittest

from research.transfer.allocation import allocation_gain, provenance_row, search_coverage
from research.transfer.objectives import label_objective
from research.transfer.probes import sketch_status
from research.transfer.selection_regret import select_once


def candidate(name, **updates):
    row = {"candidate_id": name, "parent_id": None, "estimated_bytes": 20, "actual_bytes": 20,
           "cost": 1.0, "development_loss": 0.4, "risk_sketch": "ok", "held_out_oracle": None,
           "cheaper": False, "stored_target_fit": 0.1}
    row.update(updates)
    return row


class ObjectiveTests(unittest.TestCase):
    def test_selection_refuses_oracle_overcap_and_mislabeled_utility(self):
        stored = label_objective("stored_target_fit", 0.2)
        self.assertEqual(stored["label"], "stored_target_fit")
        self.assertIsNone(stored["retraining_utility"])
        with self.assertRaises(ValueError):
            label_objective("retraining_utility", 0.2)
        self.assertEqual(sketch_status(None), "absent")
        self.assertEqual(sketch_status(False), "failed")
        chosen = select_once([
            candidate("oracle", held_out_oracle=0.01, development_loss=0.01),
            candidate("expensive", estimated_bytes=30, actual_bytes=30, development_loss=0.2, risk_sketch="failed"),
            candidate("cheap", development_loss=0.5, cheaper=True),
            candidate("estimate-only", estimated_bytes=10, actual_bytes=80),
        ], cap=32)
        self.assertEqual(chosen["selected"], "cheap")
        self.assertIsNone(chosen["retraining_utility"])
        self.assertFalse(chosen["power_measured"])
        self.assertFalse(chosen["qualification"])
        reasons = {item["reason"] for item in chosen["rejections"]}
        self.assertIn("held_out_oracle", reasons)
        self.assertIn("sketch_failed", reasons)
        self.assertIn("actual_over_cap", reasons)
        with self.assertRaises(ValueError):
            select_once([candidate("cheap", cheaper=True)], cap=32, frozen=True)
        self.assertFalse(allocation_gain(None, 1.0)["attributable"])
        self.assertTrue(allocation_gain(0.4, 0.25)["attributable"])
        self.assertFalse(search_coverage([("a", "b"), ("a",)])["complete"])
        self.assertTrue(search_coverage([("b", "a"), ("a", "b")])["complete"])
        row = provenance_row(candidate_id="cheap", parent_id="parent", estimated_bytes=20,
                             actual_bytes=20, cost=1.0, rejection=None)
        self.assertEqual(row["parent_id"], "parent")
        self.assertEqual(row["actual_bytes"], 20)


if __name__ == "__main__":
    unittest.main()
