"""Landmark student reports stay distinct from measured reports."""
from __future__ import annotations

import unittest

from research.levels.landmark_student import (
    SLOTS,
    StudentError,
    cost_record,
    distill,
    evidence,
    matched_modes,
    measured_report,
    measured_support,
    predict,
    recover_candidate,
    require_measured,
    require_same_context,
)
from research.levels.optional_landmarks import LANDMARK_SLOTS


def _mask(indexes: tuple[int, ...]) -> tuple[bool, ...]:
    return tuple(index in indexes for index in range(SLOTS))


def _values(indexes: tuple[int, ...]) -> tuple[object, ...]:
    return tuple(float(index + 1) if index in indexes else None for index in range(SLOTS))


class LandmarkStudentTests(unittest.TestCase):
    def test_slot_count_matches_the_observed_mask(self):
        self.assertEqual(SLOTS, LANDMARK_SLOTS)

    def test_predicted_numbers_stay_distinct_from_measured(self):
        full = tuple(range(SLOTS))
        measured = measured_report(_mask(full), _values(full))
        self.assertIs(require_measured(measured), measured)
        empty = measured_report(_mask(()), _values(()))
        filled = predict(empty, {index: float(index + 1) for index in full})
        self.assertEqual(measured_support(filled), ())
        self.assertEqual(filled.mode, "none")
        self.assertEqual(
            tuple(slot.value for slot in filled.slots),
            tuple(slot.value for slot in measured.slots),
        )
        self.assertNotEqual(filled.slots, measured.slots)
        with self.assertRaises(StudentError) as caught:
            require_measured(filled)
        self.assertEqual(str(caught.exception), "student_provenance")
        self.assertNotIn("1.0", str(caught.exception))

    def test_unobserved_zero_is_not_a_measured_slot(self):
        values = _values(())
        values = (0.0,) + values[1:]
        with self.assertRaises(StudentError) as caught:
            measured_report(_mask(()), values)
        self.assertEqual(str(caught.exception), "student_provenance")
        self.assertNotIn("0.0", str(caught.exception))
        with self.assertRaises(StudentError) as nonfinite:
            measured_report(_mask((0,)), (float("inf"),) + (None,) * (SLOTS - 1))
        self.assertEqual(str(nonfinite.exception), "student_nonfinite")
        self.assertNotIn("inf", str(nonfinite.exception))

    def test_modes_share_the_table_and_prediction_does_not_grow_support(self):
        masks = {
            "none": _mask(()),
            "partial": _mask((0, 2)),
            "full": _mask(tuple(range(SLOTS))),
        }
        budgets = matched_modes("table-a", "target-a", masks)
        require_same_context(budgets["none"], budgets["full"])
        partial = measured_report(masks["partial"], _values((0, 2)))
        resumed = predict(partial, {1: 0.25})
        self.assertEqual(measured_support(resumed), budgets["partial"].support)
        self.assertEqual(resumed.support, (0, 2))
        other = matched_modes("table-b", "target-a", masks)
        with self.assertRaises(StudentError) as caught:
            require_same_context(budgets["partial"], other["partial"])
        self.assertEqual(str(caught.exception), "student_budget")
        self.assertNotIn("table-b", str(caught.exception))
        with self.assertRaises(StudentError) as mode:
            matched_modes("table-a", "target-a", {**masks, "none": masks["full"], "full": masks["none"]})
        self.assertEqual(str(mode.exception), "student_mode")

    def test_distill_uses_the_same_candidates_and_does_not_promote(self):
        with self.assertRaises(StudentError) as missing:
            distill({"kept": 1.0, "dropped": None})
        self.assertEqual(str(missing.exception), "student_support")
        fitted = distill({"kept": 3, "other": 1})
        self.assertEqual(fitted["candidates"], ("kept", "other"))
        self.assertFalse(fitted["promoted"])
        self.assertAlmostEqual(sum(fitted["student"]), 1.0)
        self.assertLess(fitted["loss"], fitted["initial_loss"])

    def test_retention_and_forward_time_are_not_a_decision(self):
        for kind in ("synthetic_retention", "embedding_cosine"):
            record = evidence(kind)
            self.assertFalse(record["promoted"])
            self.assertEqual(record["status"], "not_a_decision")
        decision = evidence("jope_decision")
        self.assertFalse(decision["promoted"])
        self.assertEqual(decision["status"], "unmeasured")
        forward_only = cost_record({"forward": 1.5})
        self.assertFalse(forward_only["complete"])
        self.assertFalse(forward_only["quality_claim"])
        complete = cost_record({"forward": 1.5, "prepare": 8.0})
        self.assertTrue(complete["complete"])
        self.assertFalse(complete["quality_claim"])
        self.assertEqual(recover_candidate()["recovered"], False)
        self.assertEqual(recover_candidate()["code"], "student_unregistered")


if __name__ == "__main__":
    unittest.main()
