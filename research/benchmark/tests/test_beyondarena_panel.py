"""The BeyondArena evaluation lock keeps every eligible family and no invented median."""

import json
import sys
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from publish_beyondarena_retention import NOT_MEASURED, publish, validate_receipt
from select_beyondarena_panel import HISTORICAL_PILOT, LOCK_PATH, build_lock, pmlb_display_names


def _floats(value):
    found = []
    if isinstance(value, bool):
        return found
    if isinstance(value, float):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_floats(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_floats(item))
    return found


class BeyondArenaPanelTests(unittest.TestCase):
    def test_lock_is_the_full_eligible_set_and_excludes_the_pmlb_panel(self):
        lock = build_lock()
        committed = json.loads(LOCK_PATH.read_text())
        self.assertEqual(lock, committed)
        self.assertEqual(lock["version"], 3)
        self.assertEqual(lock["eligible_count"], 73)
        self.assertEqual(lock["selected_count"], 73)
        self.assertEqual(lock["bucket_family_count"], 142)
        self.assertEqual(
            lock["exclusion_reason_counts"],
            {"redistribution_rights": 50, "task_not_binary_or_regression": 25},
        )
        names = [row["name"] for row in lock["datasets"]]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), {row["name"] for row in lock["datasets"]})
        pmlb = pmlb_display_names()
        self.assertEqual(len(pmlb), 100)
        self.assertTrue(set(names).isdisjoint(pmlb))
        self.assertTrue(set(HISTORICAL_PILOT).issubset(names))
        self.assertIn("lung_cancer_epithelial_genexp", names)
        self.assertFalse(lock["pooled_with_s3_100"])
        self.assertFalse(lock["pooled_with_public_headline"])
        self.assertFalse(lock["official_tests_opened"])
        self.assertFalse(lock["outer_fold_authorized"])
        self.assertIsNone(lock["mfs_v2"])
        self.assertEqual(lock["displayed_configuration"], "features12_steps2048")
        self.assertEqual(lock["wave_1_fold"], "inner_validation")
        tasks = {row["task"] for row in lock["datasets"]}
        self.assertEqual(tasks, {"binary_classification", "regression"})

    def test_missing_receipt_publishes_not_measured_and_no_float(self):
        rows = publish()
        self.assertEqual(rows["receipt"], "absent")
        self.assertEqual(rows["accepted_cells"], 0)
        self.assertFalse(rows["pooled_with_s3_100"])
        self.assertIsNone(rows["mfs_v2"])
        self.assertEqual(len(rows["blocks"]), 2 * 10 * 3)
        self.assertTrue(all(block["median_retention"] == NOT_MEASURED for block in rows["blocks"]))
        self.assertTrue(all(block["n"] is None for block in rows["blocks"]))
        self.assertEqual(_floats(rows), [])
        committed = json.loads((BENCH / "results" / "beyondarena-retention-rows.json").read_text())
        self.assertEqual(rows, committed)

    def test_a_median_or_a_pooled_flag_is_rejected(self):
        lock = build_lock()
        with self.assertRaises(ValueError):
            publish({"median_retention": 0.9, "cells": []}, lock)
        with self.assertRaises(ValueError):
            publish({"pooled_with_s3_100": True, "cells": []}, lock)
        with self.assertRaises(ValueError):
            publish({"mfs_v2": 1, "cells": []}, lock)
        with self.assertRaises(ValueError):
            publish({"official_tests_opened": True, "cells": []}, lock)
        cell = {
            "family": "airfoil_self_noise",
            "task": "regression",
            "method": "DOPE",
            "auditor": "catboost",
            "status": "ok",
            "retention": 0.5,
            "median_retention": 0.5,
        }
        with self.assertRaises(ValueError):
            validate_receipt({"cells": [cell]}, lock)

    def test_accepted_cells_do_not_become_a_median(self):
        lock = build_lock()
        receipt = {
            "fold": "inner_validation",
            "configuration": "features12_steps2048",
            "pooled_with_s3_100": False,
            "official_pmlb_tests_opened": False,
            "mfs_v2": None,
            "cells": [
                {
                    "family": "airfoil_self_noise",
                    "task": "regression",
                    "method": "DOPE",
                    "auditor": "catboost",
                    "status": "ok",
                    "retention": 0.94,
                }
            ],
        }
        rows = publish(receipt, lock)
        self.assertEqual(rows["receipt"], "accepted")
        self.assertEqual(rows["accepted_cells"], 1)
        self.assertTrue(all(block["median_retention"] == NOT_MEASURED for block in rows["blocks"]))
        self.assertEqual(_floats(rows), [])

    def test_a_family_outside_the_lock_is_rejected(self):
        lock = build_lock()
        receipt = {
            "cells": [
                {
                    "family": "not_a_family",
                    "method": "DOPE",
                    "auditor": "catboost",
                    "status": "ok",
                    "retention": 0.1,
                }
            ]
        }
        with self.assertRaises(ValueError):
            publish(receipt, lock)


if __name__ == "__main__":
    unittest.main()
