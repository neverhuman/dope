"""The lineage appendix keeps incomplete auditors blank and does not invent fitness."""
import unittest

from research.benchmark.publish_lineage_appendix import group_retention, lineage_record, panel_median


def cell(seed, retention, informative=True, status="ok", bytes_value=1000, within=True):
    utility = {
        auditor: {"informative": informative, "retention": retention if informative else None}
        for auditor in ("catboost", "linear", "mlp")
    }
    return {
        "sample_seed": seed, "status": status, "utility": utility if status == "ok" else None,
        "charged_artifact_bytes": bytes_value, "artifact_within_l3_cap": within,
        "copy_counts": {"exact": 0, "near": 1}, "real_vs_real_control_counts": {"exact": 0, "near": 0},
        "mfs_v2": None, "unavailable_reason": None if status == "ok" else status,
    }


class LineageAppendix(unittest.TestCase):
    def test_incomplete_group_stays_blank(self):
        cells = [cell(101, 0.9), cell(211, 0.8, informative=False), cell(307, 0.7)]
        group = group_retention(cells)
        self.assertIsNone(group["utility"]["catboost"]["median_retention"])
        self.assertFalse(group["utility"]["catboost"]["complete_informative_sample_group"])
        self.assertEqual(group["utility"]["catboost"]["blank_reason"], "not_informative")

    def test_three_informative_seeds_use_the_middle_value(self):
        group = group_retention([cell(101, 0.2), cell(211, 0.9), cell(307, 0.4)])
        self.assertEqual(group["utility"]["catboost"]["median_retention"], 0.4)

    def test_record_keeps_every_dataset_and_a_null_score(self):
        groups = {}
        for dataset in ("b", "a"):
            for size in (1, 4):
                groups[(dataset, size)] = group_retention([cell(101, 1.0), cell(211, 1.0), cell(307, 1.0)])
        rows = lineage_record({"a", "b"}, groups, {"a": "alpha", "b": "beta"},
                              {"a": {"fit_rows": 80, "features": 4}, "b": {"fit_rows": None, "features": 3}})
        self.assertEqual([row["dataset"] for row in rows], ["a", "b"])
        self.assertEqual(panel_median(rows, "catboost", 4), (2, 1.0))
        self.assertIsNone(rows[1]["fit_rows"])

    def test_cap_flag_cannot_disagree_with_bytes(self):
        with self.assertRaises(ValueError):
            group_retention([cell(101, 0.5, bytes_value=20000, within=True),
                             cell(211, 0.5, bytes_value=20000, within=True),
                             cell(307, 0.5, bytes_value=20000, within=True)])
