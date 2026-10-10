"""Protect pairing, missing outcomes, receipt pins and manuscript regeneration."""
import copy
import json
import unittest
from unittest.mock import patch

import retained_evidence as evidence
from compute_panel import summarize
from public_hardware import banned_hits


class RetainedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = evidence.load_sources()
        cls.generated = json.loads((evidence.OUT / "retained-evidence.json").read_bytes())

    def test_lineage_groups_do_not_count_seeds_as_replicates(self):
        groups = evidence.groups_from_rows(self.docs["classical"]["rows"])
        self.assertEqual(len(groups), 1216)
        self.assertEqual(groups, self.generated["groups"])
        self.assertTrue(all(len(x["source_pointers"]) == 3 for x in groups))
        for group in groups:
            rows = [self.docs["classical"]["rows"][int(p.rsplit("/", 1)[1])] for p in group["source_pointers"]]
            self.assertEqual({row["dataset"] for row in rows}, {group["dataset"]})

    def test_incomplete_or_duplicate_seed_group_is_rejected(self):
        rows = self.docs["classical"]["rows"]
        with self.assertRaisesRegex(ValueError, "sample group"):
            evidence.groups_from_rows(rows[:-1])
        with self.assertRaisesRegex(ValueError, "sample group"):
            evidence.groups_from_rows(rows + [rows[0]])

    def test_undeclared_method_or_configuration_is_rejected(self):
        rows = self.docs["classical"]["rows"]
        trio = [rows[int(p.rsplit("/", 1)[1])] for p in self.generated["groups"][0]["source_pointers"]]
        for change in ({"method": "Mystery"}, {"selection_binding": "undeclared"}):
            with self.subTest(change=sorted(change)):
                with self.assertRaisesRegex(ValueError, "undeclared method"):
                    evidence.groups_from_rows(rows + [{**row, **change} for row in trio])

    def test_changed_receipt_pin_fails_before_metric_reduction(self):
        relative, _ = evidence.SOURCES["classical"]
        with patch.dict(evidence.SOURCES, {"classical": (relative, "0" * 64)}):
            with self.assertRaisesRegex(ValueError, "source drift"):
                evidence.load_sources()

    def test_unavailable_values_are_not_imputed(self):
        group = self.generated["groups"][0]
        rows = [copy.deepcopy(self.docs["classical"]["rows"][int(p.rsplit("/", 1)[1])]) for p in group["source_pointers"]]
        rows[0]["alpha_precision"] = None
        actual = evidence.groups_from_rows(rows)[0]
        self.assertIsNone(actual["values"]["alpha_precision"])

    def test_paired_differences_use_shared_lineages(self):
        groups = self.generated["groups"]
        for pair in self.generated["matched8_paired"]:
            method, config = pair["right"].split("/")
            left = {g["dataset"]: g["values"][pair["metric"]] for g in groups if g["method"] == "TabSyn" and g["size"] == 4}
            right = {g["dataset"]: g["values"][pair["metric"]] for g in groups if g["method"] == method and g["configuration"] == config and g["size"] == 4}
            self.assertLessEqual(pair["n"], 8)
            self.assertEqual(pair["differences"], [left[k] - right[k] for k in pair["datasets"]])
        self.assertEqual(len(self.generated["matched8_paired"]), 24)

    def test_summary_is_a_lineage_median_with_reproducible_interval(self):
        for row in self.generated["summaries"]:
            values = [g["values"]["catboost_retention"] for g in self.generated["groups"]
                      if (g["method"], g["configuration"], g["size"]) == (row["method"], row["configuration"], row["size"])
                      and (row["scope"] == "classical100" or g["dataset"] in {x["dataset"] for x in self.generated["groups"] if x["method"] == "TabSyn"})
                      and evidence.finite(g["values"]["catboost_retention"])]
            self.assertEqual(summarize(values), row["metrics"]["catboost_retention"])

    def test_arf_intervals_keep_original_receipt_point_estimates(self):
        for pair in self.generated["arf_paired"]:
            original = next(r for r in self.docs["arf"]["paired_descriptive"]
                            if r["configuration"] == "features12_steps2048" and r["size_multiplier"] == 4
                            and r["reference_configuration"] == pair["configuration"] and r["auditor"] == pair["auditor"])
            self.assertEqual(pair["n"], original["paired_complete_informative_lineages"])
            self.assertAlmostEqual(pair["median_difference"], original["median_paired_difference"], places=13)
            self.assertAlmostEqual(pair["left_median"], original["dope_median_retention"], places=13)
            self.assertAlmostEqual(pair["right_median"], original["reference_median_retention"], places=13)

    def test_small_cohort_cannot_claim_finite_simultaneous_interval(self):
        self.assertTrue(evidence.median_interval(list(range(8)), 24)["unbounded"])
        for pair in self.generated["matched8_paired"]:
            self.assertTrue(pair["simultaneous_median_interval"]["unbounded"])
            self.assertNotIn("familywise_lo", pair)

    def test_finite_order_interval_meets_binomial_tail_bound(self):
        interval = evidence.median_interval(list(range(97)), 6)
        self.assertFalse(interval["unbounded"])
        k = interval["order_index"]
        self.assertLessEqual(2 * evidence.stats.binom.cdf(k - 1, 97, .5), .05 / 6)
        self.assertEqual(interval["lo"], k - 1)
        self.assertEqual(interval["hi"], 97 - k)

    def test_tex_regenerates_without_private_tokens(self):
        for name, text in evidence.render_tables(self.generated, self.docs["first12"]).items():
            self.assertEqual(text, (evidence.OUT / name).read_text())
            self.assertEqual(banned_hits(text), [])

    def test_first12_not_pooled_and_scores_remain_null(self):
        self.assertEqual(len(self.docs["first12"]["cells"]), 12)
        self.assertIn("no bootstrap or method ranking", self.generated["procedure"]["first12"])
        self.assertFalse(self.generated["official_tests_opened"])
        for name in ("mfs_v2", "ptf_v1", "superiority"):
            self.assertIsNone(self.generated[name])

    def test_coverage_is_published_receipts_with_pending_outcomes(self):
        data = json.loads((evidence.OUT / "baseline-coverage.json").read_bytes())
        self.assertNotIn("calculated_slot_hours", data)
        rows = {r["method"]: r for r in data["rows"]}
        self.assertEqual(rows["TabSyn"]["complete_retained_sample_cells"], 48)
        self.assertTrue(rows["ARF"]["complete100"])
        self.assertTrue(all(not r["final_five_fit_complete"] for r in rows.values()))
        for method in ("TabSyn", "TabDDPM", "Forest-Flow", "Forest-Diffusion"):
            self.assertFalse(rows[method]["complete100"])
            self.assertFalse(rows[method]["eta_is_observed_completion"])
            self.assertIsNone(rows[method]["full100_eta_mt"])
            self.assertEqual(rows[method]["remaining_campaign_status"], "pending")


if __name__ == "__main__":
    unittest.main()
