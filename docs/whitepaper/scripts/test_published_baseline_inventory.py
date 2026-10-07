"""Verify that paper coverage and figure values trace to merged publications."""
import copy
import hashlib
import json
import unittest
from collections import Counter
from statistics import mean
from unittest.mock import patch
import jsonschema

import published_baseline_inventory as inventory
from public_hardware import banned_hits


class PublishedBaselineInventory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = inventory.load_sources()
        cls.data = inventory.build(cls.docs)
        cls.out = inventory.REPO / "docs/whitepaper/generated"
        cls.rows = {r["method"]: r for r in cls.data["rows"]}

    def test_all_sources_reject_drift_before_reduction(self):
        for name, (path, _) in list(inventory.SOURCES.items()):
            with self.subTest(source=name), patch.dict(inventory.SOURCES, {name:(path, "0"*64)}):
                with self.assertRaisesRegex(ValueError, "source drift"):
                    inventory.load_sources()

    def resolve(self, ref):
        key = next(k for k in inventory.SOURCES if inventory.source_ref(k, "")["path"] == ref["path"])
        self.assertEqual(ref["sha256"], inventory.SOURCES[key][1])
        value = self.docs[key]
        for token in ref["pointer"].split("/")[1:]:
            value = value[int(token)] if isinstance(value, list) else value[token]
        return key, value

    def test_every_point_recomputes_from_receipt_pointers(self):
        for point in self.data["figure_points"]:
            resolved = [self.resolve(r) for r in point["sources"]]
            if point["aggregation"] == "mean_of_three_sample_seeds":
                cells = [c for _, c in resolved]
                self.assertEqual(sorted(c["sample_seed"] for c in cells), [101,211,307])
                self.assertEqual({c["dataset"] for c in cells}, {point["dataset"]})
                self.assertEqual({c["row_multiplier"] for c in cells}, {4})
                self.assertEqual(len({(c["fit_seed"],c["config_sha256"]) for c in cells}), 1)
                values = [inventory.scalar(c,point["metric"]) if "metrics" in c else c[point["metric"]] for c in cells]
                expected = sum(values)/3 if all(inventory.finite(x) for x in values) else None
            else:
                key, stat = resolved[0]
                # Check the aggregate against independent fit means as well as
                # the published summary. Fit SD is never treated as a CI.
                values = [f["metrics"][point["metric"]] for f in self.docs[key]["per_fit"]
                          if f["dataset"] == point["dataset"] and f["row_multiplier"] == 4]
                finite = [x for x in values if inventory.finite(x)]
                self.assertEqual(len(values), 5)
                self.assertEqual(len(finite), stat["measured_fits"])
                if finite:
                    self.assertAlmostEqual(mean(finite), stat["mean"], places=13)
                expected = stat["mean"] if len(finite) == 5 else None
            self.assertEqual(point["value"], expected)
            self.assertEqual(point["support_status"], "unavailable" if expected is None else "measured")

    def test_sample_groups_reject_missing_duplicate_and_mixed_fit(self):
        for mutation in ("missing", "duplicate", "mixed_fit", "mixed_config"):
            docs = dict(self.docs)
            docs["tabsyn"] = copy.deepcopy(docs["tabsyn"])
            cells = docs["tabsyn"]["cells"]
            index = next(i for i,c in enumerate(cells) if c["row_multiplier"] == 4)
            if mutation == "missing": cells.pop(index)
            elif mutation == "duplicate": cells.append(copy.deepcopy(cells[index]))
            elif mutation == "mixed_fit": cells[index]["fit_seed"] = 23
            else: cells[index]["config_sha256"] = "0"*64
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "sample group"):
                inventory.build(docs)

    def test_missing_utility_is_null_without_partial_seed_average(self):
        docs = dict(self.docs)
        docs["tabsyn"] = copy.deepcopy(docs["tabsyn"])
        cell = next(c for c in docs["tabsyn"]["cells"] if c["row_multiplier"] == 4)
        cell["metrics"]["utility"]["auditors"]["catboost"]["retention"] = None
        point = next(p for p in inventory.build(docs)["figure_points"]
                     if p["cohort"] == "TabSyn scaled (8)" and p["dataset"] == cell["dataset"]
                     and p["metric"] == "catboost_retention")
        self.assertIsNone(point["value"])
        self.assertEqual(point["support_status"], "unavailable")

    def test_forest_cohorts_are_disjoint_and_complete(self):
        docs = dict(self.docs); docs["forest5"] = docs["forest2"]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            inventory.build(docs)
        docs = dict(self.docs); docs["forest5"] = dict(docs["forest5"],five_fit_default_cohort_complete=False)
        with self.assertRaisesRegex(ValueError, "incomplete published Forest"):
            inventory.build(docs)

    def test_coverage_separates_fit_ledger_and_common_cells(self):
        self.assertEqual((self.rows["TabSyn"]["complete_retained_lineages"],self.rows["TabSyn"]["complete_retained_sample_cells"]),(8,48))
        self.assertEqual((self.rows["TabDDPM"]["retained_models"],self.rows["TabDDPM"]["complete_retained_sample_cells"],self.rows["TabDDPM"]["logical_sample_cells"]),(21,126,132))
        self.assertEqual((self.rows["Forest-Flow"]["five_fit_lineages"],self.rows["Forest-Flow"]["retained_models"],self.rows["Forest-Flow"]["complete_retained_sample_cells"]),(13,65,390))
        self.assertEqual((self.rows["ARF"]["published_additional_physical_fits"],self.rows["ARF"]["published_additional_metric_cells"]),(796,144))
        for method in ("ARF","GaussianCopula","Chow-Liu"):
            self.assertEqual(self.rows[method]["complete_retained_sample_cells"],1200)
        self.assertEqual(Counter(p["cohort"] for p in self.data["figure_points"])["Forest-Flow five fits (13)"],52)

    def test_pending_gates_and_public_output_have_no_private_tokens(self):
        self.assertFalse(self.data["official_tests_opened"])
        for key in ("mfs_v2","ptf_v1","release_safe_l3","superiority"):
            self.assertIsNone(self.data[key])
        for row in self.data["rows"]:
            self.assertFalse(row["final_five_fit_complete"])
            self.assertFalse(row["production_certified"])
            self.assertIsNone(row["full100_eta_mt"])
            self.assertEqual(row["remaining_campaign_status"],"pending")
        for text in (json.dumps(self.data),inventory.render_csv(self.data),inventory.render_table(self.data)):
            self.assertEqual(banned_hits(text),[])

    def test_committed_outputs_are_exact_reductions(self):
        self.assertEqual(json.loads((self.out/"published-baseline-kpis.json").read_bytes()),self.data)
        self.assertEqual(json.loads((self.out/"baseline-coverage.json").read_bytes()),
                         {k:v for k,v in self.data.items() if k != "figure_points"})
        self.assertEqual((self.out/"published-baseline-kpis.csv").read_text(),inventory.render_csv(self.data))
        self.assertEqual((self.out/"baseline-coverage.tex").read_text(),inventory.render_table(self.data))
        hashes=json.loads((self.out/"published-baseline-figure-hashes.json").read_bytes())["pdf_sha256"]
        figure=inventory.REPO/"docs/whitepaper/figures/published-baseline-cohorts.pdf"
        self.assertEqual(hashes[figure.name],hashlib.sha256(figure.read_bytes()).hexdigest())

    def test_schema_rejects_forecast_and_non_null_claim(self):
        schema=json.loads((inventory.REPO/"docs/whitepaper/scripts/published_baseline_inventory.schema.json").read_bytes())
        validator=jsonschema.Draft202012Validator(schema)
        for filename in ("baseline-coverage.json", "published-baseline-kpis.json"):
            validator.validate(json.loads((self.out/filename).read_bytes()))
        for mutate in ("forecast", "gate", "unavailable"):
            data=copy.deepcopy(self.data)
            if mutate == "forecast": data["rows"][0]["full100_eta_mt"]="2026-10-14T20:00:00-06:00"
            elif mutate == "gate": data["mfs_v2"] = .99
            else:
                data["figure_points"][0]["support_status"]="unavailable"
                data["figure_points"][0]["value"]=1.
            with self.subTest(mutate=mutate), self.assertRaises(jsonschema.ValidationError):
                validator.validate(data)


if __name__ == "__main__":
    unittest.main()
