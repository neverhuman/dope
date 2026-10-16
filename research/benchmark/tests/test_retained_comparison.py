"""Focused publisher regressions; all input reads are synthetic JSON metadata."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from research.benchmark import publish_retained_comparison as publisher


class ReceiptFixture:
    def __init__(self, utility_only=False):
        self.buffers, self.utility_only = {}, utility_only
        self.inputs = {k + "_ref": dict(path="/fixture/" + k, bytes=7, sha256=str(i) * 64)
                       for i, k in enumerate(("train", "validation", "synthetic", "projection"), 1)}
        self.identity = dict(dataset="opaque_lineage", method="ARF", fit_seed=11, sample_seed=101,
                             row_multiplier=1, config_sha256="a" * 64,
                             selection_binding="author_default", charged_artifact_bytes=99)
        ok = lambda value: dict(status="ok", value=value)
        self.metrics = dict(fidelity=dict(marginals=[dict(value=.1)],
            pairwise_pearson_difference=ok(dict(mean=.2)), categorical_contingency_tv=ok(dict(mean=.3))),
            alpha_beta=ok(dict(alpha_precision=.8, beta_recall=.7)),
            detection=dict(catboost=ok(.6), logistic=ok(.55)),
            privacy=dict(dcr_fit=dict(median=.4), dcr_validation=dict(median=.5),
                         nndr_fit=ok(dict(median=.6)), distance_mia=ok(.52), domias_kde=ok(.51)),
            utility=dict(null_loss=1., auditors={k: dict(trtr_loss=.1, tstr_loss=.2, retention=8 / 9)
                                               for k in ("catboost", "linear", "mlp")}),
            official_tests_opened=False, **dict.fromkeys(publisher.CLAIMS))
        canonical = dict(final=False, fit_seed=11, kind_rule_sha256="b" * 64,
                         metric_source_sha256="c" * 64, sample_sha256=self.inputs["synthetic_ref"]["sha256"],
                         split="official_training_derived_validation", track="common-numeric", worker_key="d" * 64)
        self.job_sha = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.receipt = dict(self.identity, job_identity=canonical, job_sha256=self.job_sha,
                            input_refs=copy.deepcopy(self.inputs), status="ok", metrics=self.metrics,
                            dependencies={"numpy": "1.26.4", "scipy": "1.16.3", "scikit-learn": "1.7.2", "catboost": "1.2.10"},
                            source_refs={},
                            official_tests_opened=False, **dict.fromkeys(publisher.CLAIMS))
        self.job = dict(job_sha256=self.job_sha, job_identity=canonical, evaluator_input=copy.deepcopy(self.inputs))
        self.alias = dict(self.identity, synthetic_ref=self.inputs["synthetic_ref"])
        self.row = dict(self.identity, official_tests_opened=False, **dict.fromkeys(publisher.CLAIMS))
        if utility_only:
            expanded = {k: v for k, v in self.metrics.items() if k != "utility"}
            refs = dict(expanded_metric_ref=self.put("expanded", expanded),
                        expanded_metric_origin_ref=self.put("origin", dict(job_sha256=self.job_sha,
                                                               status="ok", diagnostics=expanded)))
            self.job["evaluator_input"].update(refs)
            self.receipt.update(refs, utility=self.metrics["utility"], utility_auditor_seed=1729)
            self.receipt.pop("metrics")

    def put(self, name, document):
        data = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        path = "/fixture/" + name + ".json"
        self.buffers[path] = data
        return dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def read(self, path, ref):
        data = self.buffers[str(path)]
        if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("frozen_receipt_changed")
        return json.loads(data)

    def run(self):
        receipt_ref = self.put("receipt", self.receipt)
        aliases = self.put("aliases", dict(aliases=[self.alias]))
        self.row.update(metric_receipt_ref=receipt_ref, metric_job_sha256=self.job_sha,
                        alias_origin_ref=self.row.get("alias_origin_ref", aliases), alias_json_pointer="/aliases/0",
                        measurement=self.receipt.get("metrics", self.metrics))
        for key, ref in self.inputs.items():
            self.row.setdefault(key.removesuffix("_ref") + "_sha256", ref["sha256"])
        manifest = self.put("inputs", dict(jobs=[] if self.utility_only else [self.job]))
        lock = self.put("lock", dict(actual_complete=True, official_tests_opened=False,
                                    count=0 if self.utility_only else 1,
                                    receipts=[] if self.utility_only else [receipt_ref]))
        joined = dict(alias_map_ref=aliases, input_manifest_ref=manifest, physical_receipt_lock_ref=lock,
                      logical_cells=1, rows=[self.row])
        if self.utility_only:
            joined["utility_receipt_lock_ref"] = self.put("utility-lock", dict(actual_complete=True,
                utility_auditor_seed=1729, receipts=[receipt_ref], manifest_ref=self.put("utility-inputs", dict(jobs=[self.job]))))
        join = self.put("join", joined)
        with mock.patch.object(publisher, "verified", side_effect=self.read), \
             mock.patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected_file_read")):
            return publisher.read_join(join, {})


class ComparisonIntegrityTests(unittest.TestCase):
    def denied_before_flatten(self, fixture, message):
        with mock.patch.object(publisher, "flatten") as flatten:
            with self.assertRaisesRegex(ValueError, message):
                fixture.run()
            flatten.assert_not_called()

    def test_positive_full_and_split_utility_metadata(self):
        for utility, deduplicated_alias in ((False, False), (True, False), (False, True)):
            with self.subTest(utility=utility, deduplicated_alias=deduplicated_alias):
                f = ReceiptFixture(utility)
                if deduplicated_alias:
                    for record in (f.alias, f.row):
                        record.update(config_sha256="e" * 64, charged_artifact_bytes=123)
                rows = f.run()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["marginal_error_mean"], .1)
                self.assertEqual(rows[0]["catboost_retention"], 8 / 9)
                self.assertEqual(rows[0]["config_sha256"], f.alias["config_sha256"])
                self.assertEqual(rows[0]["charged_artifact_bytes"], f.alias["charged_artifact_bytes"])

    def test_borrowed_expanded_origin_pair_rejected(self):
        f = ReceiptFixture(True)
        expanded = {k: v for k, v in f.metrics.items() if k != "utility"}
        f.receipt.update(expanded_metric_ref=f.put("borrowed-expanded", expanded),
                         expanded_metric_origin_ref=f.put("borrowed-origin", dict(job_sha256="f" * 64,
                                                                  status="ok", diagnostics=expanded)))
        self.denied_before_flatten(f, "retained_diagnostic_not_in_frozen_job")

    def test_raw_receipt_and_measurement_claims_rejected(self):
        for location in ("receipt", "metrics"):
            for key in ("official_tests_opened", *publisher.CLAIMS):
                with self.subTest(location=location, key=key):
                    f = ReceiptFixture()
                    record = f.receipt if location == "receipt" else f.metrics
                    record[key] = True if key == "official_tests_opened" else .9
                    self.denied_before_flatten(f, "raw_metric_validation_claim_changed")

    def test_altered_alias_configuration_and_sample_binding_rejected(self):
        cases = (("alias_origin_ref", {}, "alias_manifest_binding_mismatch"),
                 ("config_sha256", "e" * 64, "logical_configuration_binding_mismatch"),
                 ("synthetic_sha256", "e" * 64, "aliased_numeric_input_mismatch"))
        for key, value, error in cases:
            with self.subTest(key=key):
                f = ReceiptFixture()
                f.row[key] = value
                self.denied_before_flatten(f, error)

    def test_three_seed_mean_and_missing_seed_denial(self):
        f = ReceiptFixture()
        rows = [dict(f.identity, sample_seed=seed, scalar=float(i))
                for i, seed in zip((1, 3, 5), (101, 211, 307))]
        full = publisher.summarize(rows, ["opaque_lineage"], "test_scope")
        self.assertEqual(full[0]["scalar"], dict(median=3., measured_datasets=1, unavailable_datasets=0))
        with self.assertRaisesRegex(ValueError, "incomplete_matched_three_seed_group"):
            publisher.summarize(rows[:-1], ["opaque_lineage"], "test_scope")


class CommittedComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1] / "results/retained-classical-validation"
        cls.panel = json.loads((cls.root / "panel.json").read_bytes())
        ref = cls.panel["physical_metrics_file"]
        data = (cls.root / ref["path"]).read_bytes()
        if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise AssertionError("committed_physical_metrics_changed")
        cls.records = [json.loads(line) for line in data.splitlines()]
        cls.measurements = {r["receipt_sha256"]: {k: v for k, v in r.items() if k != "receipt_sha256"}
                            for r in cls.records}

    def test_every_logical_cell_has_real_metrics_and_complete_seeds(self):
        self.assertEqual((len(self.panel["rows"]), len(self.records), len(self.measurements)), (3648, 3186, 3186))
        groups = {}
        for row in self.panel["rows"]:
            record = self.measurements[row["metric_receipt_ref"]["sha256"]]
            self.assertEqual(record["receipt_ref"], row["metric_receipt_ref"])
            self.assertEqual(record["job_sha256"], row["metric_job_sha256"])
            flat = publisher.flatten({**row, "metrics": record["metrics"]})
            self.assertTrue(all(row[key] == value for key, value in flat.items()))
            key = (row["method"], row["selection_binding"], row["row_multiplier"], row["dataset"])
            groups.setdefault(key, []).append(row["sample_seed"])
        self.assertTrue(all(sorted(seeds) == [101, 211, 307] for seeds in groups.values()))
        self.assertEqual(len(groups), 1216)
        for dataset in {row["dataset"] for row in self.panel["rows"]}:
            cohort = [row for row in self.panel["rows"] if row["dataset"] == dataset]
            for name in ("train", "validation", "projection"):
                self.assertEqual(len({row["input_hashes"][name] for row in cohort}), 1)
            for name in ("null_loss", "catboost_trtr_loss", "linear_trtr_loss", "mlp_trtr_loss"):
                self.assertEqual(len({row[name] for row in cohort if row[name] is not None}), 1)

    def test_all_schemas_and_empirical_claims(self):
        import jsonschema
        for filename in ("panel.schema.json", "physical-metrics.schema.json"):
            schema = json.loads((self.root / filename).read_bytes())
            jsonschema.Draft202012Validator.check_schema(schema)
            validator = jsonschema.Draft202012Validator(schema)
            if filename == "panel.schema.json":
                validator.validate(self.panel)
            else:
                for record in self.records:
                    validator.validate(record)
        for key in publisher.CLAIMS:
            self.assertIsNone(self.panel[key])

    def test_regenerate_all_kpis_aggregates_and_figures_from_committed_metrics(self):
        rows = self.panel["rows"]
        summary = []
        for method in ("ARF", "Chow-Liu", "GaussianCopula"):
            cohort = [row for row in rows if row["method"] == method]
            datasets = sorted({row["dataset"] for row in cohort})
            self.assertEqual((len(cohort), len(datasets)), (1200, 100))
            summary += publisher.summarize(cohort, datasets, "common_100_lineages")
        common = sorted({row["dataset"] for row in rows if row["method"] == "TabSyn"})
        self.assertEqual(len(common), 8)
        summary += publisher.summarize([row for row in rows if row["dataset"] in common], common,
                                       "matched_tabsyn_eight_lineages")
        self.assertEqual(summary, self.panel["summary"])
        panel = {**self.panel, "measurements": self.measurements}
        for filename, data in publisher.render(panel).items():
            self.assertEqual(data, (self.root / filename).read_bytes(), filename)


if __name__ == "__main__":
    unittest.main()
