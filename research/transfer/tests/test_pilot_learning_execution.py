"""Fixture-only source tests; never claim native model/runtime qualification."""
from dataclasses import dataclass, replace
import hashlib
from importlib.metadata import PathDistribution
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from research.transfer.adapters.base import (
    AdapterAdmission, FeatureBatch, LearnerAdapter, ModelIdentity, PredictionResult,
    ResourceLimits, ResourceUsage, TrainingBatch,
)
from research.transfer.pilot_learning_contract import (
    ARMS, CONTEXTS, LEARNERS, SCALES, Context, DriverPlan, LearningPlan, Pool, QueryLabels, identity,
)
from research.transfer.pilot_learning_execution import LearningExecution
from research.transfer.pilot_learning_reports import execute_drivers, reduce_learning
from research.transfer.pilot_learning_runtime import NAMES, OperationAdmission, Registration, registry_digest, verify_distribution_record

from research.transfer.pilot_native_seed import NativeSeed, ORIGINAL_STREAM_HEX

PIN = "a" * 64


def row_id(value):
    return identity({"fixture_row": value})


@dataclass(frozen=True)
class FixtureRecipe:
    sha256: str = "b" * 64


class FixtureLearner(LearnerAdapter):
    prepares = 0
    predictions = 0
    fault = None

    def _prepare_backend(self, training, seed, mode):
        type(self).prepares += 1
        self.mean = sum(training.targets) / len(training.targets)
        return ResourceUsage(len(training.targets), 64, 0)

    def _predict_backend(self, query, seed):
        type(self).predictions += 1
        if type(query) is not FeatureBatch or hasattr(query, "targets"):
            raise AssertionError("query_labels_reached_learner")
        if self.fault == "missing":
            return None
        ids = query.row_ids[:-1] if self.fault == "partial" else query.row_ids
        values = tuple((float(row[0]) + self.mean,) for row in query.rows[:len(ids)])
        if self.fault == "nonfinite":
            values = tuple((float("nan"),) for _ in ids)
        result = PredictionResult(ids, values, "point", len(self._training.targets), 64, 0,
                                  embedding_unavailable="unsupported")
        if self.fault == "bool_count":
            object.__setattr__(result, "consumed_rows", True)  # adversarial fixture response
        return result


class FixtureNull(LearnerAdapter):
    """Actual SDK constant, fixture resource claims; no runtime qualification."""
    fault = None
    def _prepare_backend(self, training, seed, mode):
        from sklearn.dummy import DummyRegressor
        FixtureLearner.prepares += 1
        self.model = DummyRegressor(strategy="mean")
        self.model.fit(training.features.rows, training.targets)
        return ResourceUsage(len(training.targets), 64, 0)

    def _predict_backend(self, query, seed):
        FixtureLearner.predictions += 1
        if type(query) is not FeatureBatch or hasattr(query, "targets"):
            raise AssertionError("query_labels_reached_null")
        fault = self.fault or FixtureLearner.fault
        if fault == "missing":
            return None
        ids = query.row_ids[:-1] if fault == "partial" else query.row_ids
        values = tuple((float(v),) for v in self.model.predict(query.rows)[:len(ids)])
        if fault == "nonfinite":
            values = tuple((float("nan"),) for _ in ids)
        result = PredictionResult(ids, values, "point", len(self._training.targets), 64, 0,
                                  embedding_unavailable="unsupported")
        if fault == "bool_count":
            object.__setattr__(result, "consumed_rows", True)
        return result


def make_null():
    template = make_registry()["neural"]
    model = replace(template.model, name="regression_mean_null")
    return replace(template, role="null_reference", model=model,
                   admission=AdapterAdmission(model.sha256, PIN, PIN, "learner"), fixture_factory=FixtureNull)


def make_plan():
    def pool(role, first, n):
        batch = TrainingBatch(FeatureBatch(tuple(row_id(first + i) for i in range(n)),
                                          tuple((float(i),) for i in range(n)), PIN),
                              tuple(float(i + 1) for i in range(n)),
                              "synthetic_train" if role == "synthetic_draw" else "development_train")
        return Pool(batch, role, PIN, PIN, n, n, 13)
    real, null = pool("fit_real", 0, 4), pool("fit_null", 0, 4)
    pools = tuple((scale, arm, (real if arm == "real" else null if arm == "null" else
                              pool("synthetic_draw", 20 if scale == "primary_n" else 30,
                                   2 if scale == "primary_n" else 8)))
                  for scale in SCALES for arm in ARMS)
    contexts = tuple(Context(name, n, 17, PIN,
                             tuple((scale, arm, pool.batch.features.row_ids[:n])
                                   for scale, arm, pool in pools))
                     for name, n in zip(CONTEXTS, (2, 1)))
    seed = NativeSeed(ORIGINAL_STREAM_HEX)
    return LearningPlan(row_id(900), "dope", "chow_liu", PIN, PIN, PIN, PIN, PIN, PIN, seed.sha256,
                        2, seed.seed32, seed.seed32, pools, contexts,
                        FeatureBatch((row_id(100), row_id(101)), ((0.5,), (1.5,)), PIN),
                        "retain_for_separate_drivers", make_null().sha256, seed)


def make_registry():
    registry = {}
    for role in LEARNERS:
        name, mode = NAMES[role]
        weights = PIN if mode != "scratch" else None
        licenses = (("code", "fixture-only", PIN),) + ((("weights", "fixture-only", PIN),) if weights else ())
        model = ModelIdentity(name, mode, "c" * 40, PIN, weights, (("fixture", "1.0", PIN),), licenses,
                              PIN, FixtureRecipe().sha256, PIN, PIN, "float64", 4, "all",
                              "transductive" if role == "mitra" else "batch", 2, True,
                              "deterministic", ResourceLimits(100.0, 1 << 30, 0))
        admission = AdapterAdmission(model.sha256, PIN, PIN, "learner")
        registry[role] = Registration(role, model, admission, FixtureRecipe(), None, PIN, PIN,
                                      (("fixture", PIN),), FixtureLearner)
    return registry


class PilotLearningTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.registry, self.null = make_plan(), make_registry(), make_null()
        FixtureLearner.prepares = FixtureLearner.predictions = 0
        FixtureLearner.fault = FixtureNull.fault = None
        base = Path(__file__).resolve().parents[3] / "target/pilot-learning-fixtures"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_plan(self, plan=None, registry=None):
        plan = self.plan if plan is None else plan
        registry = self.registry if registry is None else registry
        admission = OperationAdmission("fixture", plan.sha256, registry_digest(registry), PIN, PIN, LEARNERS, self.null.sha256)
        def verify(value):
            self.assertEqual(value.kind, "fixture")
        return LearningExecution(plan, registry, self.null, admission, verify, self.root).execute()

    def labels(self, delta=0):
        return QueryLabels(self.plan.query.row_ids, (1.0 + delta, 2.0 + delta), PIN)

    def test_physical_receipt_preserves_native_rss_units(self):
        from research.transfer import pilot_learning_execution as module
        def usage(kind):
            return SimpleNamespace(ru_maxrss=32768, ru_utime=0.0, ru_stime=0.0)
        evidence = module.PhysicalEvidence(self.root)
        for platform, expected in (("darwin", 32768), ("linux", 33554432)):
            with mock.patch.object(module.sys, "platform", platform), \
                 mock.patch.object(module.resource, "getrusage", side_effect=usage):
                self.assertEqual(evidence.call(platform, "fixture", {}, lambda: (7, {})), 7)
            outcome = evidence.records[-1]
            self.assertEqual(outcome["status"], "completed")
            self.assertEqual(outcome["process_lifetime_peak_rss_bytes"], expected)

    def test_exact_counts_reuse_finite_predictions_and_label_isolation(self):
        execution = self.run_plan()
        self.assertEqual(len(execution.cases), 60)
        self.assertTrue(all(c["status"] == "completed" for c in execution.cases))
        physical_fits = {c["fit_key"] for c in execution.cases}
        physical_predictions = {c["prediction_key"] for c in execution.cases}
        self.assertEqual(len(physical_fits), 32)
        self.assertEqual(FixtureLearner.prepares, len(physical_fits))
        self.assertEqual(FixtureLearner.predictions, len(physical_predictions))
        self.assertEqual(sum(c["reused_prediction"] for c in execution.cases), 28)
        first = execution.cases[0]
        self.assertEqual(execution.predictions[first["prediction_key"]]["values"], (2.0, 3.0))
        paired = reduce_learning(execution, self.labels(), {})
        changed = reduce_learning(execution, self.labels(100), {})
        self.assertEqual(FixtureLearner.predictions, 32)
        self.assertNotEqual(paired["comparisons"][0]["losses"], changed["comparisons"][0]["losses"])
        self.assertEqual(len(paired["comparisons"]), 20)
        for row in paired["comparisons"]:
            self.assertEqual(row["status"], "incomplete")  # missing gain confidence bound keeps reduction incomplete
            self.assertTrue(all(len(loss) == 2 and all(math.isfinite(v) for v in loss)
                                for loss in row["losses"].values()))
        for case in execution.cases:
            if case["arm"] == "real" and case["scale"] == "sensitivity_4n":
                self.assertEqual(case["available_pool_rows"], 4)
                self.assertEqual(case["pool_produced_rows"], 4)  # never advertise eight real rows

    def test_uninformative_result_is_retained_without_rerun(self):
        execution = self.run_plan()
        labels = self.labels()
        inference = {(s, l, c): {"real_gain_lcb": -1.0, "evidence_sha256": PIN,
                                  "labels_sha256": labels.sha256}
                     for s in SCALES for l in LEARNERS for c in CONTEXTS}
        before = FixtureLearner.predictions
        result = reduce_learning(execution, labels, inference)
        self.assertEqual(len(result["comparisons"]), 20)
        self.assertTrue(all(row["status"] == "uninformative" for row in result["comparisons"]))
        self.assertEqual(FixtureLearner.predictions, before)
        with self.assertRaises(ValueError):
            replace(self.plan.pools[0][2], produced_rows=True)

    def test_seed_context_and_arm_invalidate_functional_reuse(self):
        admission = OperationAdmission("fixture", self.plan.sha256, registry_digest(self.registry), PIN, PIN, LEARNERS, self.null.sha256)
        execution = LearningExecution(self.plan, self.registry, self.null, admission, lambda _: None, self.root)
        entry, context = self.registry["neural"], self.plan.contexts[0]
        first = execution.functional_inputs(entry, SCALES[0], "real", context)[2]
        same = execution.functional_inputs(entry, SCALES[1], "real", context)[2]
        changed = execution.functional_inputs(entry, SCALES[0], "real", replace(context, seed=18))[2]
        null = execution.functional_inputs(self.null, SCALES[0], "null", context)[2]
        self.assertEqual(identity(first), identity(same))
        self.assertNotEqual(identity(first), identity(changed))
        self.assertNotEqual(identity(first), identity(null))
        seed = NativeSeed("0x000d0beef2026106")
        execution.plan = replace(self.plan, native_seed=seed, seed_derivation_sha256=seed.sha256,
                                 fit_seed=seed.seed32, predict_seed=seed.seed32)
        self.assertNotEqual(identity(first), identity(execution.functional_inputs(entry, SCALES[0], "real", context)[2]))

    def test_missing_learner_and_matched_reference_refuse(self):
        del self.registry["mitra"]
        with self.assertRaises(ValueError):
            self.run_plan()
        self.assertEqual(FixtureLearner.prepares, 0)
        bad = replace(self.plan.contexts[0], selections=tuple(
            (s, a, (row_id(999), ids[1])) for s, a, ids in self.plan.contexts[0].selections))
        with self.assertRaises(ValueError):
            replace(self.plan, contexts=(bad, self.plan.contexts[1]))

    def test_failures_remain_in_planned_denominator(self):
        FixtureLearner.fault = "missing"
        execution = self.run_plan()
        self.assertEqual(len(execution.cases), 60)
        self.assertTrue(all(c["status"] == "failed" for c in execution.cases))
        self.assertEqual(execution.private_result()["planned_comparison_count"], 20)
        failures = [r for r in execution.evidence.records if r["event"] == "outcome" and r["status"] == "failed"]
        self.assertEqual(len(failures), 32)
        self.assertTrue(all(r["wall_ns"] >= 0 for r in failures))

    def test_invalid_partial_nonfinite_and_boolean_counts_refuse(self):
        from research.transfer.pilot_learning_execution import validate_prediction
        query = self.plan.query
        for factory, registration in ((FixtureLearner, self.registry["neural"]), (FixtureNull, self.null)):
            for fault in ("partial", "nonfinite", "bool_count"):
                FixtureLearner.fault = fault
                learner = factory(registration.model, registration.admission)
                _, train = self.plan.selected(SCALES[0], "real", self.plan.contexts[1])
                learner.prepare(train, self.plan.fit_seed)
                with self.assertRaises(ValueError):
                    validate_prediction(learner.predict(query, self.plan.predict_seed), query, 1)

    def test_streaming_releases_models_without_extra_fits(self):
        from research.transfer.pilot_learning_reports import execute_streamed
        plan = replace(self.plan, model_retention="stream_drivers_then_release")
        admission = OperationAdmission("fixture", plan.sha256, registry_digest(self.registry), PIN, PIN, LEARNERS, self.null.sha256)
        execution = LearningExecution(plan, self.registry, self.null, admission, lambda _: None, self.root)
        _, drivers = execute_streamed(execution, self.labels(), DriverPlan(((row_id(700), (0,)),), 2, 23, PIN))
        self.assertEqual(FixtureLearner.prepares, 32)
        self.assertEqual(FixtureLearner.predictions, 128)
        self.assertTrue(all(adapter is None for adapter in execution.prepared.values()))
        self.assertEqual(len(drivers["links"]), 60)

    def test_drivers_really_predict_and_keep_perturbations(self):
        execution = self.run_plan()
        before = FixtureLearner.predictions
        driver_plan = DriverPlan(((row_id(700), (0,)),), 2, 23, PIN)
        drivers = execute_drivers(execution, self.labels(), driver_plan)
        self.assertEqual(FixtureLearner.predictions - before, len(drivers["records"]) * 3)
        self.assertEqual(len(drivers["links"]), 60)
        self.assertTrue(all(link["status"] == "completed" for link in drivers["links"]))
        self.assertEqual(sum(link["reused"] for link in drivers["links"]), 28)
        for record in drivers["records"].values():
            self.assertEqual(len(record["calls"]), 3)
            self.assertTrue(all(len(call["prediction"]["values"]) == 2 for call in record["calls"]))
            self.assertIn("perturbed_rows", record["calls"][1])
            self.assertEqual(record["applicability"]["interaction"], "not_measured")


class DistributionRecordTests(unittest.TestCase):
    def test_real_metadata_parser_selects_parent_record_with_vendored_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = root / "native_sdk-1.0.dist-info"
            info.mkdir()
            (info / "METADATA").write_text("Name: native-sdk\nVersion: 1.0\n")
            record = info / "RECORD"
            vendor = root / "native_sdk/vendor/another-2.0.dist-info/RECORD"
            vendor.parent.mkdir(parents=True)
            vendor.write_text("separate distribution bytes\n")
            parent_member = "native_sdk-1.0.dist-info/RECORD,,\n"
            vendor_member = "native_sdk/vendor/another-2.0.dist-info/RECORD,,\n"
            record.write_text(vendor_member + parent_member)
            dist = PathDistribution(info)
            expected = hashlib.sha256(record.read_bytes()).hexdigest()
            self.assertEqual(len([p for p in dist.files if str(p).endswith(".dist-info/RECORD")]), 2)
            self.assertIsNone(verify_distribution_record(dist, "1.0", expected))
            with self.assertRaisesRegex(ValueError, "installed_runtime_changed"):
                verify_distribution_record(dist, "1.0", "0" * 64)
            with self.assertRaisesRegex(ValueError, "installed_runtime_unavailable"):
                verify_distribution_record(dist, "2.0", expected)
            for malformed in (vendor_member, parent_member + parent_member):
                record.write_text(malformed)
                with self.assertRaisesRegex(ValueError, "installed_runtime_unavailable"):
                    verify_distribution_record(PathDistribution(info), "1.0",
                                               hashlib.sha256(record.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
