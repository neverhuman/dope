"""Constant-null conformance/SDK fixture tests; no admission evidence."""
from dataclasses import replace
import unittest

from research.transfer.adapters.base import AdapterAdmission, FeatureBatch
from research.transfer.pilot_constant_null import ConstantNullRecipe
from research.transfer.pilot_learning_contract import SCALES, identity
from research.transfer.pilot_learning_execution import LearningExecution
from research.transfer.pilot_learning_runtime import OperationAdmission, registry_digest
from research.transfer.pilot_learning_reports import execute_drivers, reduce_learning
from research.transfer.pilot_learning_contract import DriverPlan, LEARNERS
from research.transfer.pilot_native_seed import (
    NativeSeed, ORIGINAL_STREAM_HEX, FORMAT, derive_native_seed32,
)
from research.transfer.tests.test_pilot_learning_execution import (
    PIN, FixtureLearner, FixtureNull, FixtureRecipe, row_id,
)

from research.transfer.tests import test_pilot_learning_execution as fixtures


class ConstantNullTests(unittest.TestCase):
    setUp = fixtures.PilotLearningTests.setUp
    run_plan = fixtures.PilotLearningTests.run_plan
    labels = fixtures.PilotLearningTests.labels

    def admission(self, plan=None):
        plan = self.plan if plan is None else plan
        return OperationAdmission("fixture", plan.sha256, registry_digest(self.registry),
                                  PIN, PIN, LEARNERS, self.null.sha256)

    def execution(self):
        return LearningExecution(self.plan, self.registry, self.null, self.admission(), lambda _: None, self.root)

    def test_sdk_null_is_real_context_mean_independent_of_query_features_and_labels(self):
        execution = self.run_plan()
        nulls = [c for c in execution.cases if c["arm"] == "null"]
        self.assertEqual(len(nulls), 20)
        self.assertEqual(len({c["fit_key"] for c in nulls}), 2)
        self.assertEqual(len({c["prediction_key"] for c in nulls}), 2)
        self.assertEqual(sum(not c["reused_prepare"] for c in nulls), 2)
        self.assertEqual(sum(not c["reused_prediction"] for c in nulls), 2)
        for case in nulls:
            expected = 1.5 if case["context"] == "standard" else 1.0
            prediction = execution.predictions[case["prediction_key"]]
            self.assertEqual(prediction["values"], (expected, expected))
            self.assertEqual(case["registration_role"], "null_reference")
            real = next(c for c in execution.cases if c["arm"] == "real" and
                        (c["scale"], c["learner"], c["context"]) ==
                        (case["scale"], case["learner"], case["context"]))
            self.assertEqual(case["train_row_ids"], real["train_row_ids"])
            self.assertEqual(case["consumed_rows"], real["consumed_rows"])
        adapter = execution.prepared[nulls[0]["fit_key"]]
        altered = FeatureBatch(self.plan.query.row_ids, ((-1e9,), (1e9,)), PIN)
        self.assertEqual(adapter.predict(altered, self.plan.predict_seed).values, ((1.5,), (1.5,)))
        before = FixtureLearner.predictions
        left = reduce_learning(execution, self.labels(), {})
        right = reduce_learning(execution, self.labels(100), {})
        self.assertNotEqual(left["comparisons"][0]["losses"]["null"], right["comparisons"][0]["losses"]["null"])
        self.assertEqual(FixtureLearner.predictions, before)

    def test_missing_wrong_or_unbound_null_refuses_before_any_operation(self):
        admission = self.admission()
        for null in (None, self.registry["neural"]):
            with self.assertRaises(ValueError):
                LearningExecution(self.plan, self.registry, null, admission, lambda _: None, self.root)
        for bad in (replace(admission, null_reference_sha256="c" * 64),
                    replace(admission, plan_sha256="c" * 64)):
            with self.assertRaises(ValueError):
                LearningExecution(self.plan, self.registry, self.null, bad, lambda _: None, self.root)
        with self.assertRaises(ValueError):
            replace(self.plan, null_reference_sha256="")
        plan = replace(self.plan, null_reference_sha256="c" * 64)
        with self.assertRaises(ValueError):
            LearningExecution(plan, self.registry, self.null, self.admission(plan), lambda _: None, self.root)
        self.assertFalse((self.root / "physical-operations").exists())
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_null_provenance_training_and_order_cannot_differ_from_real(self):
        pools = tuple((s, a, replace(p, batch=replace(p.batch, targets=(99.0,) * 4))
                       if a == "null" else p) for s, a, p in self.plan.pools)
        with self.assertRaises(ValueError):
            replace(self.plan, pools=pools)
        pools = tuple((s, a, replace(p, source_sha256="c" * 64) if a == "null" else p)
                      for s, a, p in self.plan.pools)
        with self.assertRaises(ValueError):
            replace(self.plan, pools=pools)
        context = self.plan.contexts[0]
        bad = replace(context, selections=tuple((s, a, tuple(reversed(ids)) if a == "null" else ids)
                                                for s, a, ids in context.selections))
        with self.assertRaises(ValueError):
            replace(self.plan, contexts=(bad, self.plan.contexts[1]))
        overlap = replace(self.plan.query, row_ids=self.plan.pools[0][2].batch.features.row_ids[:2])
        with self.assertRaises(ValueError):
            replace(self.plan, query=overlap)

    def test_null_functional_keys_change_with_training_recipe_and_context(self):
        execution = self.execution()
        context = self.plan.contexts[0]
        def key(entry, context=context):
            return identity(execution.functional_inputs(entry, SCALES[0], "null", context)[2])
        original = key(self.null)
        self.assertNotEqual(original, key(self.null, replace(context, seed=18)))
        pools = tuple((s, a, replace(p, batch=replace(p.batch, targets=tuple(v + 10 for v in p.batch.targets)))
                       if a in ("real", "null") else p) for s, a, p in self.plan.pools)
        execution.plan = replace(self.plan, pools=pools)
        self.assertNotEqual(original, key(self.null))
        recipe = FixtureRecipe("d" * 64)
        model = replace(self.null.model, recipe_sha256=recipe.sha256)
        entry = replace(self.null, model=model, recipe=recipe,
                        admission=AdapterAdmission(model.sha256, PIN, PIN, "learner"))
        execution.plan = replace(self.plan, null_reference_sha256=entry.sha256)
        self.assertNotEqual(original, key(entry))
        with self.assertRaises(ValueError):
            execution.functional_inputs(self.registry["neural"], SCALES[0], "null", context)

    def test_failed_null_predictions_are_retained_across_all_twenty_paths(self):
        FixtureNull.fault = "missing"
        execution = self.run_plan()
        nulls = [c for c in execution.cases if c["arm"] == "null"]
        self.assertTrue(all(c["status"] == "failed" for c in nulls))
        self.assertEqual(sum(c["status"] == "completed" for c in execution.cases), 40)
        failed = [r for r in execution.evidence.records if r["event"] == "outcome" and r["status"] == "failed"]
        self.assertEqual(len(failed), len({c["prediction_key"] for c in nulls}))
        self.assertEqual(len(failed), 2)
        self.assertEqual(sum(c["reason"] == "prior_physical_failure_no_retry" for c in nulls), 18)
        self.assertEqual(FixtureLearner.predictions, 32)

    def test_null_drivers_have_real_calls_and_reuse_without_refit(self):
        execution = self.run_plan()
        plan = DriverPlan(((row_id(700), (0,)),), 2, 23, PIN)
        before = FixtureLearner.predictions
        drivers = execute_drivers(execution, self.labels(), plan)
        self.assertEqual(len(drivers["records"]), len({c["fit_key"] for c in execution.cases}))
        self.assertEqual(FixtureLearner.predictions - before, len(drivers["records"]) * 3)
        null_links = [link for link in drivers["links"] if link["arm"] == "null"]
        self.assertEqual(len({link["key"] for link in null_links}), 2)
        for key in {link["key"] for link in null_links}:
            record = drivers["records"][key]
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["result"]["learner_id"], "null_reference")
            self.assertEqual(len(record["calls"]), 3)
            self.assertTrue(all(c["status"] == "completed" for c in record["calls"]))
            self.assertEqual(record["result"]["importance"][row_id(700)], 0.0)
        before = FixtureLearner.predictions
        execute_drivers(execution, self.labels(), plan)
        self.assertEqual(FixtureLearner.predictions, before)
        self.assertEqual(FixtureLearner.prepares, len({c["fit_key"] for c in execution.cases}))

    def test_native_recipe_is_mean_only(self):
        self.assertEqual(ConstantNullRecipe().strategy, "mean")
        for args in ({"strategy": "median"}, {"task": "classification"}, {"precision": "float32"}):
            with self.assertRaises(ValueError):
                ConstantNullRecipe(**args)


class NativeSeedTests(unittest.TestCase):
    def test_owner_golden(self):
        seed, input_sha = derive_native_seed32(int(ORIGINAL_STREAM_HEX, 16))
        self.assertEqual(seed, 2858805439)
        self.assertEqual(input_sha, "bfe865aa65787e0da19efc9273d749dc371e8bc317444e8aaff8050d4ffe93c8")
        provenance = NativeSeed(ORIGINAL_STREAM_HEX)
        self.assertEqual(provenance.seed32, seed)
        self.assertEqual(provenance.provenance["input_sha256"], input_sha)
        self.assertEqual(provenance.original_stream_hex, ORIGINAL_STREAM_HEX)
        self.assertNotEqual(seed, int(ORIGINAL_STREAM_HEX, 16) % (1 << 32))

    def test_seed_refuses_truncation_unknown_policy_and_boolean(self):
        for original in (-1, 1 << 64, True, "0x000d0beef2026105"):
            with self.assertRaises(ValueError):
                derive_native_seed32(original)
        with self.assertRaises(ValueError):
            derive_native_seed32(1, FORMAT + "-unknown")
        for changes in ({"format": "V3"}, {"original_stream_hex": "0x1"},
                        {"owner_grant_sha256": "a" * 64}):
            with self.assertRaises(ValueError):
                replace(NativeSeed(ORIGINAL_STREAM_HEX), **changes)


if __name__ == "__main__":
    unittest.main()
