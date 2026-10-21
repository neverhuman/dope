"""Exact-query reuse retains metrics and durable cost; fixture work, no SDK claims."""
from dataclasses import replace
import unittest
from research.transfer.drivers import permutation_importance
from research.transfer.adapters.base import FeatureBatch
from research.transfer.pilot_learning_contract import DriverPlan, LEARNERS, identity
from research.transfer.pilot_learning_reports import execute_drivers, execute_streamed, losses
from research.transfer.pilot_physical_operations import reserve_learning_operations
from research.transfer.tests import test_pilot_physical_operations as physical_fixture
CELL = physical_fixture.CELL
from research.transfer.tests.test_pilot_learning_execution import FixtureLearner, PIN, make_plan, QueryLabels

class FunctionalDriverAliasTests(unittest.TestCase):
    def setUp(self):
        self.helper = physical_fixture.PhysicalOperationTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.driver = DriverPlan(((PIN, (0,)),), 2, 17, PIN)

    def test_cached_driver_metrics_equal_uncached_native_fixture_reference(self):
        execution = self.helper.execution(make_plan())
        execution.execute()
        labels = QueryLabels(execution.plan.query.row_ids, (1.0, 2.0), PIN)
        before = FixtureLearner.predictions
        drivers = execute_drivers(execution, labels, self.driver)
        used = FixtureLearner.predictions - before
        logical = sum(len(r["calls"]) for r in drivers["records"].values())
        self.assertLess(used, logical)
        visited = set()
        for case in execution.cases:
            key = identity({"fit": case["fit_key"], "query": execution.plan.query.sha256,
                            "labels": labels.sha256, "driver_plan": {
                                "groups": self.driver.groups, "repeats": self.driver.repeats,
                                "seed": self.driver.seed, "protocol_sha256": self.driver.protocol_sha256}})
            if key in visited:
                continue
            visited.add(key)
            record = drivers["records"][key]
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["calls"][0]["key"], case["prediction_key"])
            self.assertTrue(record["calls"][0]["reused_prediction"])
            adapter = execution.prepared[case["fit_key"]]
            registration = execution.registration_for(case["learner"], case["arm"])
            count = [0]
            def fresh_loss(columns):
                count[0] += 1
                query = FeatureBatch(execution.plan.query.row_ids,
                    tuple(tuple(r) for r in zip(*columns)), execution.plan.query.schema_sha256)
                payload = execution.predict(adapter, registration, query, case["consumed_rows"],
                    identity(["uncached-reference", key, count[0]]), parent=case["fit_key"])
                values = losses(payload, labels)
                return sum(values) / len(values)
            reference = permutation_importance(fresh_loss, tuple(zip(*execution.plan.query.rows)),
                dict(self.driver.groups), learner_id=registration.role,
                query_population=execution.plan.query.sha256, seed=self.driver.native_seed,
                repeats=self.driver.repeats)
            self.assertEqual(record["result"], reference)

    def test_changed_driver_plan_refuses_before_native_work_or_journal_mutation(self):
        execution = self.helper.execution(replace(make_plan(), model_retention="stream_drivers_then_release"))
        labels = QueryLabels(execution.plan.query.row_ids, (1.0, 2.0), PIN)
        wall = {r: {"prepare": 10000, "predict": 10000} for r in (*LEARNERS, "null_reference")}
        with self.helper.journal() as journal:
            self.helper.begin_learners(journal, 2000000)
            reserve_learning_operations(execution, journal, CELL, wall, driver_plan=self.driver, labels=labels)
            before = (self.helper.root / "development-stages.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "driver_plan_not_reserved"):
                execute_drivers(execution, labels, replace(self.driver, seed=18))
            self.assertEqual((FixtureLearner.prepares, FixtureLearner.predictions), (0, 0))
            self.assertEqual((self.helper.root / "development-stages.jsonl").read_bytes(), before)

    def test_corrupted_cached_seed_or_query_never_produces_a_completed_driver(self):
        for field, wrong in (("predict_seed", -1), ("query_sha256", "f" * 64)):
            root = self.helper.root / field
            root.mkdir(mode=0o700)
            execution = self.helper.execution(make_plan(), root=root)
            execution.execute()
            case = execution.cases[0]
            execution.cases[:] = [case]
            execution.predictions[case["prediction_key"]][field] = wrong
            before = FixtureLearner.predictions
            labels = QueryLabels(execution.plan.query.row_ids, (1.0, 2.0), PIN)
            actual = execute_drivers(execution, labels, self.driver)
            self.assertTrue(all(r["status"] == "failed" for r in actual["records"].values()))
            self.assertEqual(FixtureLearner.predictions, before)
