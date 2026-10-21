"""Funded finite lifetimes use synthetic clocks, never native model evidence."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from research.transfer.adapters.base import AdapterAdmission, AdapterError, ResourceLimits
from research.transfer.campaign_runner import CampaignRunSpec, CampaignRunnerError, StageJournal
from research.transfer.pilot_physical_operations import (
    FORMAT, LIFETIME_FORMAT, LIFETIME_POLICY, PhysicalOperationGraph, reserve_learning_operations,
)
from research.transfer.tests.test_adapters import AffineLearner, features, identity, registry, training
from research.transfer.tests.test_pilot_physical_operations import CELL, KEY, PREDICT, PIN, fixture_spec, tiny_graph

OTHER = "8" * 64


class FundedLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        doc = fixture_spec().document()
        doc["budget"]["wall_ms"] = 50000000
        self.spec = CampaignRunSpec.from_document(doc)
        self.journal = StageJournal(Path(self.tmp.name), self.spec, create=True).__enter__()
        self.addCleanup(self.journal.__exit__, None, None, None)
        for stage in ("prepare", "compile", "reload", "sample"):
            self.journal.begin(CELL, stage, 10)
            self.journal.finish(CELL, stage, "completed", observed_wall_ms=1,
                                peak_ram_bytes=512, output_sha256=PIN)
        self.journal.begin(CELL, "learners", 40000000)
        from research.transfer.tests.test_pilot_learning_execution import FixtureLearner, FixtureNull
        FixtureLearner.fault = FixtureNull.fault = None
        self.model = identity(resources=ResourceLimits(600.0, 1024, 0))
        self.adapter = AffineLearner(self.model, AdapterAdmission.from_registry(registry(self.model), self.model, "learner"))

    def graph(self, version2=True):
        doc = tiny_graph(600000).document()
        doc["campaign_plan_sha256"] = self.spec.sha256
        doc["nodes"].append({"key": OTHER, "kind": "predict", "parents": [KEY, PREDICT],
                             "wall_ms": 600000, "reserved_child_fits": 0,
                             "reserved_optimizer_updates": 0})
        for alias in doc["aliases"]:
            alias["operations"].append(OTHER)
        if version2:
            doc.update(format=LIFETIME_FORMAT, adapter_lifetime_policy=LIFETIME_POLICY)
        return PhysicalOperationGraph.from_document(doc)

    def reserve(self, version2=True):
        return self.journal.reserve_operations(CELL, "learners", self.graph(version2))

    def call(self, guard, key, seconds, fn):
        guard.begin(key, "prepare" if key == KEY else "predict")
        status = "failed"
        try:
            with mock.patch("research.transfer.adapters.base.time.monotonic", side_effect=[0.0, seconds]):
                value = fn()
            status = "completed"
            return value
        finally:
            guard.finish(key, status, int(seconds * 1000000000), {})

    def prepare(self, guard, seconds=400.0, *, bind=True):
        def run():
            if bind:
                self.adapter.bind_physical_budget(guard, KEY)
            return self.adapter.prepare(training(), 9)
        return self.call(guard, KEY, seconds, run)

    def predict(self, guard, key, seconds=400.0):
        return self.call(guard, key, seconds, lambda: self.adapter.predict(features((4.0, 5.0), "query"), 9))

    def test_original_scalar_reproduces_failure_after_two_legal_calls(self):
        guard = self.reserve(False)
        self.prepare(guard, bind=False)
        with self.assertRaisesRegex(AdapterError, "wall_budget_exceeded"):
            self.predict(guard, PREDICT)
        self.assertEqual(self.adapter._wall_seconds, 800.0)
        self.assertEqual(guard.outcomes[PREDICT]["status"], "failed")

    def test_funded_lifetime_preserves_native600_and_identity_for_three_calls(self):
        original = self.model.sha256
        guard = self.reserve()
        charged = self.journal.charged
        self.prepare(guard)
        self.predict(guard, PREDICT)
        self.predict(guard, OTHER)
        self.assertEqual(self.adapter._wall_seconds, 1200.0)
        receipt = self.adapter.receipt()
        self.assertEqual(receipt["format"], "dope-adapter-execution-v2")
        self.assertEqual(receipt["physical_budget"]["wall_ms"], 1800000)
        self.assertEqual(receipt["physical_budget"]["operation_keys"], [KEY, PREDICT, OTHER])
        self.assertEqual(receipt["native_call_wall_seconds"], 600.0)
        self.assertEqual(self.model.sha256, original)
        self.assertEqual(self.adapter.identity.resources.wall_seconds, 600.0)
        self.journal.settle_operations(CELL, "learners", guard)
        self.assertEqual(self.journal.charged, charged)

    def test_unfunded_or_v1_guard_cannot_extend_lifetime(self):
        with self.assertRaisesRegex(AdapterError, "funded_adapter_lifetime_required"):
            self.adapter.bind_physical_budget(SimpleNamespace(active=KEY), KEY)
        guard = self.reserve(False)
        guard.begin(KEY, "prepare")
        with self.assertRaisesRegex(CampaignRunnerError, "funded_adapter_lifetime_required"):
            self.adapter.bind_physical_budget(guard, KEY)

    def test_direct_call_outside_active_operation_refused_before_backend(self):
        guard = self.reserve()
        self.prepare(guard)
        with mock.patch.object(self.adapter, "_predict_backend") as backend:
            with self.assertRaisesRegex(CampaignRunnerError, "funded_adapter_call_not_active"):
                self.adapter.predict(features((4.0, 5.0), "query"), 9)
            backend.assert_not_called()

    def test_reprepare_never_resets_bound_lifetime(self):
        guard = self.reserve()
        self.prepare(guard)
        with self.assertRaisesRegex(AdapterError, "funded_adapter_prepare_replay"):
            self.adapter.prepare(training(), 9)
        self.assertEqual(self.adapter._wall_seconds, 400.0)
        with self.assertRaisesRegex(AdapterError, "funded_adapter_lifetime_required"):
            self.adapter.bind_physical_budget(guard, KEY)

    def test_failed_backend_time_stays_charged_and_has_no_retry(self):
        guard = self.reserve()
        self.prepare(guard)
        with mock.patch.object(self.adapter, "_predict_backend", side_effect=ValueError("private source value")):
            with self.assertRaisesRegex(AdapterError, "backend_failed"):
                self.predict(guard, PREDICT)
        self.assertEqual(self.adapter._wall_seconds, 800.0)
        with self.assertRaisesRegex(CampaignRunnerError, "physical_operation_not_admitted"):
            guard.begin(PREDICT, "predict")
        self.assertEqual(guard.outcomes[PREDICT]["status"], "failed")
        charged = self.journal.charged
        self.journal.settle_operations(CELL, "learners", guard)
        physical = self.journal.state[(CELL, "learners")]["physical"]
        self.assertTrue(physical["settled"])
        self.assertFalse(physical["completed"])
        self.assertEqual(self.journal.charged, charged)

    def test_per_call_overrun_does_not_borrow_other_prediction_slots(self):
        guard = self.reserve()
        with self.assertRaisesRegex(CampaignRunnerError, "physical_operation_over_budget"):
            self.prepare(guard, 601.0)
        self.assertEqual(self.adapter._wall_seconds, 601.0)
        self.assertTrue(guard.violation)
        self.assertEqual(guard.outcomes[KEY]["status"], "over_budget")

    def test_mutated_budget_or_closed_journal_refuses_before_call(self):
        guard = self.reserve()
        self.prepare(guard)
        self.adapter._physical_budget["wall_ms"] += 1
        guard.begin(PREDICT, "predict")
        with self.assertRaisesRegex(CampaignRunnerError, "funded_adapter_call_not_active"):
            self.adapter.predict(features((4.0, 5.0), "query"), 9)
        self.adapter._physical_budget["wall_ms"] -= 1
        self.journal.__exit__(None, None, None)
        with self.assertRaisesRegex(CampaignRunnerError, "funded_adapter_lifetime_required"):
            self.adapter.predict(features((4.0, 5.0), "query"), 9)

    def test_v2_refuses_mixed_fit_roots_and_unknown_policy(self):
        doc = self.graph().document()
        second = "9" * 64
        doc["nodes"].insert(1, {"key": second, "kind": "prepare", "parents": [],
                                 "wall_ms": 600000, "reserved_child_fits": 1,
                                 "reserved_optimizer_updates": 0})
        doc["nodes"][2]["parents"].append(second)
        with self.assertRaisesRegex(CampaignRunnerError, "physical_lifetime_parent_invalid"):
            PhysicalOperationGraph.from_document(doc)
        doc = self.graph().document()
        doc["adapter_lifetime_policy"] = "unlimited"
        with self.assertRaisesRegex(CampaignRunnerError, "physical_lifetime_policy_invalid"):
            PhysicalOperationGraph.from_document(doc)

    def test_v1_graph_and_receipt_vocabulary_remain_unchanged(self):
        doc = self.graph(False).document()
        self.assertEqual(doc["format"], FORMAT)
        self.assertNotIn("adapter_lifetime_policy", doc)
        self.adapter.prepare(training(), 9)
        self.assertEqual(self.adapter.receipt()["format"], "dope-adapter-execution-v1")
        self.assertNotIn("physical_budget", self.adapter.receipt())

    def test_full_streamed_fixture_consumes_v2_budget_and_keeps_aliases(self):
        from research.transfer.pilot_learning_contract import DriverPlan, LEARNERS, QueryLabels
        from research.transfer.pilot_learning_execution import LearningExecution
        from research.transfer.pilot_learning_runtime import OperationAdmission, registry_digest
        from research.transfer.pilot_learning_reports import execute_streamed
        from research.transfer.tests.test_pilot_learning_execution import make_plan, make_registry, make_null
        plan = replace(make_plan(), model_retention="stream_drivers_then_release")
        entries, null = make_registry(), make_null()
        admission = OperationAdmission("fixture", plan.sha256, registry_digest(entries), PIN, PIN, LEARNERS, null.sha256)
        evidence = Path(self.tmp.name) / "execution"
        evidence.mkdir(mode=0o700)
        execution = LearningExecution(plan, entries, null, admission, lambda _: None, evidence)
        labels = QueryLabels(plan.query.row_ids, (1.0, 2.0), PIN)
        drivers = DriverPlan(((PIN, (0,)),), 2, 17, PIN)
        wall = {role: {"prepare": 100000, "predict": 100000} for role in (*LEARNERS, "null_reference")}
        guard = reserve_learning_operations(execution, self.journal, CELL, wall, driver_plan=drivers,
                                            labels=labels, adapter_lifetime_policy=LIFETIME_POLICY)
        execution, _ = execute_streamed(execution, labels, drivers)
        self.assertEqual(sum(c["status"] == "completed" for c in execution.cases), 60)
        intents = [r for r in execution.evidence.records if r["event"] == "intent"]
        self.assertEqual(len(intents), len(guard.nodes))
        receipts = [r["output"]["receipt"] for r in execution.evidence.records
                    if r["event"] == "outcome" and r["kind"] == "prepare"]
        self.assertTrue(receipts and all(r["format"] == "dope-adapter-execution-v2" for r in receipts))
        self.journal.settle_operations(CELL, "learners", guard)

    def test_one_fit_can_bind_only_one_actual_adapter(self):
        guard = self.reserve()
        self.prepare(guard)
        other = AffineLearner(self.model, AdapterAdmission.from_registry(registry(self.model), self.model, "learner"))
        with self.assertRaisesRegex(AdapterError, "funded_adapter_lifetime_required"):
            other.bind_physical_budget(guard, KEY)
        guard.active = KEY  # adversarial same-key re-entry cannot mint another owner
        with self.assertRaisesRegex(CampaignRunnerError, "funded_adapter_already_bound"):
            other.bind_physical_budget(guard, KEY)

    def test_active_key_allows_one_call_and_cannot_outlive_guard(self):
        guard = self.reserve()
        self.prepare(guard)
        guard.begin(PREDICT, "predict")
        query = features((4.0, 5.0), "query")
        self.adapter.predict(query, 9)
        with mock.patch.object(self.adapter, "_predict_backend") as backend:
            with self.assertRaisesRegex(CampaignRunnerError, "funded_adapter_call_not_active"):
                self.adapter.predict(query, 9)
            backend.assert_not_called()
        guard.finish(PREDICT, "completed", 1000, {})
        guard.begin(OTHER, "predict")
        with mock.patch("research.transfer.pilot_physical_operations.time.monotonic_ns",
                        return_value=guard.start_ns + guard.wall_ns):
            with self.assertRaisesRegex(CampaignRunnerError, "funded_adapter_call_not_active"):
                self.adapter.predict(query, 9)

    def test_dead_adapter_weakref_cannot_authorize_a_replacement(self):
        import gc
        import weakref
        guard = self.reserve()
        guard.begin(KEY, "prepare")
        self.adapter.bind_physical_budget(guard, KEY)
        budget = dict(self.adapter._physical_budget)
        previous = weakref.ref(self.adapter)
        self.adapter = None
        gc.collect()
        self.assertIsNone(previous())
        replacement = AffineLearner(self.model, AdapterAdmission.from_registry(registry(self.model), self.model, "learner"))
        with self.assertRaisesRegex(CampaignRunnerError, "funded_adapter_already_bound"):
            replacement.bind_physical_budget(guard, KEY)
        with self.assertRaisesRegex(CampaignRunnerError, "funded_adapter_call_not_active"):
            guard.adapter_call(budget, "prepare", replacement)
