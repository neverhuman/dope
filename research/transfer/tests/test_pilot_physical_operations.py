"""Actual fixture execution and durable refusal paths, never scientific proof."""
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from research.transfer import pilot_learning_execution as learner_module
from research.transfer.campaign_runner import CampaignRunSpec, CampaignRunnerError, StageJournal
from research.transfer.pilot_learning_contract import DriverPlan, LEARNERS, identity
from research.transfer.pilot_learning_execution import LearningExecution, PhysicalEvidence
from research.transfer.pilot_learning_reports import execute_streamed
from research.transfer.pilot_learning_runtime import OperationAdmission, registry_digest
from research.transfer.pilot_physical_operations import (
    FORMAT, PhysicalOperationGraph, issue_guard, reserve_learning_operations,
)
from research.transfer.tests.test_campaign_runner import specification
from research.transfer.tests.test_pilot_learning_execution import (
    FixtureLearner, FixtureNull, PIN, make_null, make_plan, make_registry, QueryLabels,
)

CELL = "5" * 64
KEY = "6" * 64
PREDICT = "7" * 64


def fixture_spec():
    document = specification()
    document["budget"] = {"wall_ms": 5000000, "peak_ram_bytes": 1 << 30}
    document["environment_sha256"] = PIN
    document["cells"][0]["input_sha256"] = PIN
    return CampaignRunSpec.from_document(document)


def tiny_graph(wall=1000):
    return PhysicalOperationGraph.from_document({
        "format": FORMAT, "learning_plan_sha256": PIN, "registry_sha256": PIN,
        "null_reference_sha256": PIN, "driver_plan_sha256": None,
        "campaign_plan_sha256": fixture_spec().sha256, "cell_id": CELL,
        "operation_admission_sha256": PIN,
        "nodes": [{"key": KEY, "kind": "prepare", "parents": [], "wall_ms": wall,
                   "reserved_child_fits": 1, "reserved_optimizer_updates": 0},
                  {"key": PREDICT, "kind": "predict", "parents": [KEY], "wall_ms": wall,
                   "reserved_child_fits": 8, "reserved_optimizer_updates": 400}],
        "aliases": [{"logical_key": PIN, "operations": [KEY, PREDICT]},
                    {"logical_key": "b" * 64, "operations": [KEY, PREDICT]}]})


class PhysicalOperationTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[3] / "target/pilot-physical-fixtures"
        base.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = fixture_spec()
        FixtureLearner.prepares = FixtureLearner.predictions = 0
        FixtureLearner.fault = FixtureNull.fault = None

    def journal(self, create=True):
        return StageJournal(self.root, self.spec, create=create)

    def begin_learners(self, journal, wall=200000):
        for stage in ("prepare", "compile", "reload", "sample"):
            journal.begin(CELL, stage, 10)
            journal.finish(CELL, stage, "completed", observed_wall_ms=1,
                           peak_ram_bytes=512, output_sha256=PIN)
        journal.begin(CELL, "learners", wall)

    def execution(self, plan=None, root=None):
        plan = make_plan() if plan is None else plan
        registry, null = make_registry(), make_null()
        admission = OperationAdmission("fixture", plan.sha256, registry_digest(registry),
                                       PIN, PIN, LEARNERS, null.sha256)
        return LearningExecution(plan, registry, null, admission, lambda _: None, self.root if root is None else root)

    def test_full_streamed_fixture_matches_prospective_keys_and_physical_aliases(self):
        execution = self.execution(replace(make_plan(), model_retention="stream_drivers_then_release"))
        labels = QueryLabels(execution.plan.query.row_ids, (1.0, 2.0), PIN)
        drivers = DriverPlan(((PIN, (0,)),), 2, 17, PIN)
        wall = {role: {"prepare": 10000, "predict": 10000} for role in (*LEARNERS, "null_reference")}
        with self.journal() as journal:
            self.begin_learners(journal, 2000000)
            charged = journal.charged
            guard = reserve_learning_operations(execution, journal, CELL, wall,
                                                driver_plan=drivers, labels=labels)
            graph = guard.graph.document()
            self.assertEqual(len(graph["aliases"]), 60)
            self.assertLess(len(graph["nodes"]), 160)
            self.assertEqual(journal.charged, charged)  # stage charged once, aliases never charged again
            mitra = [n for n in graph["nodes"] if n["reserved_child_fits"] == 8]
            self.assertLess(len(mitra), 24)  # exact-query baselines/duplicates are physical aliases
            self.assertTrue(all(n["reserved_optimizer_updates"] == 400 for n in mitra))
            execution, actual_drivers = execute_streamed(execution, labels, drivers)
            self.assertEqual(sum(c["status"] == "completed" for c in execution.cases), 60,
                             {"refusals": guard.refusals[:3], "noncompleted": [r for r in
                              guard.settlement() if r["status"] != "completed"][:3]})
            self.assertEqual(FixtureLearner.prepares, 32)
            self.assertEqual(FixtureLearner.predictions,
                             sum(n["kind"] == "predict" for n in graph["nodes"]))
            self.assertLess(FixtureLearner.predictions, 128)
            self.assertTrue(all(r["status"] == "completed" for r in guard.settlement()))
            self.assertTrue(all(r["observed_optimizer_updates"] is None for r in guard.settlement()))
            self.assertEqual(len(actual_drivers["links"]), 60)
            journal.settle_operations(CELL, "learners", guard)
            journal.finish(CELL, "learners", "completed", observed_wall_ms=100,
                           peak_ram_bytes=1024, output_sha256=PIN)
            self.assertEqual(journal.charged, charged)
            self.assertEqual(len(list(self.root.glob("development-stages.jsonl"))), 1)
            self.assertFalse(journal.public_summary()["worker_executed"])

    def test_development_execution_and_direct_predict_require_actual_reservation(self):
        execution = self.execution()
        execution.fixture = False  # exercise refusal before any fixture/native call
        with self.assertRaisesRegex(ValueError, "physical_reservation_required"):
            execution.execute()
        with self.assertRaisesRegex(ValueError, "physical_reservation_required"):
            execution.predict(None, None, None, 1, KEY, parent=PREDICT)
        self.assertEqual((FixtureLearner.prepares, FixtureLearner.predictions), (0, 0))

    def test_reservation_requires_pending_stage_and_suballocation_fits_existing_wall(self):
        with self.journal() as journal:
            before = (self.root / "development-stages.jsonl").read_bytes()
            with self.assertRaises(CampaignRunnerError):
                journal.reserve_operations(CELL, "learners", tiny_graph())
            self.assertEqual((self.root / "development-stages.jsonl").read_bytes(), before)
            self.begin_learners(journal, 100)
            before = (self.root / "development-stages.jsonl").read_bytes()
            with self.assertRaisesRegex(CampaignRunnerError, "physical_stage_budget_exceeded"):
                journal.reserve_operations(CELL, "learners", tiny_graph())
            self.assertEqual((self.root / "development-stages.jsonl").read_bytes(), before)
            self.assertEqual(journal.charged, 140)

    def test_changed_plan_and_inadequate_native_timeout_refuse_before_work(self):
        execution = self.execution()
        with self.journal() as journal:
            self.begin_learners(journal)
            guard = journal.reserve_operations(CELL, "learners", tiny_graph())
            execution.evidence.guard = guard
            with self.assertRaisesRegex(ValueError, "physical_reservation_binding_mismatch"):
                execution.execute()
        self.assertEqual(FixtureLearner.prepares, 0)
        root = self.root / "timeout"
        root.mkdir(mode=0o700)
        with StageJournal(root, self.spec, create=True) as journal:
            self.begin_learners(journal)
            execution.fixture = False
            wall = {r: {"prepare": 1, "predict": 1} for r in (*LEARNERS, "null_reference")}
            before = (root / "development-stages.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "native_timeout_exceeds_physical_allocation"):
                reserve_learning_operations(execution, journal, CELL, wall)
            self.assertEqual((root / "development-stages.jsonl").read_bytes(), before)

    def test_requested_children_cannot_exceed_reserved_envelope_or_be_reported_as_measured(self):
        with self.journal() as journal:
            self.begin_learners(journal)
            guard = journal.reserve_operations(CELL, "learners", tiny_graph())
            evidence = PhysicalEvidence(self.root, guard)
            evidence.call(KEY, "prepare", {}, lambda: (1, {}))
            with self.assertRaisesRegex(CampaignRunnerError, "physical_operation_over_budget"):
                evidence.call(PREDICT, "predict", {}, lambda: (1, {
                    "native_attempts": [{"requested_child_fits": 9}]}))
            self.assertIsNone(guard.settlement()[1]["observed_child_fits"])
            self.assertEqual(guard.settlement()[1]["status"], "over_budget")

    def test_foreign_campaign_input_environment_or_cell_refuses_without_journal_mutation(self):
        wall = {r: {"prepare": 10000, "predict": 10000} for r in (*LEARNERS, "null_reference")}
        for field in ("input_sha256", "environment_sha256"):
            root = self.root / field
            root.mkdir(mode=0o700)
            registry = make_registry()
            plan = make_plan()
            # Change the campaign, not the already-authorized learning plan.
            document = self.spec.document()
            if field == "input_sha256":
                document["cells"][0][field] = "d" * 64
            else:
                document[field] = "d" * 64
            execution = self.execution(plan, root)
            with StageJournal(root, CampaignRunSpec.from_document(document), create=True) as journal:
                self.begin_learners(journal)
                before = (root / "development-stages.jsonl").read_bytes()
                with self.assertRaisesRegex(ValueError, "physical_campaign_input_environment_mismatch"):
                    reserve_learning_operations(execution, journal, CELL, wall)
                self.assertEqual((root / "development-stages.jsonl").read_bytes(), before)
        with self.journal() as journal:
            self.begin_learners(journal)
            document = tiny_graph().document()
            document["campaign_plan_sha256"] = "d" * 64
            before = (self.root / "development-stages.jsonl").read_bytes()
            with self.assertRaisesRegex(CampaignRunnerError, "physical_campaign_binding_mismatch"):
                journal.reserve_operations(CELL, "learners", PhysicalOperationGraph.from_document(document))
            document = tiny_graph().document()
            document["cell_id"] = "d" * 64
            with self.assertRaises(CampaignRunnerError):
                journal.reserve_operations(CELL, "learners", PhysicalOperationGraph.from_document(document))
            self.assertEqual((self.root / "development-stages.jsonl").read_bytes(), before)

    def test_mutated_guard_nodes_do_not_change_the_durable_graph(self):
        with self.journal() as journal:
            self.begin_learners(journal)
            guard = journal.reserve_operations(CELL, "learners", tiny_graph())
            guard.nodes[KEY]["wall_ms"] += 1000
            evidence = PhysicalEvidence(self.root, guard)
            called = mock.Mock()
            with self.assertRaises(CampaignRunnerError):
                evidence.call(KEY, "prepare", {}, called)
            called.assert_not_called()

    def test_reload_preserves_charge_and_cannot_reissue_a_pending_graph(self):
        with self.journal() as journal:
            self.begin_learners(journal)
            graph = tiny_graph()
            guard = journal.reserve_operations(CELL, "learners", graph)
            reservation = guard.reservation_sha256
            with self.assertRaises(CampaignRunnerError):
                with self.journal(create=False):  # no child can reopen the parent's lock
                    pass
        with self.journal(create=False) as journal:
            self.assertEqual(journal.charged, 200040)
            before = (self.root / "development-stages.jsonl").read_bytes()
            with self.assertRaises(CampaignRunnerError):
                journal.reserve_operations(CELL, "learners", graph)
            with self.assertRaisesRegex(CampaignRunnerError, "fresh_physical_reservation_required"):
                issue_guard(graph, reservation, journal, CELL, "learners")
            with self.assertRaises(CampaignRunnerError):
                guard.begin(KEY, "prepare")  # old closed issuer cannot execute either
            self.assertEqual((self.root / "development-stages.jsonl").read_bytes(), before)

    def test_failure_retains_unexecuted_nodes_and_forbids_implicit_retry(self):
        with self.journal() as journal:
            self.begin_learners(journal)
            guard = journal.reserve_operations(CELL, "learners", tiny_graph())
            evidence = PhysicalEvidence(self.root, guard)
            with self.assertRaisesRegex(RuntimeError, "private-backend-error"):
                evidence.call(KEY, "prepare", {}, lambda: (_ for _ in ()).throw(RuntimeError("private-backend-error")))
            self.assertNotIn("private-backend-error", str(evidence.records))
            called = mock.Mock(return_value=(1, {}))
            with self.assertRaises(CampaignRunnerError):
                evidence.call(KEY, "prepare", {}, called)
            called.assert_not_called()
            self.assertEqual([r["status"] for r in guard.settlement()], ["failed", "not_run"])
            journal.settle_operations(CELL, "learners", guard)
            with self.assertRaisesRegex(CampaignRunnerError, "physical_operations_incomplete"):
                journal.finish(CELL, "learners", "completed", observed_wall_ms=2,
                               peak_ram_bytes=512, output_sha256=PIN)
            journal.finish(CELL, "learners", "failed", observed_wall_ms=2, peak_ram_bytes=512)

    def test_crash_can_record_terminal_unknowns_without_guard_refund_or_reissue(self):
        with self.journal() as journal:
            self.begin_learners(journal)
            guard = journal.reserve_operations(CELL, "learners", tiny_graph())
            evidence = PhysicalEvidence(self.root, guard)
            evidence.call(KEY, "prepare", {}, lambda: (1, {}))
            # Process exits before parent receives a durable complete settlement.
        with self.journal(create=False) as journal:
            with self.assertRaises(CampaignRunnerError):
                journal.record_interrupted_operations(CELL, "learners", PIN, None)
            before = (self.root / "development-stages.jsonl").read_bytes()
            with self.assertRaises(CampaignRunnerError):
                journal.record_interrupted_operations(CELL, "learners", PIN, lambda _: False)
            self.assertEqual((self.root / "development-stages.jsonl").read_bytes(), before)
            journal.record_interrupted_operations(CELL, "learners", PIN, lambda _: None)
            outcomes = journal.state[(CELL, "learners")]["physical"]["outcomes"]
            self.assertEqual([r["status"] for r in outcomes], ["unknown", "unknown"])
            self.assertTrue(all(r["wall_ms"] is None for r in outcomes))
            self.assertEqual(journal.charged, 200040)
            with self.assertRaises(CampaignRunnerError):
                journal.record_interrupted_operations(CELL, "learners", PIN, lambda _: None)
        # A second crash between interruption disposition and finish is also terminalizable.
        with self.journal(create=False) as journal:
            journal.finish(CELL, "learners", "interrupted")
            self.assertEqual(journal.public_summary()["pending_stages"], 0)
            self.assertEqual(journal.public_summary()["unknown_physical_operations"], 2)
            self.assertEqual(journal.charged, 200040)
            for retry in (False, True):
                with self.assertRaisesRegex(CampaignRunnerError, "physical_reconciliation_required"):
                    journal.begin(CELL, "learners", 100, retry=retry)
            with self.assertRaises(CampaignRunnerError):
                journal.reserve_operations(CELL, "learners", tiny_graph())
            with self.assertRaises(CampaignRunnerError):
                issue_guard(tiny_graph(), guard.reservation_sha256, journal, CELL, "learners")
        with StageJournal(self.root, self.spec, read_only=True) as journal:
            called = mock.Mock()
            with self.assertRaises(CampaignRunnerError):
                journal.record_interrupted_operations(CELL, "learners", PIN, called)
            called.assert_not_called()

    def test_overrun_is_recorded_before_next_call_and_stops_the_campaign(self):
        with self.journal() as journal:
            self.begin_learners(journal)
            guard = journal.reserve_operations(CELL, "learners", tiny_graph(wall=1000))
            evidence = PhysicalEvidence(self.root, guard)
            fake = SimpleNamespace(monotonic_ns=mock.Mock(side_effect=[0, 1001000000]), process_time_ns=lambda: 0)
            with mock.patch.object(learner_module, "time", fake):
                with self.assertRaisesRegex(CampaignRunnerError, "physical_operation_over_budget"):
                    evidence.call(KEY, "prepare", {}, lambda: (1, {}))
            self.assertEqual(evidence.records[-1]["status"], "over_budget")
            called = mock.Mock()
            with self.assertRaises(CampaignRunnerError):
                evidence.call(PREDICT, "predict", {}, called)
            called.assert_not_called()
            journal.settle_operations(CELL, "learners", guard)
            with self.assertRaises(CampaignRunnerError):
                journal.finish(CELL, "learners", "failed")
            journal.finish(CELL, "learners", "over_budget", observed_wall_ms=2, peak_ram_bytes=512)
            with self.assertRaises(CampaignRunnerError):
                journal.begin(CELL, "learners", 100, retry=True)
        with self.journal(create=False) as journal:
            self.assertTrue(journal.over_budget)

    def test_unplanned_or_dependency_violating_call_never_invokes_backend(self):
        for key in ("d" * 64, PREDICT):
            root = self.root / key[:1]
            root.mkdir(mode=0o700)
            with StageJournal(root, self.spec, create=True) as journal:
                self.begin_learners(journal)
                guard = journal.reserve_operations(CELL, "learners", tiny_graph())
                evidence = PhysicalEvidence(root, guard)
                called = mock.Mock()
                with self.assertRaises(CampaignRunnerError):
                    evidence.call(key, "predict", {}, called)
                called.assert_not_called()
                self.assertEqual(evidence.records, [])

    def test_graph_direct_constructor_malformed_alias_cycle_and_boolean_budget_refuse(self):
        original = tiny_graph().document()
        for field, change in (("wall_ms", True), ("parents", [PREDICT]), ("key", "SECRET")):
            doc = tiny_graph().document()
            doc["nodes"][0][field] = change
            with self.assertRaises(CampaignRunnerError) as error:
                PhysicalOperationGraph.from_document(doc)
            self.assertNotIn("SECRET", str(error.exception))
        for edit in (lambda d: d["nodes"].append(d["nodes"][0]),
                     lambda d: d["aliases"].clear(),
                     lambda d: d["aliases"][0]["operations"].append({}),
                     lambda d: d["aliases"].append(d["aliases"][0])):
            doc = tiny_graph().document()
            edit(doc)
            with self.assertRaises(CampaignRunnerError):
                PhysicalOperationGraph.from_document(doc)
        original["nodes"][0]["wall_ms"] = 0
        from research.transfer.evidence import canonical_json
        with self.assertRaises(CampaignRunnerError):
            PhysicalOperationGraph(canonical_json(original)).document()

    def test_new_guard_is_not_issued_if_reservation_fsync_fails(self):
        with self.journal() as journal:
            self.begin_learners(journal)
            with mock.patch("research.transfer.campaign_runner.os.fsync", side_effect=OSError()):
                with self.assertRaises(CampaignRunnerError):
                    journal.reserve_operations(CELL, "learners", tiny_graph())
            self.assertIsNone(journal.fd)
        with self.journal(create=False) as journal:
            self.assertFalse(journal.state[(CELL, "learners")]["physical"]["settled"])
            self.assertEqual(journal.charged, 200040)


if __name__ == "__main__":
    unittest.main()
