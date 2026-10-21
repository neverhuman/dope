"""Real native n/4n bridge with fixture learners; never empirical qualification."""
import base64
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from research.transfer.adapters.base import AdapterAdmission
from research.transfer.campaign_artifacts import create_private, file_hash, read_regular
from research.transfer.campaign_runner import CampaignRunSpec, CampaignRunnerError, StageJournal
from research.transfer.evidence import canonical_json
from research.transfer.pilot_cell_plan import PilotCellExecutionPlan
from research.transfer.pilot_learning_contract import ARMS, CONTEXTS, DriverPlan, LEARNERS, SCALES, identity
from research.transfer.pilot_native_seed import NativeSeed, ORIGINAL_STREAM_HEX, OWNER_GRANT_SHA256
from research.transfer.pilot_runner import execute_pilot_cell
from research.transfer.pilot_worker_authority import WorkerAuthority
from research.transfer.pilot_worker_runtime import PilotWorkerRuntime
from research.transfer.tests.test_pilot_learning_execution import (
    FixtureLearner, FixtureNull, PIN, make_null, make_registry,
)
from research.transfer.tests.test_pilot_runner import fixture


class BoundWorkerTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[3]
        self.binary = base / "target/debug/dope-kernel"
        if not self.binary.is_file():
            subprocess.run(["cargo", "build", "--locked", "--bin", "dope-kernel"], cwd=base, check=True)
        parent = base / "target/pilot-bound-runtime-fixtures"
        parent.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        FixtureLearner.fault = FixtureNull.fault = None
        FixtureLearner.prepares = FixtureLearner.predictions = 0

    def prepare(self, *, budget=60000, wrong_rights=False, bad_index=False, verifier=None,
                learner_seed="0x00000000000006c1", comparator="dope",
                control_runtime=None, byte_budget=32768, pool_policy=False):
        # Fixture default explicitly equals its compiler seed; production has no default.
        spec, pilot, contract, cell, root, auth = fixture(
            self.directory, stage_wall_ms=10000000, binary_source=self.binary, version=2, comparator=comparator,
            candidate="chow_liu" if comparator == "dope" else comparator, byte_budget=byte_budget)
        doc = cell.document()
        registry, null = make_registry(), make_null()
        def adapt(entry):
            model = replace(entry.model, environment_sha256=spec.document()["environment_sha256"],
                            context_cap=16, query_batch_size=16)
            return replace(entry, model=model,
                           admission=AdapterAdmission(model.sha256, "7" * 64, PIN, "learner"))
        registry, null = {name: adapt(entry) for name, entry in registry.items()}, adapt(null)
        parts = {}
        for role, key in (("fit", "csv"), ("selection", "selection_csv"), ("query", "query_csv")):
            parts[role] = {"csv_sha256": doc[key + "_sha256"], "rows": [
                {"row_id": identity(["fixture", role, i]),
                 "physical_unit_sha256": identity(["fixture-unit", role, i])}
                for i in range(doc[role + "_rows"])]}
        roles = self.directory / "private-physical-roles.json"
        contexts = self.directory / "private-context-indices.json"
        create_private(roles, canonical_json({"format": "dope-pilot-private-physical-membership-v1",
            "schema_sha256": doc["schema_sha256"], "partitions": parts}))
        if pool_policy:
            from research.transfer.pilot_worker_inputs import pool_context_policy_sha256, POOL_CONTEXT_FORMAT
            items = []
            for name, rule in (("standard", "first_rows"), ("stratified_anchor", "fit_target_stratified")):
                seed = "0x000000000000000b"
                items.append({"name": name, "rows": 16, "seed": seed, "selection_policy": rule,
                              "policy_sha256": pool_context_policy_sha256(name,16,seed,rule)})
            create_private(contexts, canonical_json({"format":POOL_CONTEXT_FORMAT,"contexts":items}))
        else:
            create_private(contexts, canonical_json({"format": "dope-pilot-private-context-indices-v1",
                "contexts": [{"name": name, "rows": 16, "seed": 11, "policy_sha256": PIN,
                    "selections": [{"scale": scale, "arm": arm,
                        "indices": list(range(16)) if not bad_index else [999] + list(range(1, 16))}
                        for scale in SCALES for arm in ARMS]} for name in CONTEXTS]}))
        source = spec.document()
        authority = WorkerAuthority(cell.input_sha256, str(roles), str(contexts), file_hash(roles),
            file_hash(contexts), identity({"source_commit": source["source_commit"], "source_tree": source["source_tree"]}),
            source["environment_sha256"], PIN, PIN, NativeSeed(learner_seed).sha256, learner_seed, null.sha256,
            doc["csv_sha256"], 0, "deterministic_partition_zero", "a" * 64, "b" * 64, "a" * 64)
        driver = DriverPlan(((identity("opaque-feature-group"), (0,)),), 1, doc["driver_seed"], PIN)
        worker = PilotWorkerRuntime(authority, registry, null, driver,
            {name: {"prepare": budget, "predict": budget} for name in (*LEARNERS, "null_reference")},
            verify_assembly=(lambda binding: None) if verifier is None else verifier,
            verify_operation=lambda admission: None, kind="fixture", control_runtime=control_runtime)
        doc["worker_binary"] = str(self.directory / "bound-worker-capsule.json")
        create_private(Path(doc["worker_binary"]), worker.capsule(), 0o400)
        doc["worker_binary_sha256"] = file_hash(Path(doc["worker_binary"]))
        if wrong_rights:
            doc["learner_rights_sha256"]["mitra"] = "f" * 64
        cell = PilotCellExecutionPlan.from_document(doc)
        spec_doc = spec.document()
        spec_doc["cells"][0]["command_sha256"] = cell.sha256
        spec = CampaignRunSpec.from_document(spec_doc)
        envelope = json.loads(read_regular(auth["admission"]))
        envelope["payload"].update(run_spec_sha256=spec.sha256, cell_execution_plan_sha256=cell.sha256)
        message, sig = self.directory / "fresh-message", self.directory / "fresh-signature"
        create_private(message, canonical_json(envelope["payload"]))
        subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", str(self.directory / "test-private.pem"),
                        "-rawin", "-in", str(message), "-out", str(sig)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        envelope["signature_ed25519_base64"] = base64.b64encode(read_regular(sig)).decode()
        auth["admission"].write_bytes(canonical_json(envelope))
        return spec, pilot, contract, cell, root, auth, worker

    def run_bound(self, prepared):
        spec, pilot, contract, cell, root, auth, worker = prepared
        return execute_pilot_cell(spec, pilot, contract, cell, root, worker_runtime=worker, **auth)

    def test_frozen_per_pool_policy_runs_actual_native_eight_stages(self):
        receipt = self.run_bound(self.prepare(pool_policy=True))
        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(len(receipt["stages"]), 8)
        self.assertGreater(FixtureLearner.prepares, 0)

    def test_actual_native_draws_reach_sixty_fixture_paths_under_one_live_journal(self):
        prepared = self.prepare()
        receipt = self.run_bound(prepared)
        root, worker = prepared[4], prepared[6]
        self.assertEqual(receipt["status"], "ok", receipt)
        self.assertEqual(len(receipt["stages"]), 8)
        self.assertEqual(receipt["learner_driver_runtime"], "fixture")
        self.assertFalse(receipt["pipeline_complete"])
        self.assertIn("bound_learner_driver_metric_runtime_implementation", receipt["pipeline_prerequisites"])
        self.assertTrue(receipt["fixture_validation_complete"])
        self.assertFalse(worker.development_admission_used)
        private = json.loads(read_regular(root / "private-learners.json"))
        self.assertEqual(private["execution"]["completed_path_count"], 60)
        self.assertTrue(all(len(row["losses"]) == 3 for row in private["paired"]["comparisons"]))
        drivers = json.loads(read_regular(root / "private-drivers.json"))
        self.assertEqual(len(drivers["links"]), 60)
        self.assertTrue(all(row["status"] == "completed" for row in drivers["links"]))
        self.assertEqual(len(worker._execution.evidence.guard.graph.document()["aliases"]), 60)
        self.assertEqual(len(worker._execution.prepared), FixtureLearner.prepares)
        self.assertNotIn("SECRET", json.dumps(receipt))
        self.assertFalse(receipt["scientific_admission"])

    def test_actual_distinct_compiler_and_learner_uint64_streams_reach_learning(self):
        from research.transfer.tests import test_pilot_runner as runner_fixture
        pilot_doc = runner_fixture._pilot_plan_doc()
        compiler = "0x000d0beef2026101"
        pilot_doc["seeds"]["substreams"].update(fit=compiler, learner=ORIGINAL_STREAM_HEX)
        with mock.patch.object(runner_fixture, "_pilot_plan_doc", return_value=pilot_doc):
            prepared = self.prepare(learner_seed=ORIGINAL_STREAM_HEX)
        compiler_before = prepared[3].document()["fit_seed"]
        receipt = self.run_bound(prepared)
        self.assertEqual(receipt["status"], "ok", receipt)
        self.assertEqual(compiler_before, compiler)
        self.assertEqual(prepared[3].document()["fit_seed"], compiler)
        plan = prepared[6]._execution.plan
        self.assertEqual(plan.native_seed.original_stream_hex, ORIGINAL_STREAM_HEX)
        self.assertEqual(plan.fit_seed, 2858805439)
        self.assertEqual(plan.predict_seed, 2858805439)
        self.assertNotEqual(NativeSeed(compiler).sha256, plan.native_seed.sha256)
        assembly = json.loads(read_regular(prepared[4] / "private-input-assembly.json"))
        self.assertEqual(assembly["compiler_fit_seed"], compiler)
        self.assertEqual(assembly["learner_seed"], ORIGINAL_STREAM_HEX)
        capsule = json.loads(prepared[6].capsule())
        self.assertEqual(capsule["authority"]["learner_seed"], ORIGINAL_STREAM_HEX)
        self.assertEqual(capsule["authority"]["native_seed_sha256"], plan.native_seed.sha256)
        self.assertFalse(receipt["scientific_admission"])

    def test_seed_mutation_refuses_before_native_or_model_work(self):
        prepared = self.prepare()
        object.__setattr__(prepared[6].authority, "learner_seed", "0xffffffffffffffff")
        with mock.patch("research.transfer.pilot_runner.invoke") as native:
            receipt = self.run_bound(prepared)
        native.assert_not_called()
        self.assertEqual(receipt["stages"][0]["reason"], "pilot_worker_binding_invalid")
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_matching_wrong_seed_hash_cannot_become_development_authority(self):
        prepared = self.prepare()
        worker = prepared[6]
        wrong = "0xffffffffffffffff"
        authority = replace(worker.authority, learner_seed=wrong,
            native_seed_sha256=NativeSeed(wrong).sha256,
            owner_recipe_decision_sha256=OWNER_GRANT_SHA256)
        with mock.patch.object(PilotWorkerRuntime, "capsule") as capsule:
            with self.assertRaisesRegex(ValueError, "worker_owner_learner_stream_mismatch"):
                PilotWorkerRuntime(authority, worker.registry, worker.null_reference,
                    worker.driver_plan, worker.wall_by_role, verify_assembly=lambda binding: None,
                    verify_operation=lambda admission: None, kind="development")
        capsule.assert_not_called()
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_wrong_signed_role_rights_refuses_before_native_or_model_work(self):
        prepared = self.prepare(wrong_rights=True)
        with mock.patch("research.transfer.pilot_runner.invoke") as native:
            receipt = self.run_bound(prepared)
        native.assert_not_called()
        self.assertEqual(receipt["stages"][0]["reason"], "pilot_worker_binding_invalid")
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_insufficient_parent_physical_budget_refuses_all_model_calls(self):
        # One operation exceeds the entire 10,000,000 ms fixture stage.
        # This remains a refusal even when identical predictions are reused.
        receipt = self.run_bound(self.prepare(budget=10_000_001))
        self.assertEqual(receipt["stages"][-1]["stage"], "learners")
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_physical_overrun_records_terminal_over_budget_and_keeps_attempt(self):
        prepared = self.prepare(budget=1)
        original = FixtureLearner._prepare_backend
        def slow(adapter, training, seed, mode):
            time.sleep(0.02)
            return original(adapter, training, seed, mode)
        with mock.patch.object(FixtureLearner, "_prepare_backend", new=slow):
            receipt = self.run_bound(prepared)
        self.assertEqual(receipt["stages"][-1]["status"], "over_budget")
        self.assertEqual(receipt["journal"]["pending_stages"], 0)
        self.assertTrue(receipt["journal"]["resource_overrun"])
        self.assertTrue(prepared[6]._execution.evidence.guard.violation)

    def test_missing_context_row_refuses_all_model_calls(self):
        receipt = self.run_bound(self.prepare(bad_index=True))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_trusted_assembly_refusal_is_not_replaced_by_hash_matching(self):
        def refuse(binding):
            raise ValueError("not_admitted")
        receipt = self.run_bound(self.prepare(verifier=refuse))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_unexpected_trusted_callback_failure_finishes_stage_without_native_learning(self):
        def refuse(binding):
            raise RuntimeError("PRIVATE_CALLBACK_SENTINEL")
        prepared = self.prepare(verifier=refuse)
        receipt = self.run_bound(prepared)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["stages"][-1]["stage"], "learners")
        self.assertEqual(receipt["journal"]["pending_stages"], 0)
        self.assertEqual(FixtureLearner.prepares, 0)
        self.assertNotIn("PRIVATE_CALLBACK_SENTINEL", json.dumps(receipt))

    def test_unexpected_operation_admission_failure_finishes_without_model_calls(self):
        prepared = self.prepare()
        def refuse(admission):
            raise RuntimeError("PRIVATE_ADMISSION_SENTINEL")
        prepared[6].verify_operation = refuse
        receipt = self.run_bound(prepared)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["journal"]["pending_stages"], 0)
        self.assertEqual(FixtureLearner.prepares, 0)
        self.assertNotIn("PRIVATE_ADMISSION_SENTINEL", json.dumps(receipt))

    def test_unexpected_payload_verifier_failure_finishes_after_settlement(self):
        from research.transfer import pilot_runner as runner
        prepared = self.prepare()
        original = runner._verify_stage_payload
        def verify(label, *args):
            if label == "learners":
                raise RuntimeError("PRIVATE_PAYLOAD_VERIFIER")
            return original(label, *args)
        with mock.patch.object(runner, "_verify_stage_payload", side_effect=verify):
            receipt = self.run_bound(prepared)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["stages"][-1]["stage"], "learners")
        self.assertEqual(receipt["journal"]["pending_stages"], 0)
        self.assertNotIn("PRIVATE_PAYLOAD_VERIFIER", json.dumps(receipt))
        self.assertTrue((prepared[4] / "private-learners.json").is_file())

    def test_private_publication_io_failure_uses_parent_journal_without_another_file_write(self):
        prepared = self.prepare()
        with mock.patch("research.transfer.pilot_worker_runtime.create_private",
                        side_effect=OSError("PRIVATE_DISK_FAILURE")) as write:
            receipt = self.run_bound(prepared)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["journal"]["pending_stages"], 0)
        self.assertEqual(write.call_count, 1)
        self.assertFalse((prepared[4] / "learners-runtime-failure.json").exists())
        self.assertNotIn("PRIVATE_DISK_FAILURE", json.dumps(receipt))

    def test_unclassified_callback_bug_preserves_pending_reservation(self):
        class UnclassifiedFailure(Exception):
            pass
        def fail(binding):
            raise UnclassifiedFailure("PRIVATE_UNKNOWN_BUG")
        prepared = self.prepare(verifier=fail)
        with self.assertRaises(UnclassifiedFailure):
            self.run_bound(prepared)
        with StageJournal(prepared[4], prepared[0], create=False, read_only=True) as journal:
            self.assertEqual(journal.public_summary()["pending_stages"], 1)
        self.assertFalse((prepared[4] / "pilot-execution.json").exists())
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_durability_failure_cannot_fake_a_failed_settled_stage(self):
        prepared = self.prepare()
        with mock.patch.object(StageJournal, "settle_operations", side_effect=RuntimeError("PRIVATE_FSYNC")):
            with self.assertRaises(CampaignRunnerError):
                self.run_bound(prepared)
        with StageJournal(prepared[4], prepared[0], create=False, read_only=True) as journal:
            self.assertEqual(journal.public_summary()["pending_stages"], 1)
            self.assertTrue(journal.state[prepared[3].document()["cell_id"], "learners"]["pending"])

    def test_native_learner_failure_keeps_attempts_and_refuses_later_stages(self):
        prepared = self.prepare()
        FixtureLearner.fault = "partial"
        self.addCleanup(setattr, FixtureLearner, "fault", None)
        receipt = self.run_bound(prepared)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["stages"][-1]["stage"], "learners")
        private = json.loads(read_regular(prepared[4] / "private-learners.json"))
        self.assertEqual(private["execution"]["completed_path_count"], 0)
        self.assertTrue(any(row["event"] == "outcome" and row["status"] == "failed"
                            for row in private["execution"]["physical_operations"]))


if __name__ == "__main__":
    unittest.main()
