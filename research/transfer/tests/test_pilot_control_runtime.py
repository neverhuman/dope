"""Actual framed controls and confined children into fixture learners; unqualified.

Authored for later Root validation. These tests never confer scientific admission.
The positive fixture signs its own measured-closure budget before preparing the
cell. The real-cap negative retains 32768 and must never invoke a sampler/model.
"""
import copy
import json
from pathlib import Path
import unittest
from unittest import mock

from research.transfer.adapters.base import FeatureBatch
from research.transfer.campaign_artifacts import read_regular
from research.transfer.campaign_runner import CampaignRunnerError, StageJournal
from research.transfer.pilot_control_runtime import COMPILER_FILES, PilotControlRuntime
from research.transfer.pilot_materialized_draws import load_pair
from research.transfer.pilot_native_seed import ORIGINAL_STREAM_HEX
from research.transfer.tdl import control_boundary as boundary
from research.transfer.tests import test_control_boundary as boundary_fixtures
from research.transfer.tests import test_pilot_worker_runtime as worker_fixtures
from research.transfer.tests.test_pilot_learning_execution import FixtureLearner

GRANT = b"explicit synthetic typed-control executor fixture, not a development grant"


class FixtureExecutor:
    """Assert operation identity and sequence; not a production authorizer."""
    def __init__(self, runtime):
        self.runtime = copy.deepcopy(runtime)
        self.events, self.cell_sha = [], None

    def __call__(self, operation, claims, evidence):
        if evidence != GRANT or claims["cell_plan_sha256"] != self.cell_sha:
            raise ValueError("fixture_operation_not_admitted")
        if operation == "prepare":
            if self.events or claims["capsule"]["runtime"] != self.runtime:
                raise ValueError("fixture_runtime_mismatch")
            if claims["authorization"]["rights_admission_sha256"] != "a" * 64:
                raise ValueError("fixture_rights_mismatch")
        elif operation == "compile":
            if self.events != ["prepare"]:
                raise ValueError("fixture_compile_order")
        elif operation == "compiled":
            if self.events[-1] != "compile":
                raise ValueError("fixture_compile_witness_order")
        elif operation == "reload":
            if self.events[-1] != "compiled":
                raise ValueError("fixture_reload_order")
        elif operation == "sample":
            if self.events[-1] != "reload":
                raise ValueError("fixture_sample_order")
        elif operation == "sample_draw":
            request = claims["request"]
            if request["closure_pins"] != self.runtime["closure_pins"] or claims["remaining_wall_seconds"] <= 0:
                raise ValueError("fixture_draw_mismatch")
        elif operation == "draw_handoff":
            if claims["library_sha256"] != self.runtime["library_sha256"]:
                raise ValueError("fixture_library_mismatch")
        else:
            raise ValueError("fixture_unknown_operation")
        self.events.append(operation)
        return None


class TypedControlParentTests(unittest.TestCase):
    setUp = worker_fixtures.BoundWorkerTests.setUp
    prepare = worker_fixtures.BoundWorkerTests.prepare
    run_bound = worker_fixtures.BoundWorkerTests.run_bound

    @classmethod
    def setUpClass(cls):
        # Existing mandatory real-boundary fixture builds the own-checkout DSO.
        boundary_fixtures.ControlBoundaryTests.setUpClass()
        cls.runtime = copy.deepcopy(boundary_fixtures.ControlBoundaryTests.runtime)

    def configured(self, method="compressed_real_rows", *, byte_budget=None, verifier=None):
        executor = FixtureExecutor(self.runtime)
        source = Path(boundary.__file__).resolve().parent
        control = PilotControlRuntime(trusted_runtime=self.runtime,
            compiler_source_pins={n: boundary.digest(boundary.regular_bytes(source / n)) for n in COMPILER_FILES},
            admission_manifest=GRANT, verify_operation=executor if verifier is None else verifier)
        if byte_budget is None:
            measured = sum(len(boundary.regular_bytes(p)) for p in self.runtime["closure_pins"])
            byte_budget = measured + (1 << 20)  # distinct signed fixture budget, not a waiver
            self.assertLessEqual(byte_budget, 1 << 30)
        prepared = self.prepare(comparator=method, control_runtime=control, byte_budget=byte_budget,
                                learner_seed=ORIGINAL_STREAM_HEX)
        executor.cell_sha = prepared[3].sha256
        return prepared, control, executor

    def verify_success(self, method):
        prepared, control, executor = self.configured(method)
        observed = []
        original = FixtureLearner._predict_backend
        def predict(adapter, query, seed):
            self.assertIs(type(query), FeatureBatch)
            self.assertFalse(hasattr(query, "targets"))
            self.assertNotIn("query_targets", vars(adapter))
            self.assertFalse(set(query.row_ids) & set(adapter._training.features.row_ids))
            observed.append(query.row_ids)
            return original(adapter, query, seed)
        with mock.patch.object(FixtureLearner, "_predict_backend", new=predict):
            receipt = self.run_bound(prepared)
        self.assertEqual(receipt["status"], "ok", receipt)
        self.assertEqual(len(receipt["stages"]), 8)
        self.assertEqual(receipt["journal"]["pending_stages"], 0)
        self.assertTrue(observed)
        self.assertEqual(control.processes, 2)
        self.assertEqual(executor.events.count("sample_draw"), 2)
        self.assertFalse(receipt["scientific_admission"])
        self.assertFalse(receipt["pipeline_complete"])
        self.assertNotIn("SECRET", json.dumps(receipt))
        worker, root = prepared[6], prepared[4]
        self.assertEqual(worker._execution.plan.native_seed.original_stream_hex, ORIGINAL_STREAM_HEX)
        self.assertNotEqual(control._binding.fit_seed_hex, ORIGINAL_STREAM_HEX)
        self.assertEqual(len(worker._execution.evidence.guard.graph.document()["aliases"]), 60)
        draw_pair = load_pair(root, receipt["materialized_draw_manifest_sha256"], artifact=root / "model.raw",
                              comparator=method, control_runtime=control)
        self.assertEqual([d.spec.rows for d in draw_pair], [16, 64])
        self.assertEqual([d.spec.seed_hex for d in draw_pair], ["0xfffffffffffffffe", "0xffffffffffffffff"])
        self.assertEqual(draw_pair[0].binding.input_sha256, prepared[3].input_sha256)
        self.assertEqual(draw_pair[0].binding.fit_csv_sha256, prepared[3].document()["csv_sha256"])
        self.assertNotEqual(draw_pair[0].binding.input_sha256, draw_pair[0].binding.fit_csv_sha256)
        self.assertEqual(draw_pair[0].binding.ordered_column_ids, ("c0001", "c0002"))
        self.assertNotIn("schema", json.loads(draw_pair[0].receipt))
        if method == "compressed_real_rows":
            # Source CSV is y,x; the actual framed generator and isolated output
            # must agree on x,y, not just a two-column width assertion.
            for draw in draw_pair:
                training = draw.pool().batch
                for features, target in zip(training.features.rows, training.targets):
                    self.assertAlmostEqual(target, 10 * features[0], places=9)
        # Fixture CSV is target-first; retained schema is feature-first. Width
        # alone cannot authorize replacement names or source/order provenance.
        for draw in draw_pair:
            self.assertEqual(draw.receipt, read_regular(root / (draw.spec.scale + ".control.stderr")))
            self.assertEqual(draw.raw, read_regular(root / (draw.spec.scale + ".control.stdout")))
        with StageJournal(root, prepared[0], create=False, read_only=True) as journal:
            begins = [r for r in journal.records if r.get("type") == "begin"
                      and r.get("data", {}).get("stage") == "sample"]
            self.assertEqual(len(begins), 1)
        return prepared, control, draw_pair

    def test_actual_compressed_control_eight_stages_and_metric_only_query_labels(self):
        self.verify_success("compressed_real_rows")

    def test_actual_copula_control_eight_stages_and_metric_only_query_labels(self):
        self.verify_success("gaussian_copula")

    def test_real_full_closure_cap_refuses_before_sample_or_learner(self):
        prepared, control, _ = self.configured(byte_budget=32768)
        with mock.patch.object(boundary, "sample_control") as sample:
            receipt = self.run_bound(prepared)
        sample.assert_not_called()
        self.assertEqual(receipt["stages"][-1]["stage"], "sample")
        self.assertEqual(receipt["stages"][-1]["status"], "over_budget")
        self.assertEqual(FixtureLearner.prepares, 0)
        self.assertEqual(control.processes, 0)
        charge = json.loads(read_regular(prepared[4] / "control-sampling-charge.json"))
        self.assertGreater(max(charge["per_draw_bytes"]), charge["byte_budget"])
        self.assertFalse(charge["admitted"])
        self.assertEqual(receipt["journal"]["pending_stages"], 0)

    def test_signed_runtime_change_refuses_prepare(self):
        prepared, control, _ = self.configured()
        control.runtime["library_sha256"] = "f" * 64
        with mock.patch("research.transfer.pilot_runner.invoke") as native:
            receipt = self.run_bound(prepared)
        native.assert_not_called()
        self.assertEqual(receipt["stages"][0]["status"], "failed")
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_sampler_uses_one_decreasing_wall_allowance(self):
        prepared, _, _ = self.configured()
        original, allowances = boundary.sample_control, []
        def sample(*args, **kwargs):
            allowances.append(kwargs["wall_seconds"])
            return original(*args, **kwargs)
        with mock.patch.object(boundary, "sample_control", side_effect=sample):
            receipt = self.run_bound(prepared)
        self.assertEqual(receipt["status"], "ok", receipt)
        self.assertEqual(len(allowances), 2)
        self.assertGreater(allowances[0], allowances[1])
        self.assertLess(allowances[0], prepared[3].document()["stage_wall_ms"]["sample"] / 1000)

    def test_no_control_authority_is_not_external_capsule_execution(self):
        prepared = self.prepare(comparator="compressed_real_rows")
        with mock.patch("research.transfer.pilot_runner.invoke") as native:
            with self.assertRaisesRegex(CampaignRunnerError, "pilot_v2_control_runtime_unavailable"):
                self.run_bound(prepared)
        native.assert_not_called()
        self.assertFalse((prepared[4] / "pilot-execution.json").exists())

    def test_typed_callback_refusal_is_terminal_and_redacted(self):
        def refuse(*args):
            raise RuntimeError("SECRET_PRIVATE_CONTROL_GRANT")
        prepared, _, _ = self.configured(verifier=refuse)
        receipt = self.run_bound(prepared)
        self.assertEqual(receipt["stages"][0]["status"], "failed")
        self.assertEqual(receipt["journal"]["pending_stages"], 0)
        self.assertEqual(FixtureLearner.prepares, 0)
        self.assertNotIn("SECRET", json.dumps(receipt))

    def test_unclassified_callback_fault_keeps_original_pending_reservation(self):
        class UnknownFailure(Exception):
            pass
        prepared, _, executor = self.configured()
        original = prepared[6].control_runtime.verify_operation
        def verify(operation, *args):
            if operation == "compile":
                raise UnknownFailure()
            return original(operation, *args)
        prepared[6].control_runtime.verify_operation = verify
        with self.assertRaises(UnknownFailure):
            self.run_bound(prepared)
        with StageJournal(prepared[4], prepared[0], create=False, read_only=True) as journal:
            self.assertTrue(journal.state[prepared[3].document()["cell_id"], "compile"]["pending"])
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_control_journal_finish_fault_cannot_claim_settlement(self):
        prepared, _, _ = self.configured()
        finish = StageJournal.finish
        def fault(journal, cell, stage, *args, **kwargs):
            if stage == "compile":
                raise RuntimeError("SECRET_FSYNC_FAULT")
            return finish(journal, cell, stage, *args, **kwargs)
        with mock.patch.object(StageJournal, "finish", new=fault):
            with self.assertRaises(RuntimeError):
                self.run_bound(prepared)
        with StageJournal(prepared[4], prepared[0], create=False, read_only=True) as journal:
            self.assertTrue(journal.state[prepared[3].document()["cell_id"], "compile"]["pending"])
        self.assertEqual(FixtureLearner.prepares, 0)


    def test_retained_draw_tamper_refuses_without_another_sample(self):
        prepared, control, draws = self.verify_success("compressed_real_rows")
        root = prepared[4]
        target = root / "materialized-draws/primary_n.raw"
        target.chmod(0o600)
        target.write_bytes(draws[0].raw[:-8])
        with mock.patch.object(boundary, "sample_control") as sample:
            with self.assertRaises(CampaignRunnerError):
                control.load_pair(root, control._manifest, artifact=root / "model.raw")
        sample.assert_not_called()

    def test_failed_sampler_exit_retains_raw_evidence_and_prevents_second_draw(self):
        prepared, control, _ = self.configured()
        with mock.patch.object(boundary, "bounded_child", return_value=(7, b"partial", b"control_boundary_refused\n")) as child:
            receipt = self.run_bound(prepared)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(receipt["stages"][-1]["stage"], "sample")
        self.assertEqual(receipt["stages"][-1]["status"], "failed")
        self.assertEqual(FixtureLearner.prepares, 0)
        self.assertEqual(read_regular(prepared[4] / "primary_n.control.stdout"), b"partial")
        self.assertEqual(json.loads(read_regular(prepared[4] / "primary_n.control.exit.json")), {"returncode": 7})
        self.assertFalse((prepared[4] / "control-sensitivity_4n").exists())

    def test_changed_captured_fit_bytes_stop_reload_before_sampling(self):
        prepared, control, executor = self.configured()
        original = control.verify_operation
        def mutate(operation, claims, evidence):
            result = original(operation, claims, evidence)
            if operation == "reload":
                path = control._paths["csv"]
                path.chmod(0o600)
                path.write_bytes(read_regular(path) + b"999,999\n")
            return result
        control.verify_operation = mutate
        with mock.patch.object(boundary, "sample_control") as sample:
            receipt = self.run_bound(prepared)
        sample.assert_not_called()
        self.assertEqual(receipt["stages"][-1]["stage"], "reload")
        self.assertEqual(receipt["stages"][-1]["status"], "failed")
        self.assertEqual(FixtureLearner.prepares, 0)

    def test_control_stage_requires_fresh_open_original_journal(self):
        prepared, control, _ = self.configured()
        with mock.patch("research.transfer.pilot_control_runtime.CompressedRowsGenerator.fit") as fit:
            outcome, payload, _ = control.stage("compile", None, prepared[3], {}, prepared[4])
        fit.assert_not_called()
        self.assertEqual(outcome["status"], "failed")
        self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
