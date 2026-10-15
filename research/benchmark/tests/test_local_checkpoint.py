"""Synthetic offline hook fixtures only; no model assets, network or vendor run."""
from dataclasses import replace
import importlib.util
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from research.benchmark.local_checkpoint import LocalCheckpointHook
from research.transfer.adapters.base import AdapterAdmission, FeatureBatch, ResourceLimits
from research.transfer.tests.test_adapters import AffineLearner, identity, pin, registry


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("checkpoint_public_demo", ROOT / "examples/public-regression/run.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class QueryBoundaryFixture(AffineLearner):
    def _predict_backend(self, query, seed):
        assert type(query) is FeatureBatch and not hasattr(query, "targets")
        return super()._predict_backend(query, seed)


class SlowFixture(QueryBoundaryFixture):
    marker = None

    def _prepare_backend(self, training, seed, mode):
        time.sleep(.3)
        self.marker.write_text("late fixture action")
        return super()._prepare_backend(training, seed, mode)


class PrivateFailureFixture(QueryBoundaryFixture):
    def _prepare_backend(self, training, seed, mode):
        raise ValueError("private checkpoint path / private source header")


def hook(cls=QueryBoundaryFixture, **changes):
    model = identity(name="local_checkpoint_fixture", mode="frozen_context", weights_sha256=pin("synthetic weights"),
                     licenses=identity().licenses + (("weights", "fixture_license", pin("synthetic license")),),
                     resources=ResourceLimits(2., 4 * 1024**3, 0), **changes)
    admission = AdapterAdmission.from_registry(registry(model), model, "learner")
    adapter = cls(model, admission)
    return LocalCheckpointHook(adapter, admission, model.sha256, admission.eligibility_sha256,
                               admission.native_reference_sha256, fixture_only=True)


class LocalCheckpointTests(unittest.TestCase):
    def call(self, candidate, targets=(10., 30., 50.), arm="real"):
        return candidate.predict(((0.,), (1.,), (2.,)), targets, ((3.,), (4.,)), 1729, arm, pin("schema"))

    def test_default_skips_optional_assets_imports_and_execution(self):
        with patch("research.benchmark.local_checkpoint.multiprocessing.get_context", side_effect=AssertionError("unexpected worker")):
            self.assertEqual(demo.optional_checkpoint_arm(None, (), (), (), (), 1729, "real", Path("absent"), []),
                             {"state": "not_run_no_admitted_local_checkpoint"})

    def test_optional_receipt_is_explicit_successor_and_default_v1_stays_fixed(self):
        original = {"format": "dope-public-airfoil-demo-v1", "optional_foundation_model": "not_run_no_admitted_local_checkpoint"}
        self.assertEqual(demo.attach_optional_checkpoint(original.copy(), None), original)
        result = demo.attach_optional_checkpoint(original.copy(), {"real": {"state": "interface_fixture_completed"}})
        self.assertEqual(result["format"], "dope-public-airfoil-demo-local-checkpoint-v1")
        self.assertIsInstance(result["optional_foundation_model"], str)
        self.assertFalse(result["optional_checkpoint_v1"]["scientific_admission"])
        report = demo.render_report(result)
        self.assertIn("Optional local checkpoint", report)
        self.assertIn("interface_fixture_completed", report)

    def test_expected_pins_and_nonfixture_offline_port_refuse_before_work(self):
        candidate = hook()
        with self.assertRaisesRegex(ValueError, "public_checkpoint_pin_mismatch"):
            replace(candidate, expected_eligibility_sha256=pin("different custody decision"))
        with self.assertRaisesRegex(ValueError, "public_checkpoint_offline_native_port_required"):
            replace(candidate, fixture_only=False)

    def test_query_labels_are_absent_and_fixture_output_does_not_qualify(self):
        with patch("research.benchmark.local_checkpoint.multiprocessing.get_context", wraps=multiprocessing.get_context) as start:
            predictions, receipt = self.call(hook())
            start.assert_called_once_with("spawn")
        self.assertEqual(predictions, (70., 90.))
        self.assertEqual(receipt["state"], "interface_fixture_completed")
        self.assertEqual(receipt["provided_rows"], 3)
        self.assertEqual(receipt["consumed_rows"], 3)
        self.assertFalse(receipt["scientific_admission"])
        self.assertFalse(receipt["release_qualified"])
        self.assertEqual(receipt["gpu_bytes"], 0)
        self.assertGreater(receipt["memory_bytes"], 128)
        self.assertGreater(receipt["elapsed_seconds"], 0)
        self.assertEqual(receipt["elapsed_seconds"], receipt["cumulative_hook_seconds"])

    def test_declared_head_policy_records_provided_and_consumed_context(self):
        candidate = hook(context_cap=3, context_policy="head")
        predictions, receipt = candidate.predict(((0.,), (1.,), (2.,), (999.,)), (10., 30., 50., -99999.),
                                                 ((3.,), (4.,)), 1729, "synthetic_4n", pin("schema"))
        self.assertEqual(predictions, (70., 90.))
        self.assertEqual((receipt["provided_rows"], receipt["consumed_rows"]), (4, 3))
        self.assertEqual(receipt["context_policy"], "head")

    def test_request_caps_and_missing_pins_refuse_before_child_allocation(self):
        candidate = hook()
        with patch("research.benchmark.local_checkpoint.multiprocessing.get_context", side_effect=AssertionError("allocated before preflight")):
            with self.assertRaisesRegex(ValueError, "public_checkpoint_input_cap_exceeded"):
                candidate.predict(((0.,),), (10.,), ((3.,),) * 4097, 1729, "real", pin("schema"))
            with self.assertRaisesRegex(ValueError, "public_checkpoint_context_cap_exceeded"):
                candidate.predict(((0.,),) * 4, (10.,) * 4, ((3.,),), 1729, "real", pin("schema"))
            with self.assertRaisesRegex(ValueError, "public_checkpoint_input_invalid"):
                self.call(candidate, (10., float("nan"), 50.))
            with self.assertRaisesRegex(ValueError, "public_checkpoint_input_invalid"):
                candidate.predict(None, None, None, 1729, "real", pin("schema"))
        with self.assertRaisesRegex(ValueError, "public_checkpoint_expected_pin_invalid"):
            replace(candidate, expected_native_reference_sha256="not-a-pin")

    def test_timeout_kills_owned_worker_and_shared_budget_does_not_reset(self):
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="checkpoint-timeout-") as directory:
            candidate = hook(SlowFixture)
            candidate.adapter.marker = Path(directory) / "late"
            model = replace(candidate.adapter.identity, resources=ResourceLimits(.05, 4 * 1024**3, 0))
            admission = AdapterAdmission.from_registry(registry(model), model, "learner")
            adapter = SlowFixture(model, admission)
            adapter.marker = candidate.adapter.marker
            candidate = LocalCheckpointHook(adapter, admission, model.sha256, admission.eligibility_sha256,
                                            admission.native_reference_sha256, fixture_only=True)
            with self.assertRaisesRegex(ValueError, "public_checkpoint_wall_budget_exhausted"):
                self.call(candidate)
            time.sleep(.35)
            self.assertFalse(adapter.marker.exists())
            with patch("research.benchmark.local_checkpoint.multiprocessing.get_context", side_effect=AssertionError("reset budget")):
                with self.assertRaisesRegex(ValueError, "public_checkpoint_wall_budget_exhausted"):
                    self.call(candidate)

    def test_optional_failure_retains_failed_cost_without_private_text(self):
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="checkpoint-failure-") as directory:
            output = Path(directory)
            attempts = []
            result = demo.optional_checkpoint_arm(hook(PrivateFailureFixture), ((0.,), (1.,), (2.,)),
                                                  (10., 30., 50.), ((3.,), (4.,)), (70., 90.),
                                                  1729, "real", output, attempts)
            self.assertEqual(result["state"], "unavailable_or_failed")
            self.assertFalse(result["scientific_admission"])
            records = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
            self.assertEqual([v["state"] for v in records], ["started", "failed"])
            self.assertGreater(records[1]["elapsed_seconds"], 0)
            self.assertNotIn("private checkpoint", json.dumps(result) + (output / "attempts.jsonl").read_text())

    def test_optional_scoring_uses_same_predictions_when_query_truth_changes(self):
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="checkpoint-scoring-") as directory:
            output = Path(directory)
            def score(truth):
                return demo.optional_checkpoint_arm(hook(), ((0.,), (1.,), (2.,)), (10., 30., 50.),
                                                    ((3.,), (4.,)), truth, 1729, "real", output, [])
            a, b = score((70., 90.)), score((-99999., 99999.))
            self.assertEqual(a["prediction_sha256"], b["prediction_sha256"])
            self.assertEqual(a["loss"]["mse_db_squared"], 0.)
            self.assertGreater(b["loss"]["mse_db_squared"], 0.)
            self.assertEqual(a["state"], "interface_fixture_completed")


if __name__ == "__main__":
    unittest.main()
