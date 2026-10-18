"""Offline conformance fixtures, independent of vendor code or protected data."""
import ast
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import unittest
from unittest.mock import patch

from research.transfer.adapters.base import (
    AdapterAdmission, AdapterError, AdapterUnavailable, FeatureBatch,
    GeneratorAdapter, LearnerAdapter, ModelIdentity, PredictionResult,
    REFUSED_ENCODERS, ResourceLimits, ResourceUsage, SampleResult, TrainingBatch,
)
from research.transfer.adapters.auditors.challengers import validate_model_registry

ROOT = Path(__file__).resolve().parents[3]


def pin(text):
    return hashlib.sha256(text.encode()).hexdigest()


def identity(**changes):
    fields = dict(name="linear_fixture", mode="scratch", code_revision=pin("source"),
                  code_sha256=pin("source"), weights_sha256=None,
                  dependencies=(("python", "3.10.0", pin("runtime fixture")),),
                  licenses=(("code", "fixture_license", pin("license fixture")),),
                  preprocessing_sha256=pin("affine fit-only transform"),
                  recipe_sha256=pin("OLS"), output_aggregation_sha256=pin("point"),
                  environment_sha256=pin("offline fixture environment"),
                  precision="float64", context_cap=3, context_policy="all",
                  query_policy="batch", query_batch_size=16, order_sensitive=False,
                  determinism="deterministic", resources=ResourceLimits(60.0, 1024, 0))
    fields.update(changes)
    return ModelIdentity(**fields)


def registry(model, kind="learner"):
    return {"format": "dope-adapter-registry-v1", "kind": kind, "entries": [{
        "name": model.name, "status": "reviewed", "identity_sha256": model.sha256,
        "eligibility_sha256": pin("fixture eligibility"),
        "native_reference_sha256": pin("fixture reference"), "license_status": "compatible"}]}


def features(xs, prefix="train"):
    return FeatureBatch(tuple(pin(f"{prefix}-{i}") for i in range(len(xs))),
                        tuple((x,) for x in xs), pin("one numeric input"))


def training():
    return TrainingBatch(features((0., 1., 2.)), (10., 30., 50.), "development_train")


class AffineLearner(LearnerAdapter):
    """A local OLS recipe using fit-only target standardization."""
    def _prepare_backend(self, training, seed, mode):
        self.context = training
        self.mode_seen = mode
        self.center = sum(training.targets) / len(training.targets)
        self.scale = max(abs(y - self.center) for y in training.targets) or 1.
        xs = [row[0] for row in training.features.rows]
        ys = [(y - self.center) / self.scale for y in training.targets]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        self.slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx)**2 for x in xs)
        self.intercept = my - self.slope * mx
        return ResourceUsage(len(training.targets), 128, 0)

    def _predict_backend(self, query, seed):
        self.query_seen = query
        values = tuple(((self.intercept + self.slope * row[0]) * self.scale + self.center,)
                       for row in query.rows)
        return PredictionResult(query.row_ids, values, "point", len(self.context.targets),
                                128, 0, embedding_unavailable="unsupported")


class ConstantGenerator(GeneratorAdapter):
    def _fit_backend(self, training, seed, mode):
        self.context = training
        self.mode_seen = mode
        return ResourceUsage(len(training.targets), 128, 0)

    def _sample_backend(self, rows, seed):
        data = TrainingBatch(features((self.context.features.rows[0][0],) * rows, f"sample-{seed}"),
                             (self.context.targets[0],) * rows, "synthetic_train")
        return SampleResult(data, ResourceUsage(len(self.context.targets), 128, 0))


class FrequencyLearner(LearnerAdapter):
    def _prepare_backend(self, training, seed, mode):
        self.count = len(training.targets)
        self.probabilities = tuple(training.targets.count(c) / self.count for c in ("a", "b", "c"))
        return ResourceUsage(self.count, 128, 0)

    def _predict_backend(self, query, seed):
        return PredictionResult(query.row_ids, (self.probabilities,) * len(query.rows),
                                "probability", self.count, 128, 0, ("a", "b", "c"),
                                embedding_unavailable="not_requested")


def learner(model=None):
    model = model or identity()
    return AffineLearner(model, AdapterAdmission.from_registry(registry(model), model, "learner"))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.train = training()
        self.query = features((3., 4.), "query")
        self.adapter = learner()
        self.adapter.prepare(self.train, 7)

    def test_native_reference_parity_in_original_units_and_no_query_labels(self):
        # Independent closed-form native recipe: y = 10 + 20*x, not rank units.
        reference = tuple((10 + 20 * row[0],) for row in self.query.rows)
        result = self.adapter.predict(self.query, 8)
        self.assertEqual(result.values, reference)
        self.assertEqual(result.row_ids, self.query.row_ids)
        self.assertEqual(result.consumed_rows, 3)
        self.assertIs(self.adapter.query_seen, self.query)
        self.assertFalse(hasattr(self.adapter.query_seen, "targets"))

    def test_unavailable_embedding_preserves_valid_predictions_and_cache_key(self):
        before = self.adapter.cache_key(self.query.sha256, 8, 2)
        result = self.adapter.predict(self.query, 8)
        self.assertEqual(result.embedding_unavailable, "unsupported")
        self.assertIsNone(result.embedding)
        self.assertEqual(result.values, ((70.,), (90.,)))
        self.assertEqual(before, self.adapter.cache_key(self.query.sha256, 8, 2))
        with_embedding = replace(result, embedding=((1.,), (2.,)), embedding_unavailable=None)
        self.assertEqual(with_embedding.values, result.values)

    def test_context_cap_is_declared_and_consumed_count_is_exact(self):
        capped = learner(identity(context_cap=2, context_policy="head"))
        capped.prepare(self.train, 7)
        result = capped.predict(self.query, 8)
        self.assertEqual(result.consumed_rows, 2)
        self.assertEqual(capped.context.features.row_ids, self.train.features.row_ids[:2])
        strict = learner(identity(context_cap=2))
        with self.assertRaisesRegex(AdapterError, "context_budget_exceeded"):
            strict.prepare(self.train, 7)

    def test_cache_invalidates_every_model_pin_and_policy(self):
        original = self.adapter.cache_key(self.query.sha256, 8, 2)
        changes = dict(code_revision=pin("revision2"), code_sha256=pin("code2"),
                       dependencies=(("python", "3.10.1", pin("runtime2")),),
                       licenses=(("code", "fixture_license", pin("license2")),),
                       preprocessing_sha256=pin("preprocessing2"), recipe_sha256=pin("recipe2"),
                       output_aggregation_sha256=pin("aggregation2"), environment_sha256=pin("env2"),
                       precision="float32", context_cap=4, context_policy="head",
                       query_policy="transductive", query_batch_size=17,
                       order_sensitive=True, determinism="seeded_best_effort",
                       resources=ResourceLimits(61., 1024, 0))
        for field, value in changes.items():
            with self.subTest(field=field):
                other = learner(identity(**{field: value}))
                other.prepare(self.train, 7)
                self.assertNotEqual(original, other.cache_key(self.query.sha256, 8, 2))

    def test_checkpoint_change_invalidates_frozen_context_identity(self):
        keys = []
        for weights in (pin("weights1"), pin("weights2")):
            model = identity(mode="frozen_context", weights_sha256=weights,
                             licenses=identity().licenses + (("weights", "fixture_license", pin("weights license")),))
            port = learner(model)
            port.prepare(self.train, 7)
            keys.append(port.cache_key(self.query.sha256, 8, 2))
        self.assertNotEqual(*keys)

    def test_cache_binds_data_order_counts_and_fit_query_seeds(self):
        original = self.adapter.cache_key(self.query.sha256, 8, 2)
        for request, seed, count in ((self.query.sha256, 9, 2), (self.query.sha256, 8, 3),
                                     (features((4., 3.), "query").sha256, 8, 2)):
            self.assertNotEqual(original, self.adapter.cache_key(request, seed, count))
        for batch, seed in ((self.train, 9), (replace(self.train, targets=(11., 31., 51.)), 7),
                            (replace(self.train, features=features((1., 2., 3.))), 7)):
            port = learner()
            port.prepare(batch, seed)
            self.assertNotEqual(original, port.cache_key(self.query.sha256, 8, 2))
        capped = learner(identity(context_cap=2, context_policy="head"))
        capped.prepare(self.train, 7)
        first = capped.cache_key(self.query.sha256, 8, 2)
        capped.prepare(replace(self.train, targets=(10., 30., 999.)), 7)
        self.assertNotEqual(first, capped.cache_key(self.query.sha256, 8, 2))

    def test_query_batch_and_order_behavior_are_explicit(self):
        reversed_query = FeatureBatch(self.query.row_ids[::-1], self.query.rows[::-1], self.query.schema_sha256)
        self.assertEqual(self.adapter.predict(reversed_query, 8).values, ((90.,), (70.,)))
        rowwise = learner(identity(query_policy="rowwise", query_batch_size=1))
        rowwise.prepare(self.train, 7)
        with self.assertRaisesRegex(AdapterError, "query_budget_exceeded"):
            rowwise.predict(self.query, 8)
        for i in range(2):
            single = FeatureBatch(self.query.row_ids[i:i+1], self.query.rows[i:i+1], self.query.schema_sha256)
            self.assertEqual(rowwise.predict(single, 8).values, self.adapter.predict(self.query, 8).values[i:i+1])

    def test_query_labels_and_protected_training_roles_are_rejected(self):
        with self.assertRaises(TypeError):
            FeatureBatch(self.query.row_ids, self.query.rows, self.query.schema_sha256, targets=(70., 90.))
        for role in ("selection", "confirmation", "test"):
            with self.subTest(role=role), self.assertRaisesRegex(AdapterError, "protected_training_role"):
                replace(self.train, role=role)
        for query in (self.train, {"features": self.query, "targets": (70., 90.)}):
            with self.assertRaisesRegex(AdapterError, "query_features_required"):
                self.adapter.predict(query, 8)

    def test_schema_and_full_provided_context_overlap_fail_before_prediction(self):
        cases = (replace(self.query, schema_sha256=pin("other schema")), features((3., 4.), "train"))
        for query in cases:
            with self.assertRaises(AdapterError):
                self.adapter.predict(query, 8)
        capped = learner(identity(context_cap=2, context_policy="head"))
        capped.prepare(self.train, 7)
        excluded = FeatureBatch(self.train.features.row_ids[2:], ((3.,),), self.query.schema_sha256)
        with self.assertRaisesRegex(AdapterError, "query_context_overlap"):
            capped.predict(excluded, 8)

    def test_wrong_alignment_context_count_and_memory_budget_fail(self):
        correct = self.adapter.predict(self.query, 8)
        for bad in (replace(correct, row_ids=correct.row_ids[::-1]), replace(correct, consumed_rows=2),
                    replace(correct, memory_bytes=2048), replace(correct, gpu_bytes=1)):
            with patch.object(self.adapter, "_predict_backend", return_value=bad), self.assertRaises(AdapterError):
                self.adapter.predict(self.query, 8)

    def test_invalid_and_nonfinite_outputs_fail(self):
        correct = self.adapter.predict(self.query, 8)
        for values in (((float("nan"),), (90.,)), ((float("inf"),), (90.,)),
                       ((10**400,), (90.,)), ((1., 2.), (3., 4.))):
            with self.assertRaises(AdapterError):
                replace(correct, values=values)
        with self.assertRaisesRegex(AdapterError, "invalid_embedding_status"):
            replace(correct, embedding_unavailable="source-secret")

    def test_probability_class_order_includes_absent_training_levels(self):
        schema = pin("categorical input with missingness and unseen-level policy")
        labels = replace(self.train, targets=("a", "a", "b"),
                         features=replace(features(("red", "red", "blue")), schema_sha256=schema))
        model = identity(name="frequency_fixture", preprocessing_sha256=pin("frequency preprocessing"))
        port = FrequencyLearner(model, AdapterAdmission.from_registry(registry(model), model, "learner"))
        port.prepare(labels, 7)
        prediction = port.predict(replace(features((None, "unseen"), "query"), schema_sha256=schema), 8)
        self.assertEqual(prediction.class_order, ("a", "b", "c"))
        self.assertEqual(prediction.values, ((2/3, 1/3, 0.),) * 2)
        for values in (((-.1, .9, .2),) * 2, ((.1, .1, .1),) * 2):
            with self.assertRaisesRegex(AdapterError, "invalid_probability"):
                replace(prediction, values=values)

    def test_errors_and_representations_do_not_echo_private_values(self):
        with patch.object(self.adapter, "_predict_backend", side_effect=RuntimeError("secret value 12345")):
            with self.assertRaisesRegex(AdapterError, "^backend_failed$") as caught:
                self.adapter.predict(self.query, 8)
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertNotIn("12345", repr(TrainingBatch(features(("12345",)), ("12345",), "development_train")))
        with self.assertRaises(FrozenInstanceError):
            self.query.rows = ((99.,),)
        with self.assertRaises(AttributeError):
            self.adapter.identity = identity()

    def test_failed_preparation_invalidates_previous_state(self):
        with patch.object(self.adapter, "_prepare_backend", side_effect=RuntimeError("failure")):
            with self.assertRaisesRegex(AdapterError, "backend_failed"):
                self.adapter.prepare(self.train, 9)
        with self.assertRaisesRegex(AdapterError, "adapter_not_prepared"):
            self.adapter.predict(self.query, 8)

    def test_missing_offline_dependency_is_typed_and_redacts_path(self):
        cases = ((ModuleNotFoundError("private module"), "dependency_unavailable"),
                 (FileNotFoundError("private checkpoint path"), "dependency_unavailable"),
                 (AdapterUnavailable("private backend diagnostic"), "backend_unavailable"))
        for failure, code in cases:
            with patch.object(self.adapter, "_predict_backend", side_effect=failure):
                with self.assertRaisesRegex(AdapterUnavailable, f"^{code}$") as caught:
                    self.adapter.predict(self.query, 8)
                self.assertTrue(caught.exception.__suppress_context__)

    def test_backend_exit_cannot_terminate_caller_or_echo_source(self):
        cases = ((SystemExit, AdapterError, "backend_failed"),
                 (KeyboardInterrupt, KeyboardInterrupt, ""))
        for fault, outcome, diagnostic in cases:
            with self.subTest(fault=fault), patch.object(self.adapter, "_predict_backend",
                                                       side_effect=fault("secret source value")):
                with self.assertRaises(outcome) as caught:
                    self.adapter.predict(self.query, 8)
                self.assertEqual(str(caught.exception), diagnostic)
                self.assertTrue(caught.exception.__suppress_context__)

    def test_fit_resource_usage_and_failed_call_time_are_checked(self):
        for usage in (None, ResourceUsage(2, 128, 0), ResourceUsage(3, 2048, 0)):
            port = learner()
            with patch.object(port, "_prepare_backend", return_value=usage), self.assertRaises(AdapterError):
                port.prepare(self.train, 7)
            with self.assertRaisesRegex(AdapterError, "adapter_not_prepared"):
                port.predict(self.query, 8)
        port = learner(identity(resources=ResourceLimits(1., 1024, 0)))
        with patch("research.transfer.adapters.base.time.monotonic", side_effect=(0., .25)):
            port.prepare(self.train, 7)
        with patch("research.transfer.adapters.base.time.monotonic", side_effect=(1., 2.)), \
             patch.object(port, "_predict_backend", side_effect=RuntimeError("backend failure")):
            with self.assertRaisesRegex(AdapterError, "backend_failed"):
                port.predict(self.query, 8)
        with self.assertRaisesRegex(AdapterError, "wall_budget_exceeded"):
            port.predict(self.query, 8)

    def test_distribution_vectors_and_invalid_output_shapes(self):
        result = PredictionResult(self.query.row_ids, ((69., 70., 71.), (89., 90., 91.)),
                                  "distribution", 3, 128, 0, embedding_unavailable="not_requested")
        with patch.object(self.adapter, "_predict_backend", return_value=result):
            self.assertEqual(self.adapter.predict(self.query, 8).values, result.values)
        with self.assertRaisesRegex(AdapterError, "invalid_classes"):
            replace(result, class_order=("a",))
        with self.assertRaises(AdapterError):
            FeatureBatch(self.query.row_ids, ((1.,), (2., 3.)), self.query.schema_sha256)

    def test_wall_limit_is_cumulative_for_preparation_and_prediction(self):
        port = learner(identity(resources=ResourceLimits(1., 1024, 0)))
        with patch("research.transfer.adapters.base.time.monotonic", side_effect=(0., .75)):
            port.prepare(self.train, 7)
        with patch("research.transfer.adapters.base.time.monotonic", side_effect=(1., 1.5)):
            with self.assertRaisesRegex(AdapterError, "wall_budget_exceeded"):
                port.predict(self.query, 8)

    def test_generator_training_sample_counts_schema_and_seed_identity(self):
        model = identity(name="constant_generator")
        port = ConstantGenerator(model, AdapterAdmission.from_registry(registry(model, "generator"), model, "generator"))
        port.fit(self.train, 7)
        sample = port.sample(4, 9)
        self.assertEqual(len(sample.data.targets), 4)
        self.assertEqual(sample.data.role, "synthetic_train")
        self.assertEqual(sample.usage.consumed_rows, 3)
        self.assertEqual(sample, port.sample(4, 9))
        for bad in (replace(sample.data, targets=sample.data.targets[:3], features=features((0.,) * 3, "sample")),
                    replace(sample.data, role="development_train"),
                    replace(sample.data, features=replace(sample.data.features, schema_sha256=pin("wrong")))):
            bad = SampleResult(bad, sample.usage)
            with patch.object(port, "_sample_backend", return_value=bad), self.assertRaises(AdapterError):
                port.sample(4, 9)
        with patch.object(port, "_sample_backend", return_value=replace(sample, usage=ResourceUsage(3, 2048, 0))):
            with self.assertRaisesRegex(AdapterError, "resource_budget_exceeded"):
                port.sample(4, 9)
        with self.assertRaisesRegex(AdapterError, "protected_training_role"):
            port.fit(replace(self.train, role="synthetic_train"), 7)
        with self.assertRaisesRegex(AdapterError, "adapter_not_prepared"):
            port.sample(4, 9)
        port.fit(self.train, 7)
        other = learner(model)
        other.prepare(self.train, 7)
        self.assertNotEqual(port.cache_key(self.query.sha256, 8, 2), other.cache_key(self.query.sha256, 8, 2))

    def test_offline_fixture_has_no_network_process_or_product_dependency(self):
        with patch.object(socket, "socket", side_effect=AssertionError("network")), \
             patch.object(subprocess, "Popen", side_effect=AssertionError("process")):
            port = learner()
            port.prepare(self.train, 7)
            self.assertEqual(port.predict(self.query, 8).values, ((70.,), (90.,)))
        tree = ast.parse((ROOT / "research/transfer/adapters/base.py").read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        self.assertEqual(imports, {"abc", "dataclasses", "math", "re", "time", "research.transfer.evidence"})


class AdmissionTests(unittest.TestCase):
    def test_all_four_modes_are_explicit_and_checkpoint_requirements_hold(self):
        for mode in ("scratch", "frozen_context", "fine_tuning", "pretraining"):
            fields = {"mode": mode}
            if mode in {"frozen_context", "fine_tuning"}:
                fields.update(weights_sha256=pin("weights"), licenses=identity().licenses +
                              (("weights", "fixture_license", pin("weights license")),))
            port = learner(identity(**fields))
            port.prepare(training(), 7)
            self.assertEqual(port.mode_seen, mode)
        for mode in ("frozen_context", "fine_tuning"):
            with self.assertRaisesRegex(AdapterError, "missing_weights"):
                identity(mode=mode)

    def test_registry_unknowns_and_incompatible_licenses_are_typed(self):
        model = identity()
        for field, value, code in (("status", "unresolved", "model_unavailable"),
                                   ("status", "unavailable", "model_unavailable"),
                                   ("license_status", "incompatible", "license_unavailable"),
                                   ("identity_sha256", pin("different checkpoint"), "identity_unavailable")):
            policy = registry(model)
            policy["entries"][0][field] = value
            with self.assertRaisesRegex(AdapterUnavailable, code):
                AdapterAdmission.from_registry(policy, model, "learner")

    def test_draft_vendor_registries_cannot_admit_official_models(self):
        for kind, filename in (("learner", "models.json"), ("generator", "generators.json")):
            policy = json.loads((ROOT / "research/transfer/registries" / filename).read_bytes())
            for entry in policy["entries"]:
                self.assertEqual(entry["status"], "unresolved")
                self.assertIsNone(entry["identity_sha256"])
                model = identity(name=entry["name"])
                with self.assertRaises(AdapterUnavailable):
                    AdapterAdmission.from_registry(policy, model, kind)

    def test_frozen_panel_roles_match_schema_and_reject_drift(self):
        panel = json.loads((ROOT / "schemas/panel-v1.schema.json").read_bytes())["examples"][0]
        registry = json.loads((ROOT / "research/transfer/registries/models.json").read_bytes())
        checkpoints = {entry["id"]: entry for entry in panel["checkpoints"]}
        for candidate in registry["entries"]:
            frozen = checkpoints[candidate["name"]]
            self.assertEqual(candidate["panel_edition"], panel["edition"])
            self.assertEqual(candidate["roles_by_workload"], frozen["roles_by_workload"])
        self.assertTrue(all(set(entry) == {
            "name", "status", "identity_sha256", "eligibility_sha256",
            "native_reference_sha256", "license_status"}
            for entry in validate_model_registry(registry)["entries"]))
        changed = copy.deepcopy(registry["entries"][0]["roles_by_workload"])
        changed["regression"]["compiler_teacher"] = not changed["regression"]["compiler_teacher"]
        with self.assertRaises(AssertionError):
            self.assertEqual(changed, checkpoints[registry["entries"][0]["name"]]["roles_by_workload"])
        broken = copy.deepcopy(registry)
        broken["entries"][0]["roles_by_workload"]["regression"]["compiler_teacher"] = "drift"
        with self.assertRaises(AdapterError):
            validate_model_registry(broken)

    def test_refused_encoders_preserve_historical_set_even_with_reviewed_entry(self):
        tree = ast.parse((ROOT / "research/benchmark/representation.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REFUSED_ENCODERS" for t in node.targets):
                historical = ast.literal_eval(node.value)
                break
        self.assertEqual(REFUSED_ENCODERS, frozenset(historical))
        for name in REFUSED_ENCODERS:
            model = identity(name=name)
            with self.assertRaisesRegex(AdapterUnavailable, "refused_encoder"):
                AdapterAdmission.from_registry(registry(model), model, "learner")
            admission = AdapterAdmission(model.sha256, pin("eligibility"), pin("reference"), "learner")
            with self.assertRaisesRegex(AdapterUnavailable, "refused_encoder"):
                AffineLearner(model, admission)

    def test_incomplete_pins_unknown_fields_and_duplicate_names_fail_closed(self):
        for changes in ({"dependencies": ()}, {"licenses": ()}, {"code_sha256": "unknown"},
                        {"dependencies": (("python", ">=3.10", pin("runtime")),)},
                        {"context_cap": True}, {"mode": []}, {"precision": {}},
                        {"weights_sha256": pin("scratch initialization")}):
            with self.assertRaises(AdapterError):
                identity(**changes)
        model = identity()
        policy = registry(model)
        for broken in ({**policy, "query_labels": []},
                       {**policy, "entries": policy["entries"] * 2}):
            with self.assertRaisesRegex(AdapterError, "invalid_registry"):
                AdapterAdmission.from_registry(broken, model, "learner")
        broken = copy.deepcopy(policy)
        broken["entries"][0]["eligibility_sha256"] = None
        with self.assertRaises(AdapterError):
            AdapterAdmission.from_registry(broken, model, "learner")


if __name__ == "__main__":
    unittest.main()
