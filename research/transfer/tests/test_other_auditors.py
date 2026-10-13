"""Offline interface regressions; fixture factories do not qualify vendor weights."""
import ast
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.transfer.adapters.base import (
    AdapterAdmission, AdapterError, AdapterUnavailable, FeatureBatch, ResourceLimits,
    ResourceUsage, TrainingBatch,
)
from research.transfer.adapters.auditors.conventional import (
    ConventionalAdapter, ConventionalLearnerRecipe, OfflineArtifacts, OfflineRuntime,
    file_sha256, select_conventional_recipe,
)
from research.transfer.adapters.auditors.mitra import MitraAdapter, MitraRecipe
from research.transfer.adapters.auditors.tabicl import CODE_REVISION, TabICLAdapter, TabICLRecipe
from research.transfer.tests.test_adapters import identity, pin, registry


ROOT = Path(__file__).resolve().parents[3]
TEMP = ROOT / "target/vnext/auditor-fixtures"


def batch(xs=(0., 1., 2.), targets=(10., 30., 50.), prefix="context", role="development_train"):
    view = FeatureBatch(tuple(pin(f"{prefix}-{i}") for i in range(len(xs))),
                        tuple((x,) for x in xs), pin("numeric-port-schema"))
    return TrainingBatch(view, targets, role)


def query(xs=(3., 4.)):
    return batch(xs, (0.,) * len(xs), "query").features


class NativeFixture:
    """A recording public sklearn interface; not an official reference model."""
    def __init__(self, values=None, classes=(0, 1)):
        self.values = values
        self.classes_ = np.asarray(classes)
        self.calls = []

    def fit(self, X, y, **kwargs):
        self.calls.append(("fit", np.asarray(X).copy(), np.asarray(y).copy(), kwargs))
        self.n_samples_in_ = len(y)
        return self

    def predict(self, X, **kwargs):
        self.calls.append(("predict", np.asarray(X).copy(), kwargs))
        return np.asarray(self.values if self.values is not None else [70., 90.])

    def predict_proba(self, X):
        self.calls.append(("probability", np.asarray(X).copy()))
        return np.asarray(self.values if self.values is not None else [[.7, .3], [.2, .8]])


class PortFixture:
    """Private synthetic artifacts plus explicit test-only executor observations."""
    def __init__(self, testcase, recipe, name, revision, mode="frozen_context", factory=None):
        TEMP.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=TEMP)
        testcase.addCleanup(directory.cleanup)
        root = Path(directory.name)
        def artifact(name, payload):
            path = root / name
            path.write_bytes(payload)
            return path
        code = artifact("code", b"fixture code, not vendor source")
        dependency = artifact("dependency", b"fixture dependency, not installed closure")
        license_code = artifact("license-code", b"fixture code license")
        weights = artifact("model.safetensors", b"fixture weights, not pretrained") if mode != "scratch" else None
        license_weights = artifact("license-weights", b"fixture weights license") if weights else None
        self.config = artifact("config.json", b"{}")
        licenses = (("code", "fixture_license", file_sha256(license_code)),)
        if weights:
            licenses += (("weights", "fixture_license", file_sha256(license_weights)),)
        self.identity = identity(name=name, mode=mode, code_revision=revision, code_sha256=file_sha256(code),
                                 weights_sha256=file_sha256(weights) if weights else None,
                                 dependencies=(("fixture-package", "1.0", file_sha256(dependency)),),
                                 licenses=licenses, precision=recipe.precision, recipe_sha256=recipe.sha256,
                                 preprocessing_sha256=recipe.preprocessing_sha256,
                                 output_aggregation_sha256=recipe.output_sha256, order_sensitive=True,
                                 query_policy="transductive" if mode == "fine_tuning" else "batch",
                                 resources=ResourceLimits(60., 1024 * 1024, 0))
        ancillary = ((self.config, recipe.config_sha256),) if isinstance(recipe, MitraRecipe) else ()
        self.artifacts = OfflineArtifacts(code, (dependency,), (license_code, license_weights) if weights else
                                          (license_code,), weights, ancillary)
        self.events, self.kwargs = [], []
        self.native = NativeFixture()
        def create(**kwargs):
            self.kwargs.append(kwargs)
            return factory(**kwargs) if factory else self.native
        def observe(event):
            self.events.append(event)
            # Test-only usage with an explicitly known fixture context count.
            return ResourceUsage(self.rows, 128, 0)
        self.rows = 3
        self.runtime = OfflineRuntime(self.artifacts, observe, create, lambda _: "1.0")

    def adapter(self, cls, recipe, model=None, runtime=None):
        model = model or self.identity
        admission = AdapterAdmission.from_registry(registry(model), model, "learner")
        return cls(model, admission, recipe, runtime or self.runtime)


class OtherAuditorTests(unittest.TestCase):
    def test_tabicl_passes_only_local_checkpoint_and_declared_recipe(self):
        recipe = TabICLRecipe("regression", estimators=4, estimator_batch_size=2, kv_cache=True)
        fixture = PortFixture(self, recipe, "tabicl_v2", CODE_REVISION)
        port = fixture.adapter(TabICLAdapter, recipe)
        port.prepare(batch(), 17)
        out = port.predict(query(), 23)
        kwargs = fixture.kwargs[0]
        self.assertFalse(kwargs["allow_auto_download"])
        self.assertEqual(kwargs["model_path"], str(fixture.artifacts.weights))
        self.assertEqual((kwargs["n_estimators"], kwargs["batch_size"], kwargs["random_state"]), (4, 2, 17))
        self.assertFalse(kwargs["use_amp"])
        self.assertEqual(kwargs["n_jobs"], 1)
        np.testing.assert_array_equal(fixture.native.calls[0][2], [10., 30., 50.])
        np.testing.assert_array_equal(fixture.native.calls[1][1], [[3.], [4.]])
        self.assertEqual(out.values, ((70.,), (90.,)))
        self.assertEqual(fixture.events, ["prepare", "predict"])

    def test_classification_reorders_native_codes_and_refuses_absent_classes(self):
        recipe = TabICLRecipe("classification", class_order=("positive", "negative"))
        fixture = PortFixture(self, recipe, "tabicl_v2", CODE_REVISION)
        fixture.native = NativeFixture([[.2, .8], [.9, .1]], classes=(1, 0))
        port = fixture.adapter(TabICLAdapter, recipe)
        port.prepare(batch(targets=("positive", "negative", "positive")), 7)
        result = port.predict(query(), 8)
        self.assertEqual(result.class_order, recipe.class_order)
        self.assertEqual(result.values, ((.8, .2), (.1, .9)))
        for targets in (("positive",) * 3, ("positive", "negative", "source_secret")):
            with self.assertRaises(AdapterError) as caught:
                port.prepare(batch(targets=targets), 7)
            self.assertNotIn("source_secret", str(caught.exception))
            with self.assertRaisesRegex(AdapterError, "adapter_not_prepared"):
                port.predict(query(), 8)

    def test_context_caps_queries_and_observed_consumption(self):
        recipe = TabICLRecipe("regression")
        fixture = PortFixture(self, recipe, "tabicl_v2", CODE_REVISION)
        model = replace(fixture.identity, context_cap=2, context_policy="head", query_batch_size=2)
        fixture.rows = 2
        port = fixture.adapter(TabICLAdapter, recipe, model)
        port.prepare(batch(), 1)
        self.assertEqual(len(fixture.native.calls[0][2]), 2)
        self.assertEqual(port.predict(query(), 2).consumed_rows, 2)
        with self.assertRaisesRegex(AdapterError, "query_budget_exceeded"):
            port.predict(query((3., 4., 5.)), 2)
        with self.assertRaisesRegex(AdapterError, "query_features_required"):
            port.predict(batch(), 2)
        fixture.rows = 1
        with self.assertRaises(AdapterError):
            port.prepare(batch(), 1)

    def test_artifacts_dependencies_and_licenses_fail_before_vendor_loading(self):
        recipe = TabICLRecipe("regression")
        fixture = PortFixture(self, recipe, "tabicl_v2", CODE_REVISION)
        port = fixture.adapter(TabICLAdapter, recipe)
        fixture.artifacts.weights.write_bytes(b"source_secret")
        with self.assertRaises(AdapterUnavailable) as caught:
            port.prepare(batch(), 7)
        self.assertNotIn("source_secret", str(caught.exception))
        self.assertEqual(fixture.kwargs, [])
        fixture.artifacts.weights.unlink()
        with self.assertRaises(AdapterUnavailable):
            port.prepare(batch(), 7)
        decision = registry(fixture.identity)
        decision["entries"][0]["license_status"] = "incompatible"
        with self.assertRaisesRegex(AdapterUnavailable, "license_unavailable"):
            AdapterAdmission.from_registry(decision, fixture.identity, "learner")
        fresh = PortFixture(self, recipe, "tabicl_v2", CODE_REVISION)
        bad = replace(fresh.runtime, version_reader=lambda _: "2.0")
        with self.assertRaises(AdapterUnavailable):
            fresh.adapter(TabICLAdapter, recipe, runtime=bad).prepare(batch(), 7)
        self.assertEqual(fresh.kwargs, [])

    def test_symlink_artifacts_are_unavailable(self):
        recipe = TabICLRecipe("regression")
        fixture = PortFixture(self, recipe, "tabicl_v2", CODE_REVISION)
        link = fixture.config.with_name("link")
        link.symlink_to(fixture.artifacts.code)
        with self.assertRaisesRegex(AdapterUnavailable, "artifact_unavailable"):
            file_sha256(link)

    def test_native_errors_and_resource_mismatch_are_redacted(self):
        recipe = TabICLRecipe("regression")
        fixture = PortFixture(self, recipe, "tabicl_v2", CODE_REVISION)
        port = fixture.adapter(TabICLAdapter, recipe)
        with patch.object(fixture.native, "fit", side_effect=RuntimeError("source_secret")):
            with self.assertRaisesRegex(AdapterError, "^backend_failed$"):
                port.prepare(batch(), 7)
        port.prepare(batch(), 7)
        fixture.native.values = [float("nan"), 90.]
        with self.assertRaises(AdapterError):
            port.predict(query(), 8)
        bad = replace(fixture.runtime, observer=lambda _: ResourceUsage(3, 2 * 1024 * 1024, 0))
        with self.assertRaisesRegex(AdapterError, "resource_budget_exceeded"):
            fixture.adapter(TabICLAdapter, recipe, runtime=bad).prepare(batch(), 7)

    def test_recipe_immutable_and_cache_keys_bind_seed_cap_batch_ensemble_and_weights(self):
        recipe = TabICLRecipe("regression")
        fixture = PortFixture(self, recipe, "tabicl_v2", CODE_REVISION)
        port = fixture.adapter(TabICLAdapter, recipe)
        port.prepare(batch(), 7)
        old = port.cache_key(query().sha256, 8, 2)
        with self.assertRaises(FrozenInstanceError):
            recipe.estimators = 3
        with self.assertRaises(AttributeError):
            port.recipe = replace(recipe, estimators=3)
        for changed in (replace(recipe, estimators=3), replace(recipe, estimator_batch_size=2),
                        replace(recipe, norm_methods=("none",)), replace(recipe, kv_cache=True)):
            other = PortFixture(self, changed, "tabicl_v2", CODE_REVISION)
            adapted = other.adapter(TabICLAdapter, changed)
            adapted.prepare(batch(), 7)
            self.assertNotEqual(old, adapted.cache_key(query().sha256, 8, 2))
        for changed in (replace(fixture.identity, query_batch_size=8), replace(fixture.identity, context_cap=4),
                        replace(fixture.identity, weights_sha256=pin("different weights"))):
            self.assertNotEqual(changed.sha256, fixture.identity.sha256)
        self.assertNotEqual(old, port.cache_key(query().sha256, 9, 2))

    def test_mitra_classifier_v2_is_frozen_with_explicit_seed_and_raw_context(self):
        recipe = MitraRecipe("classification", pin("{}"), class_order=("b", "a"))
        fixture = PortFixture(self, recipe, "mitra_v2", recipe.code_revision)
        port = fixture.adapter(MitraAdapter, recipe)
        port.prepare(batch(targets=("b", "a", "b")), 11)
        kwargs = fixture.kwargs[0]
        self.assertFalse(kwargs["fine_tune"])
        self.assertEqual((kwargs["fine_tune_steps"], kwargs["seed"]), (0, 11))
        self.assertEqual(kwargs["hf_model"], str(fixture.artifacts.weights.parent))
        self.assertEqual(port.predict(query(), 12).values, ((.7, .3), (.2, .8)))
        self.assertEqual(port.attempts, ())

    def test_mitra_regression_charges_prediction_time_fits_and_refuses_frozen_mode(self):
        recipe = MitraRecipe("regression", pin("{}"))
        fixture = PortFixture(self, recipe, "mitra_v2", recipe.code_revision, mode="fine_tuning")
        port = fixture.adapter(MitraAdapter, recipe)
        port.prepare(batch(), 11)
        self.assertTrue(fixture.kwargs[0]["in_process"])
        self.assertEqual(fixture.kwargs[0]["num_bag_folds"], 8)
        self.assertEqual(port.predict(query(), 12).values, ((70.,), (90.,)))
        port.predict(query(), 13)
        self.assertEqual(len(port.attempts), 2)
        self.assertTrue(all(a["requested_child_fits"] == 8 for a in port.attempts))
        self.assertEqual(fixture.events, ["prepare", "predict", "predict"])
        frozen = replace(fixture.identity, mode="frozen_context", query_policy="batch")
        with self.assertRaisesRegex(AdapterUnavailable, "frozen_v2_regression_unavailable"):
            fixture.adapter(MitraAdapter, recipe, frozen)
        with patch.object(fixture.native, "predict", side_effect=RuntimeError("source_secret")):
            with self.assertRaisesRegex(AdapterError, "backend_failed"):
                port.predict(query(), 14)
        self.assertEqual(len(port.attempts), 3)
        self.assertEqual(port.attempts[-1]["status"], "failed")
        port.attempts[0]["status"] = "mutated"
        self.assertEqual(port.attempts[0]["status"], "returned")

    def test_native_linear_control_matches_independent_formula(self):
        recipe = ConventionalLearnerRecipe("linear", "regression", ())
        fixture = PortFixture(self, recipe, "linear", pin("native sklearn fixture"), mode="scratch")
        # Exercise the real public sklearn API, with fixture admission clearly
        # distinct from any vendor checkpoint/scientific qualification.
        dependency = fixture.artifacts.dependencies[0]
        model = replace(fixture.identity, dependencies=(("scikit-learn", "1.7.2", file_sha256(dependency)),))
        runtime = replace(fixture.runtime, factory=None, version_reader=lambda _: "1.7.2")
        port = fixture.adapter(ConventionalAdapter, recipe, model, runtime)
        port.prepare(batch(), 7)
        np.testing.assert_allclose(port.predict(query(), 8).values, [[70.], [90.]], atol=1e-10)

    def test_native_catboost_keeps_raw_units_and_independent_families_run_offline(self):
        recipes = (
            ConventionalLearnerRecipe("catboost", "regression", (("iterations", 12), ("depth", 2)),
                                      pin("fixture development selection"), precision="float32"),
            ConventionalLearnerRecipe("histogram_boosting", "regression", (("max_iter", 12), ("min_samples_leaf", 1)),
                                      pin("fixture development selection")),
            ConventionalLearnerRecipe("mlp", "classification", (("hidden_layer_sizes", (4,)), ("solver", "lbfgs"), ("max_iter", 300)),
                                      pin("fixture development selection"), class_order=("b", "a")),
        )
        for recipe in recipes:
            with self.subTest(family=recipe.family):
                package, exact = ("catboost", "1.2.10") if recipe.family == "catboost" else ("scikit-learn", "1.7.2")
                fixture = PortFixture(self, recipe, recipe.model_name, pin("native conventional fixture"), mode="scratch")
                model = replace(fixture.identity, dependencies=((package, exact, file_sha256(fixture.artifacts.dependencies[0])),))
                runtime = replace(fixture.runtime, factory=None, version_reader=lambda _: exact)
                port = fixture.adapter(ConventionalAdapter, recipe, model, runtime)
                targets = (1000., 3000., 5000.) if recipe.task == "regression" else ("b", "a", "b")
                port.prepare(batch(targets=targets), 7)
                result = port.predict(query(), 8)
                if recipe.task == "regression":
                    self.assertTrue(all(1000. <= row[0] <= 5000. for row in result.values))
                else:
                    self.assertEqual(result.kind, "probability")
                    self.assertEqual(result.class_order, ("b", "a"))

    def test_native_primary_dependency_is_required_and_no_backend_fallback_occurs(self):
        recipe = TabICLRecipe("regression")
        fixture = PortFixture(self, recipe, "tabicl_v2", CODE_REVISION)
        runtime = replace(fixture.runtime, factory=None)
        with self.assertRaises(AdapterUnavailable):
            fixture.adapter(TabICLAdapter, recipe, runtime=runtime).prepare(batch(), 7)
        self.assertEqual(fixture.kwargs, [])

    def test_mitra_refuses_undeclared_native_environment_overrides(self):
        recipe = MitraRecipe("regression", pin("{}"))
        fixture = PortFixture(self, recipe, "mitra_v2", recipe.code_revision, mode="fine_tuning")
        with patch.dict("os.environ", {"MITRA_FT_STEPS": "1"}):
            with self.assertRaises(AdapterUnavailable):
                fixture.adapter(MitraAdapter, recipe).prepare(batch(), 7)
        self.assertEqual(fixture.kwargs, [])

    def test_tuning_rejects_overlap_mutable_parameters_and_reserved_resource_overrides(self):
        with self.assertRaisesRegex(AdapterError, "invalid_parameters"):
            ConventionalLearnerRecipe("linear", "regression", (("bad", ({"mutable": 1},)),))
        with self.assertRaisesRegex(AdapterError, "reserved_parameter"):
            ConventionalLearnerRecipe("linear", "regression", (("n_jobs", 8),))
        candidate = ConventionalLearnerRecipe("linear", "regression", ())
        with self.assertRaisesRegex(AdapterError, "development_overlap"):
            select_conventional_recipe((candidate,), batch(), batch(), lambda *_: None, ResourceLimits(60., 1024, 0))

    def test_tuning_contains_backend_failures_and_enforces_budget(self):
        candidate = ConventionalLearnerRecipe("linear", "regression", ())
        validation = batch((3., 4.), (70., 90.), "validation")
        def failure(*_):
            raise RuntimeError("source_secret")
        with self.assertRaisesRegex(AdapterError, "^backend_failed$"):
            select_conventional_recipe((candidate,), batch(), validation, failure, ResourceLimits(60., 1024, 0))
        with self.assertRaisesRegex(AdapterError, "resource_budget_exceeded"):
            select_conventional_recipe((candidate,), batch(), validation,
                                       lambda *_: (1., ResourceUsage(3, 1025, 0)), ResourceLimits(60., 1024, 0))

    def test_development_search_retains_all_candidates_and_freezes_selected_recipe(self):
        candidates = tuple(ConventionalLearnerRecipe("catboost", "regression", (("depth", d),), precision="float32") for d in (4, 6, 8))
        train, validation = batch(), batch((3., 4.), (70., 90.), "validation")
        called = []
        def fit_and_score(recipe, training, validation):
            called.append(recipe)
            return {4: 2., 6: 1., 8: 3.}[dict(recipe.parameters)["depth"]], ResourceUsage(3, 128, 0)
        selected, receipt = select_conventional_recipe(candidates, train, validation, fit_and_score, ResourceLimits(60., 1024, 0))
        self.assertEqual(dict(selected.parameters)["depth"], 6)
        self.assertEqual(len(receipt["attempts"]), 3)
        self.assertIsNotNone(selected.selection_sha256)
        fixture = PortFixture(self, selected, "catboost", pin("catboost fixture"), mode="scratch")
        real = fixture.adapter(ConventionalAdapter, selected)
        real.prepare(train, 5)
        synthetic = fixture.adapter(ConventionalAdapter, selected)
        synthetic.prepare(replace(train, role="synthetic_train"), 5)
        self.assertEqual(fixture.kwargs[0], fixture.kwargs[1])
        self.assertEqual(real.identity.resources, synthetic.identity.resources)
        with self.assertRaisesRegex(AdapterError, "development_only_selection"):
            select_conventional_recipe(candidates, replace(train, role="synthetic_train"), validation,
                                       fit_and_score, ResourceLimits(60., 1024, 0))
        with self.assertRaisesRegex(AdapterError, "development_selection_required"):
            fixture.adapter(ConventionalAdapter, candidates[0])

    def test_locks_keep_related_lineages_and_unmeasured_gates_visible(self):
        lock = json.loads((ROOT / "research/transfer/locks/other-auditors.json").read_text())
        self.assertEqual(lock["status"], "unqualified")
        self.assertFalse(lock["mitra"]["checkpoints"]["regression"]["local_verification"])
        self.assertEqual(lock["mitra"]["family_group"], "mitra_architecture")
        self.assertIn("official_checkpoint_minimal_example_parity", lock["unmeasured_gates"])
        self.assertIsNone(lock["exact_dependency_closure"])

    def test_port_source_has_no_network_subprocess_workstation_or_product_imports(self):
        for path in (ROOT / "research/transfer/adapters/auditors").glob("*.py"):
            source = path.read_text()
            self.assertNotIn("/home/", source)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    modules = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    modules = []
                self.assertFalse(any(m.split('.')[0] in {"subprocess", "socket", "urllib", "requests",
                                                         "sqlite3", "dope_kernel"} for m in modules))


if __name__ == "__main__":
    unittest.main()
