"""Contract fixtures only: no author model fit, vendor installation or dataset access."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from research.transfer.adapters.generators import structured as S


PIN = "a" * 64


def spec(name="arf_default", regime="native_recipe", variant="native"):
    profile = next(s.value["profile"] for s in S.load_specs() if s.value["profile"]["name"] == name)
    profile["status"] = "reviewed"
    if profile["source"]["revision"] is None:
        profile["source"]["revision"] = "c" * 40
    for key in profile:
        if key.endswith("_sha256"):
            profile[key] = hashlib.sha256(key.encode()).hexdigest()
    return S.GeneratorSpec(profile, regime, variant)


def view(rows=((1.0, 10.0), (3.0, 20.0))):
    return S.FitView(rows, PIN, "b" * 64, "c" * 64)


class SpyBackend:
    """A recording fixture, not an ARF or other author implementation."""
    def __init__(self, identity, consumed=2):
        self.identity_sha256, self.consumed = identity, consumed
        self.calls, self.level = [], 0.0

    def fit(self, fitting, seed, profile):
        self.calls.append(("fit", fitting, seed, profile))
        self.consumed = len(fitting.rows)
        self.level = sum(row[-1] for row in fitting.rows) / self.consumed
        return S.BackendUsage(self.consumed, 64)

    def sample(self, rows, seed):
        self.calls.append(("sample", rows, seed))
        return S.SampleRows(tuple((float(i + seed), self.level) for i in range(rows)), S.BackendUsage(self.consumed, 64))

    def export(self):
        self.calls.append(("export",))
        return (S.StateComponent("weights.json", "weights", json.dumps({"level": self.level}).encode()),
                S.StateComponent("inverse.json", "inverse_transform", b'{"offset":0,"scale":1}'))

    def reload(self, state):
        self.calls.append(("reload",))
        loaded = SpyBackend(self.identity_sha256, state.metadata["consumed_rows"])
        loaded.level = json.loads(next(c.data for c in state.components if c.name == "weights.json"))["level"]
        return loaded


class StructuredTests(unittest.TestCase):
    def engine(self, name="arf_default", **limits):
        declaration = spec(name)
        backend = SpyBackend(declaration.sha256)
        cls = {"arf": S.ARFAdapter, "forest": S.ForestFlowAdapter, "tabpc": S.TabPCAdapter,
               "compact_density": S.CompactDensityControls}[declaration.value["profile"]["family"]]
        return cls(declaration, backend, **limits), backend

    def test_all_nine_author_and_control_profiles_are_unresolved(self):
        declared = S.load_specs()
        self.assertEqual({s.value["profile"]["name"] for s in declared}, {
            "arf_default", "arf_native", "forest_flow", "forest_diffusion", "tabpc", "gaussian_copula",
            "chow_liu", "independent_marginals", "cart_synthpop"})
        for profile in declared:
            self.assertEqual(profile.value["profile"]["status"], "unresolved")
            with self.assertRaises(S.StructuredUnavailable):
                profile.admit()

    def test_identity_separates_modes_regimes_variants_and_every_pin(self):
        profiles = [spec(name) for name in ("arf_default", "arf_native", "forest_flow", "forest_diffusion")]
        self.assertEqual(len({p.sha256 for p in profiles}), 4)
        original = spec()
        changes = [spec(regime="resource_matched"), spec(regime="common_transfer"), spec(variant="pruned"), spec(variant="quantized")]
        for field in original.value["profile"]:
            if field.endswith("_sha256"):
                altered = original.value["profile"]
                altered[field] = "d" * 64
                changes.append(S.GeneratorSpec(altered))
        self.assertTrue(all(p.sha256 != original.sha256 for p in changes))

    def test_profile_is_a_snapshot_and_public_values_do_not_mutate_it(self):
        profile = spec().value["profile"]
        snapshot = S.GeneratorSpec(profile)
        before = snapshot.sha256
        profile["source"]["revision"] = "e" * 40
        public = snapshot.value
        public["profile"]["history"].append("changed")
        self.assertEqual(snapshot.sha256, before)

    def test_incomplete_admission_and_wrong_family_refuse_before_backend_calls(self):
        profile = spec().value["profile"]
        for field in ("native_reference_sha256", "export_reload_sha256", "license_sha256", "dependencies_sha256"):
            changed = copy.deepcopy(profile)
            changed[field] = None
            declaration = S.GeneratorSpec(changed)
            backend = SpyBackend(declaration.sha256)
            with self.assertRaises(S.StructuredUnavailable):
                S.ARFAdapter(declaration, backend)
            self.assertEqual(backend.calls, [])
        with self.assertRaisesRegex(S.StructuredError, "wrong_generator_family"):
            S.TabPCAdapter(spec(), SpyBackend(spec().sha256))

    def test_exact_types_and_closed_registry_versions_refuse(self):
        document = json.loads(S.LOCK.read_bytes())
        variants = [{**document, "version": True}, {**document, "version": 1.0},
                    {**document, "extra": "private_value"}, {**document, "profiles": document["profiles"] * 2}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            for value in variants:
                path.write_text(json.dumps(value))
                with self.assertRaises(S.StructuredError):
                    S.load_specs(path)
            path.write_text('{"version":1,"version":1}')
            with self.assertRaises(S.StructuredUnavailable):
                S.load_specs(path)
        for role in ([], True):
            with self.assertRaises(S.StructuredError):
                S.StateComponent("weights.bin", role, b"x")
        for field in ("family", "status", "native_direction"):
            profile = spec().value["profile"]
            profile[field] = []
            with self.assertRaises(S.StructuredError):
                S.GeneratorSpec(profile)

    def test_only_development_numeric_fit_view_reaches_backend(self):
        engine, backend = self.engine()
        for role in ("query", "final_test", "synthetic_train"):
            with self.assertRaisesRegex(S.StructuredError, "protected_training_role"):
                S.FitView(((1.0, 2.0),), PIN, PIN, PIN, role)
        with self.assertRaisesRegex(S.StructuredError, "development_view_required"):
            engine.fit({"query_labels": (12345.0,)}, 1)
        for rows in (((True, 1.0),), ((float("nan"), 1.0),), (("private_value", 1.0),)):
            with self.assertRaises(S.StructuredError):
                view(rows)
        self.assertEqual(backend.calls, [])
        fitting = view()
        engine.fit(fitting, 17)
        self.assertIs(backend.calls[0][1], fitting)
        self.assertEqual(backend.calls[0][2], 17)

    def test_mutated_role_refuses_before_fit_and_failed_refit_invalidates_old_state(self):
        engine, backend = self.engine()
        fitting = view()
        object.__setattr__(fitting, "role", "final_test")
        with self.assertRaises(S.StructuredError):
            engine.fit(fitting, 1)
        self.assertEqual(backend.calls, [])
        engine.fit(view(), 1)
        with patch.object(backend, "fit", side_effect=RuntimeError("private_value_999")):
            with self.assertRaisesRegex(S.StructuredError, "backend_failed"):
                engine.fit(view(), 2)
        with self.assertRaisesRegex(S.StructuredError, "generator_not_fitted"):
            engine.sample(1, 1)

    def test_backend_errors_are_redacted_and_unavailable_is_typed(self):
        engine, backend = self.engine()
        for error, code, kind in [(RuntimeError("private_value_999"), "backend_failed", S.StructuredError),
                                  (SystemExit("private_header"), "backend_failed", S.StructuredError),
                                  (ModuleNotFoundError("secret_path"), "backend_unavailable", S.StructuredUnavailable)]:
            with patch.object(backend, "fit", side_effect=error), self.assertRaises(kind) as caught:
                engine.fit(view(), 1)
            self.assertEqual(str(caught.exception), code)
            self.assertTrue(caught.exception.__suppress_context__)

    def test_resource_limits_counts_and_actual_wall_clock_are_enforced(self):
        engine, backend = self.engine(memory_bytes=100, gpu_bytes=0, wall_seconds=2)
        for usage in (S.BackendUsage(1, 64), S.BackendUsage(2, 101), S.BackendUsage(2, 64, 1)):
            with patch.object(backend, "fit", return_value=usage), self.assertRaises(S.StructuredError):
                engine.fit(view(), 1)
        with patch.object(S.time, "monotonic", side_effect=[10.0, 13.0]), self.assertRaisesRegex(S.StructuredError, "wall_budget_exceeded"):
            engine.fit(view(), 1)
        for count in (True, 1.0, -1):
            with self.assertRaises(S.StructuredError):
                S.BackendUsage(count, 1)

    def test_backend_identity_checked_before_and_after_calls(self):
        engine, backend = self.engine()
        backend.identity_sha256 = "d" * 64
        with self.assertRaisesRegex(S.StructuredError, "backend_identity_mismatch"):
            engine.fit(view(), 1)
        self.assertEqual(backend.calls, [])
        backend.identity_sha256 = engine.spec.sha256
        def swapped(*_args):
            backend.identity_sha256 = "e" * 64
            return S.BackendUsage(2, 64)
        with patch.object(backend, "fit", swapped), self.assertRaises(S.StructuredError):
            engine.fit(view(), 1)

    def test_backend_identity_property_errors_are_contained(self):
        declaration = spec()
        class BadBackend:
            @property
            def identity_sha256(self):
                raise RuntimeError("private_header_and_value")
        with self.assertRaises(S.StructuredError) as caught:
            S.ARFAdapter(declaration, BadBackend())
        self.assertEqual(str(caught.exception), "backend_failed")
        self.assertTrue(caught.exception.__suppress_context__)

    def test_mutated_resource_record_and_source_row_component_refuse(self):
        engine, backend = self.engine()
        usage = S.BackendUsage(2, 64)
        object.__setattr__(usage, "memory_bytes", -1)
        with patch.object(backend, "fit", return_value=usage), self.assertRaises(S.StructuredError):
            engine.fit(view(), 1)
        engine.fit(view(), 1)
        component = S.StateComponent("weights.bin", "weights", b"private_row")
        object.__setattr__(component, "role", "source_rows")
        with patch.object(backend, "export", return_value=(component,)), self.assertRaises(S.StructuredError):
            engine.export()

    def test_sample_counts_width_and_original_values_are_preserved(self):
        engine, backend = self.engine()
        engine.fit(view(), 1)
        self.assertEqual(engine.sample(2, 999).rows, ((999.0, 15.0), (1000.0, 15.0)))
        for sample in (S.SampleRows(((1.0, 15.0),), S.BackendUsage(2, 64)),
                       S.SampleRows(((1.0, 2.0, 3.0),) * 2, S.BackendUsage(2, 64)),
                       S.SampleRows(((1.0, 15.0),) * 2, S.BackendUsage(1, 64))):
            with patch.object(backend, "sample", return_value=sample), self.assertRaises(S.StructuredError):
                engine.sample(2, 1)
        for rows, seed in [(True, 1), (1, True), (0, 1), (1, -1)]:
            with self.assertRaises(S.StructuredError):
                engine.sample(rows, seed)

    def test_cache_invalidation_includes_fit_seed_rows_schema_cohort_and_auditors(self):
        engine, _backend = self.engine()
        engine.fit(view(), 1)
        old = engine.cache_key(2, 17)
        self.assertNotEqual(old, engine.cache_key(3, 17))
        self.assertNotEqual(old, engine.cache_key(2, 18))
        for fitting, seed in [(view(), 2), (view(((1.0, 100.0), (3.0, 200.0))), 1),
                              (S.FitView(view().rows, "d" * 64, "b" * 64, "c" * 64), 1),
                              (S.FitView(view().rows, PIN, "d" * 64, "c" * 64), 1),
                              (S.FitView(view().rows, PIN, "b" * 64, "d" * 64), 1)]:
            engine.fit(fitting, seed)
            self.assertNotEqual(old, engine.cache_key(2, 17))

    def test_complete_charged_state_and_reload_need_no_fit_view(self):
        engine, backend = self.engine()
        engine.fit(view(), 1)
        expected = engine.sample(2, 13).rows
        exported = engine.export()
        body = exported.to_bytes()
        self.assertEqual(exported.charged_bytes, len(body))
        self.assertGreater(exported.charged_bytes, sum(len(c.data) for c in exported.components))
        self.assertTrue(exported.fits_tier(len(body)))
        self.assertFalse(exported.fits_tier(len(body) - 1))
        loaded = engine.reload(S.ExportState.from_bytes(body))
        self.assertIsNot(loaded.backend, backend)
        self.assertEqual(loaded.backend.calls, [])
        self.assertEqual(loaded.sample(2, 13).rows, expected)
        self.assertFalse(hasattr(loaded, "rows"))
        self.assertFalse(any(call[0] == "fit" for call in loaded.backend.calls))

    def test_rows_paths_and_duplicate_or_unlisted_export_components_refuse(self):
        for name in ("../rows", "/rows", "a/b", "a\\b", "a\x00b"):
            with self.assertRaises(S.StructuredError):
                S.StateComponent(name, "weights", b"x")
        for role in ("source_rows", "train_dataframe", "query_labels"):
            with self.assertRaisesRegex(S.StructuredError, "unadmitted_component_role"):
                S.StateComponent("rows.csv", role, b"private_value")
        engine, _backend = self.engine()
        engine.fit(view(), 1)
        exported = engine.export()
        with self.assertRaises(S.StructuredError):
            S.ExportState(exported.metadata, exported.components * 2)
        with self.assertRaises(S.StructuredError):
            S.ExportState.from_bytes(exported.to_bytes() + b"private_sidecar")

    def test_state_corruption_versions_unknown_fields_and_bool_lengths_refuse(self):
        engine, _backend = self.engine()
        engine.fit(view(), 1)
        body = engine.export().to_bytes()
        with self.assertRaisesRegex(S.StructuredError, "component_digest_mismatch"):
            S.ExportState.from_bytes(body[:-1] + bytes([body[-1] ^ 1]))
        length = struct.unpack(">Q", body[:8])[0]
        header = json.loads(body[8:8 + length])
        for mutation in ("version", "field", "length", "pin", "role"):
            changed = copy.deepcopy(header)
            if mutation == "version": changed["version"] = True
            if mutation == "field": changed["metadata"]["private_header"] = "private_value"
            if mutation == "length": changed["files"][0]["bytes"] = True
            if mutation == "pin": changed["files"][0]["sha256"] = "short"
            if mutation == "role": changed["files"][0]["role"] = "source_rows"
            raw = json.dumps(changed).encode()
            with self.assertRaises(S.StructuredError) as caught:
                S.ExportState.from_bytes(struct.pack(">Q", len(raw)) + raw + body[8 + length:])
            self.assertNotIn("private", str(caught.exception))

    def test_reload_identity_and_reusing_fitted_backend_refuse(self):
        engine, backend = self.engine()
        engine.fit(view(), 1)
        exported = engine.export()
        changed = exported.metadata
        changed["spec_sha256"] = "d" * 64
        with self.assertRaisesRegex(S.StructuredError, "state_identity_mismatch"):
            engine.reload(S.ExportState(changed, exported.components))
        with patch.object(backend, "reload", return_value=backend), self.assertRaisesRegex(S.StructuredError, "reload_reused_fitting_backend"):
            engine.reload(exported)

    def test_large_circuit_state_is_valid_and_visible_but_tier_infeasible(self):
        engine, backend = self.engine("tabpc")
        engine.fit(view(), 1)
        with patch.object(backend, "export", return_value=(S.StateComponent("weights.bin", "weights", b"x" * 20_000),)):
            exported = engine.export()
        self.assertFalse(exported.fits_tier(10_240))
        self.assertGreater(exported.charged_bytes, 20_000)

    def test_native_losses_compare_only_within_exact_method_regime_and_view(self):
        low = S.NativeObjective.for_view(spec(), view(), -2.0)
        high = S.NativeObjective.for_view(spec(), view(), -1.0)
        self.assertEqual(S.compare_native(high, low), 1.0)
        for other in [S.NativeObjective.for_view(spec("tabpc"), view(), -1.0),
                      S.NativeObjective.for_view(spec(regime="resource_matched"), view(), -1.0),
                      S.NativeObjective.for_view(spec(), view(((1.0, 11.0), (3.0, 21.0))), -1.0)]:
            with self.assertRaisesRegex(S.StructuredError, "native_objectives_not_comparable"):
                S.compare_native(low, other)

    def test_full_cohort_and_auditor_matrix_retains_failure_and_adverse_history(self):
        schedule = S.CohortSchedule((PIN, "b" * 64), (11, 23), (101,), (1, 4), ("catboost", "kumo_tabular"))
        profiles = (spec(), spec("forest_flow"))
        original = schedule.ledger(profiles, {})
        key = (profiles[1].sha256, PIN, 11, 101, 1, "catboost")
        retained = schedule.ledger(profiles, {key: "timeout"})
        self.assertEqual(len(original), 32)
        self.assertEqual(len(retained), len(original))
        self.assertEqual(sum(row["status"] == "timeout" for row in retained), 1)
        self.assertEqual(sum(row["status"] == "unresolved" for row in retained), 31)
        self.assertTrue(all(row["history"] for row in retained if row["spec_sha256"] == profiles[1].sha256))
        self.assertEqual({row["schedule_sha256"] for row in retained}, {schedule.sha256})
        for status in ("unsupported", "oom", "invalid", "byte_infeasible"):
            self.assertEqual(len(schedule.ledger(profiles, {key: status})), 32)
        with self.assertRaisesRegex(S.StructuredError, "unscheduled_outcome"):
            schedule.ledger(profiles, {("private_source",): "complete"})

    def test_ambiguous_schedule_and_execution_status_refuse(self):
        for seeds in ((True,), (1.0,), (11, 11)):
            with self.assertRaises(S.StructuredError):
                S.CohortSchedule((PIN,), seeds, (101,), (1,), ("catboost",))
        schedule = S.CohortSchedule((PIN,), (11,), (101,), (1,), ("catboost",))
        declaration = spec()
        key = (declaration.sha256, PIN, 11, 101, 1, "catboost")
        for status in (True, "dropped", "private_value"):
            with self.assertRaises(S.StructuredError):
                schedule.ledger((declaration,), {key: status})

    def test_pr09_bridge_is_lazy_and_missing_dependency_is_explicit(self):
        engine, _backend = self.engine()
        with patch.object(S.importlib, "import_module", side_effect=ModuleNotFoundError("private_path")):
            with self.assertRaisesRegex(S.StructuredUnavailable, "pr09_unavailable"):
                engine.as_pr09(None, None, PIN, PIN)

    def test_merged_pr09_bridge_fits_and_samples_original_units(self):
        from research.transfer.adapters import base as B
        from research.transfer.tests.test_adapters import identity, registry

        profile = spec().value["profile"]
        model = identity(name=profile["name"], code_revision=profile["source"]["revision"],
                         code_sha256=profile["code_sha256"],
                         preprocessing_sha256=profile["preprocessing_sha256"],
                         recipe_sha256=profile["recipe_sha256"])
        admission = B.AdapterAdmission.from_registry(registry(model, "generator"), model, "generator")
        profile["dependencies_sha256"] = S.digest([list(pin) for pin in model.dependencies])
        profile["license_sha256"] = S.digest([list(pin) for pin in model.licenses])
        profile["native_reference_sha256"] = admission.native_reference_sha256
        declaration = S.GeneratorSpec(profile)
        backend = SpyBackend(declaration.sha256)
        engine = S.ARFAdapter(declaration, backend)
        bridge = engine.as_pr09(model, admission, "b" * 64, "c" * 64)
        fitting = B.TrainingBatch(B.FeatureBatch(("1" * 64, "2" * 64), ((1.0,), (3.0,)), PIN),
                                  (10.0, 20.0), "development_train")
        bridge.fit(fitting, 7)
        result = bridge.sample(2, 13)
        self.assertEqual(result.data.targets, (15.0, 15.0))
        self.assertEqual(result.data.role, "synthetic_train")
        self.assertEqual(result.data.features.schema_sha256, PIN)
        self.assertEqual(result.usage.consumed_rows, 2)
        self.assertFalse(set(result.data.features.row_ids).intersection(fitting.features.row_ids))
        self.assertEqual(backend.calls[0][1].rows, ((1.0, 10.0), (3.0, 20.0)))
        with self.assertRaises(B.AdapterError):
            bridge.fit(B.TrainingBatch(fitting.features, fitting.targets, "synthetic_train"), 7)


if __name__ == "__main__":
    unittest.main()
