"""Structured generator contracts for admitted offline author backends.

The custodian supplies and verifies the backend, dependencies and admission
evidence. This module validates scheduling, input roles, native-objective scope
and exported byte accounting; it grants no scientific or release authority.
Official backend and source-denial qualification remain independent gates.
"""
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import struct
import time

from research.transfer.evidence import EvidenceError, digest, parse_json


LOCK = Path(__file__).resolve().parents[2] / "locks" / "structured-generators.json"
REGIMES = {"native_recipe", "resource_matched", "common_transfer"}
STATUSES = {"complete", "unsupported", "timeout", "oom", "invalid", "byte_infeasible", "unresolved"}
ROLES = {"weights", "preprocessor", "inverse_transform", "dictionary", "decoder", "recipe"}
MAX_STATE = 64 << 20
MAX_MANIFEST = 1 << 20


class StructuredError(ValueError):
    """Stable rule names; backend messages and source values are contained."""


class StructuredUnavailable(StructuredError):
    """Required admission, reference evidence or offline runtime is unavailable."""


def _require(condition, code):
    if not condition:
        raise StructuredError(code)


def _count(value, minimum=0):
    _require(type(value) is int and minimum <= value <= 2**53 - 1, "invalid_count")


def _hash(value):
    _require(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value), "invalid_pin")


def _finite(value):
    return (type(value) is float and math.isfinite(value)) or (type(value) is int and abs(value) <= 2**53 - 1)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class GeneratorSpec:
    """Immutable snapshot of a custodian's profile, not a receipt verifier."""
    def __init__(self, profile, regime="native_recipe", variant="native"):
        fields = {"name", "family", "source", "mode", "native_objective", "native_direction", "status",
                  "code_sha256", "dependencies_sha256", "license_sha256", "native_reference_sha256",
                  "export_reload_sha256", "preprocessing_sha256", "recipe_sha256", "history"}
        _require(type(profile) is dict and set(profile) == fields, "invalid_profile")
        _require(type(profile["name"]) is str and re.fullmatch(r"[a-z0-9_]+", profile["name"]), "invalid_profile")
        _require(type(profile["family"]) is str and profile["family"] in {"arf", "forest", "tabpc", "compact_density"}, "invalid_profile")
        _require(type(profile["status"]) is str and profile["status"] in {"unresolved", "reviewed", "unavailable"}, "invalid_profile")
        _require(type(regime) is str and regime in REGIMES and type(variant) is str
                 and variant in {"native", "pruned", "quantized"}, "invalid_regime")
        source = profile["source"]
        _require(type(source) is dict and set(source) == {"url", "revision", "scope"}, "invalid_source")
        _require(type(source["url"]) is str and source["url"].startswith("https://"), "invalid_source")
        _require(type(source["scope"]) is str and bool(source["scope"]), "invalid_source")
        _require(source["revision"] is None or (type(source["revision"]) is str and
                 re.fullmatch(r"[a-f0-9]{40}", source["revision"])), "invalid_source")
        _require(type(profile["mode"]) is str and bool(profile["mode"]), "invalid_profile")
        _require(type(profile["native_objective"]) is str and bool(profile["native_objective"]), "invalid_profile")
        _require(type(profile["native_direction"]) is str and profile["native_direction"] in {"minimize", "maximize"}, "invalid_profile")
        _require(type(profile["history"]) is list and all(type(x) is str and bool(x) for x in profile["history"]), "invalid_profile")
        for field in fields:
            if field.endswith("_sha256") and profile[field] is not None:
                _hash(profile[field])
        try:
            self._bytes = _json({"profile": profile, "regime": regime, "variant": variant})
        except (ValueError, TypeError):
            raise StructuredError("invalid_profile") from None
        _require(len(self._bytes) <= MAX_MANIFEST, "profile_too_large")

    @property
    def value(self):
        return json.loads(self._bytes)

    @property
    def sha256(self):
        return hashlib.sha256(self._bytes).hexdigest()

    def admit(self):
        profile = self.value["profile"]
        pins = [v for k, v in profile.items() if k.endswith("_sha256")]
        if profile["status"] != "reviewed" or profile["source"]["revision"] is None or any(v is None for v in pins):
            raise StructuredUnavailable("profile_unavailable")


def load_specs(path=LOCK):
    try:
        document = parse_json(Path(path).read_bytes())
    except (OSError, EvidenceError):
        raise StructuredUnavailable("registry_unavailable") from None
    _require(type(document) is dict and set(document) == {"format", "version", "profiles"}, "invalid_registry")
    _require(document["format"] == "dope-structured-generators" and type(document["version"]) is int
             and document["version"] == 1 and type(document["profiles"]) is list, "invalid_registry")
    specs, seen = [], set()
    for profile in document["profiles"]:
        spec = GeneratorSpec(profile)
        _require(profile["name"] not in seen, "duplicate_profile")
        seen.add(profile["name"])
        specs.append(spec)
    _require(bool(specs), "invalid_registry")
    return tuple(specs)


@dataclass(frozen=True, repr=False)
class FitView:
    rows: tuple  # Numeric joint rows, target last, in declared original units.
    schema_sha256: str
    cohort_sha256: str
    auditors_sha256: str
    role: str = "development_train"

    def __post_init__(self):
        for pin in (self.schema_sha256, self.cohort_sha256, self.auditors_sha256):
            _hash(pin)
        _require(type(self.role) is str and self.role == "development_train", "protected_training_role")
        _rows(self.rows)

    @property
    def sha256(self):
        return digest({"rows": [list(row) for row in self.rows], "schema": self.schema_sha256, "cohort": self.cohort_sha256,
                       "auditors": self.auditors_sha256, "role": self.role})


def _rows(rows, width=None, count=None):
    _require(type(rows) is tuple and bool(rows) and type(rows[0]) is tuple, "invalid_rows")
    columns = len(rows[0])
    _require(2 <= columns <= 256 and len(rows) * columns <= 1_000_000, "input_budget_exceeded")
    _require(width is None or columns == width, "schema_mismatch")
    _require(count is None or len(rows) == count, "sample_count_mismatch")
    _require(all(type(row) is tuple and len(row) == columns and all(_finite(v) for v in row) for row in rows), "invalid_rows")


@dataclass(frozen=True)
class BackendUsage:
    consumed_rows: int
    memory_bytes: int
    gpu_bytes: int = 0

    def __post_init__(self):
        _count(self.consumed_rows, 1)
        _count(self.memory_bytes)
        _count(self.gpu_bytes)


@dataclass(frozen=True, repr=False)
class SampleRows:
    rows: tuple
    usage: BackendUsage

    def __post_init__(self):
        _rows(self.rows)
        _require(type(self.usage) is BackendUsage, "invalid_usage")


@dataclass(frozen=True, repr=False)
class StateComponent:
    name: str
    role: str
    data: bytes

    def __post_init__(self):
        _require(type(self.name) is str and re.fullmatch(r"[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*", self.name)
                 and len(self.name) <= 128, "unsafe_component_name")
        _require(type(self.role) is str and self.role in ROLES, "unadmitted_component_role")
        _require(type(self.data) is bytes and 0 < len(self.data) <= MAX_STATE, "invalid_component")


class ExportState:
    """Account every provided component; opaque contents are not a privacy proof."""
    def __init__(self, metadata, components):
        fields = {"spec_sha256", "fit_sha256", "schema_sha256", "cohort_sha256", "auditors_sha256", "columns", "consumed_rows"}
        _require(type(metadata) is dict and set(metadata) == fields, "invalid_state")
        for name in fields - {"columns", "consumed_rows"}:
            _hash(metadata[name])
        _count(metadata["columns"], 2)
        _count(metadata["consumed_rows"], 1)
        _require(metadata["columns"] <= 256, "invalid_state")
        _require(type(components) is tuple and 0 < len(components) <= 128
                 and all(type(c) is StateComponent for c in components), "invalid_components")
        for component in components:
            component.__post_init__()
        _require(len({c.name for c in components}) == len(components), "duplicate_component")
        _require(sum(len(c.data) for c in components) <= MAX_STATE, "state_budget_exceeded")
        self._metadata = _json(metadata)
        self.components = tuple(sorted(components, key=lambda c: c.name))

    @property
    def metadata(self):
        return json.loads(self._metadata)

    def to_bytes(self):
        files = [{"name": c.name, "role": c.role, "bytes": len(c.data), "sha256": hashlib.sha256(c.data).hexdigest()}
                 for c in self.components]
        header = _json({"format": "dope-structured-state", "version": 1, "metadata": self.metadata, "files": files})
        _require(len(header) <= MAX_MANIFEST, "state_budget_exceeded")
        return struct.pack(">Q", len(header)) + header + b"".join(c.data for c in self.components)

    @property
    def charged_bytes(self):
        return len(self.to_bytes())

    def fits_tier(self, byte_cap):
        _count(byte_cap, 1)
        return self.charged_bytes <= byte_cap

    @classmethod
    def from_bytes(cls, body):
        _require(type(body) is bytes and 8 <= len(body) <= MAX_STATE + MAX_MANIFEST + 8, "invalid_state")
        length = struct.unpack(">Q", body[:8])[0]
        _require(0 < length <= MAX_MANIFEST and 8 + length <= len(body), "invalid_state")
        try:
            header = parse_json(body[8:8 + length])
        except EvidenceError:
            raise StructuredError("invalid_state") from None
        _require(type(header) is dict and set(header) == {"format", "version", "metadata", "files"}, "invalid_state")
        _require(header["format"] == "dope-structured-state" and type(header["version"]) is int
                 and header["version"] == 1 and type(header["files"]) is list and 0 < len(header["files"]) <= 128, "invalid_state")
        components, position = [], 8 + length
        for item in header["files"]:
            _require(type(item) is dict and set(item) == {"name", "role", "bytes", "sha256"}, "invalid_state")
            _count(item["bytes"], 1)
            _hash(item["sha256"])
            size = item["bytes"]
            _require(size <= MAX_STATE and position + size <= len(body), "invalid_state")
            payload = body[position:position + size]
            _require(hashlib.sha256(payload).hexdigest() == item["sha256"], "component_digest_mismatch")
            components.append(StateComponent(item["name"], item["role"], payload))
            position += size
        _require(position == len(body), "unlisted_state_bytes")
        state = cls(header["metadata"], tuple(components))
        _require(state.to_bytes() == body, "noncanonical_state")
        return state


class _StructuredGenerator:
    families = ()

    def __init__(self, spec, backend, *, wall_seconds=600.0, memory_bytes=16 << 30, gpu_bytes=0):
        _require(type(spec) is GeneratorSpec and spec.value["profile"]["family"] in self.families, "wrong_generator_family")
        spec.admit()
        _require(_finite(wall_seconds) and wall_seconds > 0, "invalid_budget")
        _count(memory_bytes, 1)
        _count(gpu_bytes)
        self.spec, self.backend = spec, backend
        self._limits = (float(wall_seconds), memory_bytes, gpu_bytes)
        self._elapsed, self._metadata = 0.0, None
        self._backend_identity()

    def _backend_identity(self):
        try:
            identity = getattr(self.backend, "identity_sha256", None)
        except KeyboardInterrupt:
            raise KeyboardInterrupt() from None
        except BaseException:
            raise StructuredError("backend_failed") from None
        _require(identity == self.spec.sha256, "backend_identity_mismatch")

    def _call(self, name, *args):
        self._backend_identity()
        _require(self._elapsed < self._limits[0], "wall_budget_exceeded")
        started = time.monotonic()
        try:
            result = getattr(self.backend, name)(*args)
        except KeyboardInterrupt:
            raise KeyboardInterrupt() from None
        except BaseException as error:
            if isinstance(error, (ModuleNotFoundError, FileNotFoundError, StructuredUnavailable)):
                raise StructuredUnavailable("backend_unavailable") from None
            raise StructuredError("backend_failed") from None
        finally:
            self._elapsed += time.monotonic() - started
        _require(self._elapsed <= self._limits[0], "wall_budget_exceeded")
        self._backend_identity()
        return result

    def _usage(self, usage, count):
        _require(type(usage) is BackendUsage, "invalid_usage")
        usage.__post_init__()
        _require(usage.consumed_rows == count, "context_count_mismatch")
        _require(usage.memory_bytes <= self._limits[1] and usage.gpu_bytes <= self._limits[2], "resource_budget_exceeded")

    def fit(self, view, seed):
        self._metadata, self._elapsed = None, 0.0
        _require(type(view) is FitView, "development_view_required")
        view.__post_init__()
        _count(seed)
        usage = self._call("fit", view, seed, self.spec.value)
        self._usage(usage, len(view.rows))
        self._metadata = {"spec_sha256": self.spec.sha256, "fit_sha256": digest({"view": view.sha256, "seed": seed}),
                          "schema_sha256": view.schema_sha256, "cohort_sha256": view.cohort_sha256,
                          "auditors_sha256": view.auditors_sha256, "columns": len(view.rows[0]), "consumed_rows": len(view.rows)}
        return usage

    def _ready(self):
        _require(self._metadata is not None, "generator_not_fitted")
        self._backend_identity()

    def sample(self, rows, seed):
        self._ready()
        _count(rows, 1)
        _count(seed)
        result = self._call("sample", rows, seed)
        _require(type(result) is SampleRows, "invalid_sample")
        _rows(result.rows, self._metadata["columns"], rows)
        self._usage(result.usage, self._metadata["consumed_rows"])
        return result

    def cache_key(self, requested_rows, seed):
        self._ready()
        _count(requested_rows, 1)
        _count(seed)
        return digest({"format": "dope-structured-invocation", **self._metadata, "rows": requested_rows, "seed": seed})

    def export(self):
        self._ready()
        return ExportState(self._metadata, self._call("export"))

    def reload(self, state):
        _require(type(state) is ExportState and state.metadata["spec_sha256"] == self.spec.sha256, "state_identity_mismatch")
        backend = self._call("reload", state)
        _require(backend is not self.backend, "reload_reused_fitting_backend")
        loaded = type(self)(self.spec, backend, wall_seconds=self._limits[0], memory_bytes=self._limits[1], gpu_bytes=self._limits[2])
        loaded._metadata = state.metadata
        return loaded

    def as_pr09(self, identity, admission, cohort_sha256, auditors_sha256):
        """Bind PR09 lazily; its recipe pin must cover the complete spec SHA256."""
        try:
            base = importlib.import_module("research.transfer.adapters.base")
        except ModuleNotFoundError:
            raise StructuredUnavailable("pr09_unavailable") from None
        _hash(cohort_sha256)
        _hash(auditors_sha256)
        _require(type(identity) is base.ModelIdentity and type(admission) is base.AdapterAdmission, "invalid_admission")
        _require(identity.name == self.spec.value["profile"]["name"] and identity.mode == "scratch", "identity_mismatch")
        profile = self.spec.value["profile"]
        _require(identity.code_revision == profile["source"]["revision"] and identity.code_sha256 == profile["code_sha256"]
                 and identity.preprocessing_sha256 == profile["preprocessing_sha256"]
                 and identity.recipe_sha256 == self.spec.sha256
                 and digest([list(pin) for pin in identity.dependencies]) == profile["dependencies_sha256"]
                 and digest([list(pin) for pin in identity.licenses]) == profile["license_sha256"]
                 and admission.native_reference_sha256 == profile["native_reference_sha256"], "identity_mismatch")
        engine = self

        class Bridge(base.GeneratorAdapter):
            def __init__(self, identity, admission):
                super().__init__(identity, admission)
                self._engine_fit = None

            def _ready(self):
                super()._ready()
                base._require(self._engine_fit is not None and self._engine_fit is engine._metadata, "structured_fit_changed")

            def cache_key(self, request_sha256, seed, requested_rows):
                invocation = super().cache_key(request_sha256, seed, requested_rows)
                return digest({"invocation": invocation, "fit_sha256": self._engine_fit["fit_sha256"]})

            def _invoke(self, method, *args):
                try:
                    return getattr(engine, method)(*args)
                except StructuredUnavailable:
                    raise base.AdapterUnavailable("backend_unavailable") from None

            def _fit_backend(self, training, seed, mode):
                self._engine_fit = None
                rows = tuple((*features, target) for features, target in zip(training.features.rows, training.targets))
                usage = self._invoke("fit", FitView(rows, training.features.schema_sha256, cohort_sha256, auditors_sha256, training.role), seed)
                self._engine_fit = engine._metadata
                return base.ResourceUsage(usage.consumed_rows, usage.memory_bytes, usage.gpu_bytes)

            def _sample_backend(self, rows, seed):
                result = self._invoke("sample", rows, seed)
                self._ready()
                ids = tuple(digest({"invocation": engine.cache_key(rows, seed), "row": i}) for i in range(rows))
                data = base.TrainingBatch(base.FeatureBatch(ids, tuple(r[:-1] for r in result.rows), engine._metadata["schema_sha256"]),
                                          tuple(r[-1] for r in result.rows), "synthetic_train")
                return base.SampleResult(data, base.ResourceUsage(result.usage.consumed_rows, result.usage.memory_bytes, result.usage.gpu_bytes))

        return Bridge(identity, admission)


class ARFAdapter(_StructuredGenerator):
    families = ("arf",)


class ForestFlowAdapter(_StructuredGenerator):
    families = ("forest",)


class TabPCAdapter(_StructuredGenerator):
    families = ("tabpc",)


class CompactDensityControls(_StructuredGenerator):
    families = ("compact_density",)


@dataclass(frozen=True)
class NativeObjective:
    scope_sha256: str
    direction: str
    value: float

    def __post_init__(self):
        _hash(self.scope_sha256)
        _require(type(self.direction) is str and self.direction in {"minimize", "maximize"}
                 and _finite(self.value), "invalid_native_objective")

    @classmethod
    def for_view(cls, spec, view, value):
        _require(type(spec) is GeneratorSpec and type(view) is FitView, "invalid_native_objective")
        profile = spec.value["profile"]
        scope = digest({"spec": spec.sha256, "view": view.sha256, "objective": profile["native_objective"]})
        return cls(scope, profile["native_direction"], value)


def compare_native(left, right):
    _require(type(left) is NativeObjective and type(right) is NativeObjective and
             left.scope_sha256 == right.scope_sha256 and left.direction == right.direction, "native_objectives_not_comparable")
    difference = left.value - right.value
    return difference if left.direction == "maximize" else -difference


@dataclass(frozen=True)
class CohortSchedule:
    sources: tuple
    fit_seeds: tuple
    sample_seeds: tuple
    sizes: tuple
    auditors: tuple

    def __post_init__(self):
        for axis in (self.sources, self.fit_seeds, self.sample_seeds, self.sizes, self.auditors):
            _require(type(axis) is tuple and bool(axis) and len(axis) <= 4096, "invalid_schedule")
        for source in self.sources:
            _hash(source)
        for axis in (self.fit_seeds, self.sample_seeds, self.sizes):
            for value in axis:
                _count(value, 1)
        _require(all(type(a) is str and re.fullmatch(r"[a-z0-9_]+", a) for a in self.auditors), "invalid_auditor")
        _require(all(len(set(axis)) == len(axis) for axis in
                     (self.sources, self.fit_seeds, self.sample_seeds, self.sizes, self.auditors)), "duplicate_schedule_axis")
        _require(math.prod(len(a) for a in (self.sources, self.fit_seeds, self.sample_seeds, self.sizes, self.auditors)) <= 1_000_000,
                 "schedule_budget_exceeded")

    @property
    def sha256(self):
        return digest({"sources": list(self.sources), "fit_seeds": list(self.fit_seeds), "sample_seeds": list(self.sample_seeds),
                       "sizes": list(self.sizes), "auditors": list(self.auditors)})

    def ledger(self, specs, outcomes):
        _require(type(specs) is tuple and bool(specs) and all(type(s) is GeneratorSpec for s in specs), "invalid_profiles")
        _require(len({s.sha256 for s in specs}) == len(specs), "duplicate_profile")
        _require(type(outcomes) is dict, "invalid_outcomes")
        _require(len(specs) * math.prod(len(a) for a in (self.sources, self.fit_seeds, self.sample_seeds, self.sizes, self.auditors))
                 <= 1_000_000, "schedule_budget_exceeded")
        rows, seen, schedule_sha = [], set(), self.sha256
        for spec in specs:
            spec_sha, history = spec.sha256, spec.value["profile"]["history"]
            for source in self.sources:
                for fit in self.fit_seeds:
                    for sample in self.sample_seeds:
                        for size in self.sizes:
                            for auditor in self.auditors:
                                key = (spec_sha, source, fit, sample, size, auditor)
                                status = outcomes.get(key, "unresolved")
                                _require(type(status) is str and status in STATUSES, "invalid_execution_status")
                                seen.add(key)
                                rows.append({"spec_sha256": spec_sha, "source_sha256": source, "fit_seed": fit,
                                             "sample_seed": sample, "size": size, "auditor": auditor, "status": status,
                                             "schedule_sha256": schedule_sha, "history": list(history)})
        _require(set(outcomes) <= seen, "unscheduled_outcome")
        return tuple(rows)
