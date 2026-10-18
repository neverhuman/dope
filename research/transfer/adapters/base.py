"""Portable offline research adapters. Admission is supplied by the custodian.

These interfaces receive no query labels and confer no release authority. The
executor still owns isolation, artifact verification and scientific receipts.
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import re
import time

from research.transfer.evidence import digest
from research.transfer.adapters.worker_abi import (
    AdapterError, ResourceLimits, _require, _hash, _count, _finite, _cells, build_request,
)

MODES = frozenset({"scratch", "frozen_context", "fine_tuning", "pretraining"})
REFUSED_ENCODERS = frozenset({
    "context2048", "foundation", "hyperion_v6all", "hyperion_v7", "tabdpt13",
    "tabpfn35", "ta_dope_v2", "ta_hyperion_v8", "ta_hyperion_v8braw", "target",
})


class AdapterUnavailable(AdapterError):
    """A model, license, or required offline dependency is unavailable."""


def _choice(value, choices):
    return type(value) is str and value in choices


def _backend_failure(error):
    if isinstance(error, (ModuleNotFoundError, FileNotFoundError)):
        return AdapterUnavailable("dependency_unavailable")
    if isinstance(error, AdapterUnavailable):
        return AdapterUnavailable("backend_unavailable")
    return AdapterError("backend_failed")


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    mode: str
    code_revision: str
    code_sha256: str
    weights_sha256: str | None
    dependencies: tuple  # (package, exact version, offline artifact SHA-256)
    licenses: tuple  # (code/weights scope, license identifier, text SHA-256)
    preprocessing_sha256: str
    recipe_sha256: str
    output_aggregation_sha256: str
    environment_sha256: str
    precision: str
    context_cap: int
    context_policy: str
    query_policy: str
    query_batch_size: int
    order_sensitive: bool
    determinism: str
    resources: ResourceLimits

    def __post_init__(self):
        _require(type(self.name) is str and re.fullmatch(r"[a-z0-9_]+", self.name), "invalid_name")
        _require(_choice(self.mode, MODES), "invalid_mode")
        _require(type(self.code_revision) is str and re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", self.code_revision), "invalid_revision")
        for pin in (self.code_sha256, self.preprocessing_sha256, self.recipe_sha256,
                    self.output_aggregation_sha256, self.environment_sha256):
            _hash(pin)
        if self.weights_sha256 is not None:
            _hash(self.weights_sha256)
        _require(self.mode not in {"frozen_context", "fine_tuning"} or self.weights_sha256 is not None, "missing_weights")
        _require(self.mode != "scratch" or self.weights_sha256 is None, "scratch_has_weights")
        for pins in (self.dependencies, self.licenses):
            _require(type(pins) is tuple and bool(pins), "missing_dependency_or_license")
            for pin in pins:
                _require(type(pin) is tuple and len(pin) == 3, "invalid_dependency_or_license")
                _require(all(type(v) is str and bool(v) for v in pin), "invalid_dependency_or_license")
                _hash(pin[2])
            _require(len({p[0] for p in pins}) == len(pins), "duplicate_pin")
        _require(all(not re.search(r"[<>=*!~^\s]", p[1]) for p in self.dependencies), "unpinned_dependency")
        scopes = {p[0] for p in self.licenses}
        _require(scopes == ({"code", "weights"} if self.weights_sha256 else {"code"}), "missing_license")
        _require(_choice(self.precision, {"float32", "float64", "bfloat16"}), "invalid_precision")
        _count(self.context_cap, 1)
        _count(self.query_batch_size, 1)
        _require(_choice(self.context_policy, {"all", "head"}), "invalid_context_policy")
        _require(_choice(self.query_policy, {"rowwise", "batch", "transductive"}), "invalid_query_policy")
        _require(self.query_policy != "rowwise" or self.query_batch_size == 1, "invalid_query_policy")
        _require(type(self.order_sensitive) is bool, "invalid_order_policy")
        _require(_choice(self.determinism, {"deterministic", "seeded_best_effort"}), "invalid_determinism")
        _require(type(self.resources) is ResourceLimits, "invalid_budget")

    @property
    def sha256(self):
        value = asdict(self)
        value["dependencies"] = [list(p) for p in self.dependencies]
        value["licenses"] = [list(p) for p in self.licenses]
        return digest({"format": "dope-adapter-identity-v1", **value})


@dataclass(frozen=True)
class AdapterAdmission:
    """Registry decision, not an artifact verifier or scientific qualification."""
    identity_sha256: str
    eligibility_sha256: str
    native_reference_sha256: str
    kind: str

    def __post_init__(self):
        for pin in (self.identity_sha256, self.eligibility_sha256, self.native_reference_sha256):
            _hash(pin)
        _require(_choice(self.kind, {"learner", "generator"}), "invalid_adapter_kind")

    @classmethod
    def from_registry(cls, registry, identity, kind):
        _require(type(identity) is ModelIdentity, "invalid_identity")
        _require(type(registry) is dict and set(registry) == {"format", "kind", "entries"}, "invalid_registry")
        _require(registry["format"] == "dope-adapter-registry-v1" and registry["kind"] == kind, "invalid_registry")
        _require(type(registry["entries"]) is list, "invalid_registry")
        seen, entry = set(), None
        for candidate in registry["entries"]:
            _require(type(candidate) is dict and set(candidate) == {
                "name", "status", "identity_sha256", "eligibility_sha256",
                "native_reference_sha256", "license_status"}, "invalid_registry")
            name = candidate["name"]
            _require(type(name) is str and name not in seen, "invalid_registry")
            seen.add(name)
            _require(_choice(candidate["status"], {"unresolved", "reviewed", "unavailable"}), "invalid_registry")
            _require(_choice(candidate["license_status"], {"unresolved", "compatible", "incompatible"}), "invalid_registry")
            for key in ("identity_sha256", "eligibility_sha256", "native_reference_sha256"):
                if candidate[key] is not None:
                    _hash(candidate[key])
                _require(candidate["status"] != "reviewed" or candidate[key] is not None, "invalid_registry")
            if name == identity.name:
                entry = candidate
        if identity.name in REFUSED_ENCODERS:
            raise AdapterUnavailable("refused_encoder")
        if entry is None or entry["status"] != "reviewed":
            raise AdapterUnavailable("model_unavailable")
        if entry["license_status"] != "compatible":
            raise AdapterUnavailable("license_unavailable")
        if entry["identity_sha256"] != identity.sha256:
            raise AdapterUnavailable("identity_unavailable")
        return cls(entry["identity_sha256"], entry["eligibility_sha256"],
                   entry["native_reference_sha256"], kind)


@dataclass(frozen=True, repr=False)
class FeatureBatch:
    """Private immutable feature rows; row IDs are opaque commitments."""
    row_ids: tuple
    rows: tuple
    schema_sha256: str

    def __post_init__(self):
        _hash(self.schema_sha256)
        _cells(self.rows)
        _require(type(self.row_ids) is tuple and len(self.row_ids) == len(self.rows), "invalid_row_ids")
        for row_id in self.row_ids:
            _hash(row_id)
        _require(len(set(self.row_ids)) == len(self.row_ids), "duplicate_row_id")

    @property
    def sha256(self):
        return digest({"schema_sha256": self.schema_sha256, "row_ids": list(self.row_ids),
                       "features": [list(row) for row in self.rows]})


@dataclass(frozen=True, repr=False)
class TrainingBatch:
    features: FeatureBatch
    targets: tuple
    role: str

    def __post_init__(self):
        _require(type(self.features) is FeatureBatch, "invalid_features")
        _require(type(self.targets) is tuple and len(self.targets) == len(self.features.rows), "invalid_targets")
        _require(all(type(v) is str or _finite(v) for v in self.targets), "invalid_targets")
        _require(_choice(self.role, {"development_train", "synthetic_train"}), "protected_training_role")

    @property
    def sha256(self):
        return digest({"features_sha256": self.features.sha256,
                       "targets": list(self.targets), "role": self.role})


@dataclass(frozen=True, repr=False)
class PredictionResult:
    """Original-unit predictions. Optional embeddings never gate predictions."""
    row_ids: tuple
    values: tuple
    kind: str
    consumed_rows: int
    memory_bytes: int
    gpu_bytes: int
    class_order: tuple = ()
    embedding: tuple | None = field(default=None, repr=False)
    embedding_unavailable: str | None = None

    def __post_init__(self):
        _require(_choice(self.kind, {"point", "probability", "distribution"}), "invalid_output_kind")
        _cells(self.values)
        _require(all(_finite(v) for row in self.values for v in row), "nonfinite_prediction")
        _require(type(self.row_ids) is tuple and len(self.row_ids) == len(self.values), "prediction_alignment")
        for row_id in self.row_ids:
            _hash(row_id)
        _require(len(set(self.row_ids)) == len(self.row_ids), "prediction_alignment")
        _require(type(self.class_order) is tuple, "invalid_classes")
        if self.kind == "point":
            _require(len(self.values[0]) == 1 and not self.class_order, "invalid_output_shape")
        elif self.kind == "probability":
            _require(len(self.class_order) == len(self.values[0]) and bool(self.class_order), "invalid_classes")
            _require(all(type(c) is str for c in self.class_order) and len(set(self.class_order)) == len(self.class_order), "invalid_classes")
            _require(all(all(0 <= p <= 1 for p in row) and abs(sum(row) - 1) <= 1e-7 for row in self.values), "invalid_probability")
        else:
            _require(not self.class_order, "invalid_classes")
        _count(self.consumed_rows, 1)
        _count(self.memory_bytes)
        _count(self.gpu_bytes)
        if self.embedding is not None:
            _cells(self.embedding)
            _require(len(self.embedding) == len(self.values) and all(_finite(v) for row in self.embedding for v in row), "invalid_embedding")
            _require(self.embedding_unavailable is None, "invalid_embedding_status")
        else:
            _require(_choice(self.embedding_unavailable, {"not_requested", "unsupported", "unavailable"}), "invalid_embedding_status")


@dataclass(frozen=True)
class ResourceUsage:
    """Executor observations; backend self-reports require independent checking."""
    consumed_rows: int
    memory_bytes: int
    gpu_bytes: int

    def __post_init__(self):
        _count(self.consumed_rows, 1)
        _count(self.memory_bytes)
        _count(self.gpu_bytes)


@dataclass(frozen=True)
class ImportanceResult:
    """Grouped real-test loss increases; grouping/perturbations are evaluator pins."""
    protocol_sha256: str
    query_sha256: str
    group_ids: tuple
    loss_increases: tuple

    def __post_init__(self):
        _hash(self.protocol_sha256)
        _hash(self.query_sha256)
        _require(type(self.group_ids) is tuple and bool(self.group_ids), "invalid_groups")
        _require(all(type(g) is int and g >= 0 for g in self.group_ids) and
                 len(set(self.group_ids)) == len(self.group_ids), "invalid_groups")
        _require(type(self.loss_increases) is tuple and
                 len(self.loss_increases) == len(self.group_ids) and
                 all(_finite(v) for v in self.loss_increases), "invalid_importance")


@dataclass(frozen=True, repr=False)
class SampleResult:
    data: TrainingBatch
    usage: ResourceUsage

    def __post_init__(self):
        _require(type(self.data) is TrainingBatch and type(self.usage) is ResourceUsage, "invalid_sample")


class _Adapter(ABC):
    def __init__(self, identity, admission, kind):
        _require(type(identity) is ModelIdentity and type(admission) is AdapterAdmission, "invalid_admission")
        _require(admission.kind == kind and admission.identity_sha256 == identity.sha256, "invalid_admission")
        if identity.name in REFUSED_ENCODERS:
            raise AdapterUnavailable("refused_encoder")
        self._identity = identity
        self._kind = kind
        self._training = None
        self._provided_training = None
        self._provided_sha256 = None
        self._provided_ids = ()
        self._provided_rows = 0
        self._fit_seed = None
        self._wall_seconds = 0.0

    @property
    def identity(self):
        return self._identity

    def _context(self, training, seed):
        _require(type(training) is TrainingBatch, "invalid_training")
        _require(self._kind != "generator" or training.role == "development_train", "protected_training_role")
        _count(seed)
        count = len(training.features.rows)
        if self.identity.context_policy == "all":
            _require(count <= self.identity.context_cap, "context_budget_exceeded")
            return training
        n = min(count, self.identity.context_cap)
        return TrainingBatch(FeatureBatch(training.features.row_ids[:n], training.features.rows[:n],
                                          training.features.schema_sha256), training.targets[:n], training.role)

    def _call(self, operation, *args):
        _require(self._wall_seconds < self.identity.resources.wall_seconds, "wall_budget_exceeded")
        started = time.monotonic()
        try:
            result = operation(*args)
        except BaseException as error:
            # Native libraries can exit with a source-bearing message. Contain
            # that exit, while retaining operator cancellation without payload.
            if isinstance(error, KeyboardInterrupt):
                raise KeyboardInterrupt() from None
            raise _backend_failure(error) from None
        finally:
            self._wall_seconds += time.monotonic() - started
        _require(self._wall_seconds <= self.identity.resources.wall_seconds, "wall_budget_exceeded")
        return result

    def _prepare(self, training, seed, operation):
        # Invalidate old state before calling any fallible backend operation.
        self._training = None
        self._provided_training = None
        self._provided_sha256 = None
        self._wall_seconds = 0.0
        context = self._context(training, seed)
        usage = self._call(operation, context, seed, self.identity.mode)
        _require(type(usage) is ResourceUsage, "invalid_resource_usage")
        self._usage(usage, len(context.targets))
        self._training = context
        self._provided_training = training
        self._provided_sha256 = training.sha256
        self._provided_ids = training.features.row_ids
        self._provided_rows = len(training.targets)
        self._fit_seed = seed

    def _ready(self):
        _require(self._training is not None, "adapter_not_prepared")

    def _usage(self, usage, consumed_rows):
        _require(usage.consumed_rows == consumed_rows, "context_count_mismatch")
        _require(usage.memory_bytes <= self.identity.resources.memory_bytes and
                 usage.gpu_bytes <= self.identity.resources.gpu_bytes, "resource_budget_exceeded")

    def cache_key(self, request_sha256, seed, requested_rows):
        """Invocation key only; scientific completion requires PR02 verification."""
        self._ready()
        _hash(request_sha256)
        _count(seed)
        _count(requested_rows, 1)
        return digest({"format": "dope-adapter-invocation-v1", "kind": self._kind,
                       "model_sha256": self.identity.sha256,
                       "provided_sha256": self._provided_sha256, "provided_rows": self._provided_rows,
                       "context_sha256": self._training.sha256, "consumed_rows": len(self._training.targets),
                       "fit_seed": self._fit_seed, "request_sha256": request_sha256,
                       "request_seed": seed, "requested_rows": requested_rows})

    def receipt(self):
        """Portable execution metadata only; independent qualification is required."""
        self._ready()
        return {"format": "dope-adapter-execution-v1", "kind": self._kind,
                "identity_sha256": self.identity.sha256,
                "provided_sha256": self._provided_sha256,
                "context_sha256": self._training.sha256,
                "fit_seed": self._fit_seed, "provided_rows": self._provided_rows,
                "consumed_rows": len(self._training.targets),
                "wall_seconds": self._wall_seconds, "citable": False}

    def worker_request(self, pins, resources, *, seed, primary_n, query=None,
                       sample_multiplier=1, predict_time_fit=False, subsampling="none", output=None):
        """Export the admitted context without query labels; launch is executor-owned.

        request_sha256, rather than the legacy cache_key, binds all native runtime,
        resource and sensitivity policies. This contract remains fixture-only.
        """
        self._ready()
        _require(type(pins) is dict and pins.get("model_sha256") == self.identity.sha256,
                 "invalid_worker_pins")
        if self.identity.weights_sha256 is not None:
            _require(pins.get("checkpoint_sha256") == self.identity.weights_sha256, "invalid_worker_pins")
        _require(type(resources) is dict and all(type(resources.get(k)) is int for k in
                 ("memory_bytes", "vram_bytes", "wall_ms"))
                 and resources["memory_bytes"] <= self.identity.resources.memory_bytes
                 and resources.get("vram_bytes", 2**53) <= self.identity.resources.gpu_bytes
                 and resources.get("wall_ms", 2**53) <= self.identity.resources.wall_seconds * 1000,
                 "invalid_worker_resources")
        policy = {"primary_n": primary_n, "sample_multiplier": sample_multiplier,
                  "context_cap": self.identity.context_cap,
                  "truncation": "head" if self.identity.context_policy == "head" else "none",
                  "subsampling": subsampling, "predict_time_fit": predict_time_fit,
                  "query_policy": self.identity.query_policy, "query_batch_size": self.identity.query_batch_size,
                  "order_sensitive": self.identity.order_sensitive}
        return build_request(pins, resources, policy,
                             "predict" if query is not None else "fit_or_prepare_context", seed,
                             training=self._provided_training, query=query,
                             context_row_ids=self._training.features.row_ids, fit_seed=self._fit_seed, output=output)


class LearnerAdapter(_Adapter):
    def __init__(self, identity, admission):
        super().__init__(identity, admission, "learner")

    def prepare(self, training, seed):
        self._prepare(training, seed, self._prepare_backend)

    def fit(self, training, seed):
        """TDL ABI spelling; prepare remains compatible with historical callers."""
        self.prepare(training, seed)
        return self.receipt()

    @abstractmethod
    def _prepare_backend(self, training, seed, mode):
        """Prepare the declared mode and return observed ResourceUsage."""

    @abstractmethod
    def _predict_backend(self, query, seed):
        """Receive features only; restore original target units before returning."""

    def predict(self, query, seed):
        self._ready()
        _require(type(query) is FeatureBatch, "query_features_required")
        _count(seed)
        _require(len(query.rows) <= self.identity.query_batch_size, "query_budget_exceeded")
        train = self._training.features
        _require(query.schema_sha256 == train.schema_sha256 and len(query.rows[0]) == len(train.rows[0]), "schema_mismatch")
        _require(not set(query.row_ids).intersection(self._provided_ids), "query_context_overlap")
        result = self._call(self._predict_backend, query, seed)
        _require(type(result) is PredictionResult, "invalid_prediction_result")
        _require(result.row_ids == query.row_ids, "prediction_alignment")
        self._usage(result, len(self._training.targets))
        return result

    def importance(self, real_test, perturbations, score_loss, protocol_sha256, seed):
        """Evaluate frozen grouped/conditional permutations on identical real rows.

        perturbations is ((opaque_group_id, (FeatureBatch, ...)), ...). The
        evaluator freezes training-only groups and conditional resampling and
        supplies score_loss(prediction), keeping query labels out of predict.
        Raw negative loss increases are retained for evaluator diagnostics.
        """
        self._ready()
        _hash(protocol_sha256)
        _require(type(real_test) is FeatureBatch, "query_features_required")
        _require(type(perturbations) is tuple and bool(perturbations), "invalid_groups")
        _require(callable(score_loss), "loss_evaluator_required")
        baseline = self._call(score_loss, self.predict(real_test, seed))
        _require(_finite(baseline) and baseline >= 0, "invalid_loss")
        ids, increases = [], []
        for group_id, views in perturbations:
            _count(group_id)
            _require(group_id not in ids and type(views) is tuple and bool(views), "invalid_groups")
            losses = []
            for view in views:
                _require(type(view) is FeatureBatch and view.row_ids == real_test.row_ids and
                         view.schema_sha256 == real_test.schema_sha256, "importance_alignment")
                value = self._call(score_loss, self.predict(view, seed))
                _require(_finite(value) and value >= 0, "invalid_loss")
                losses.append(value)
            increase = sum(value / len(losses) for value in losses) - baseline
            _require(_finite(increase), "invalid_importance")
            ids.append(group_id)
            increases.append(increase)
        return ImportanceResult(protocol_sha256, real_test.sha256, tuple(ids), tuple(increases))


class GeneratorAdapter(_Adapter):
    def __init__(self, identity, admission):
        super().__init__(identity, admission, "generator")

    def fit(self, training, seed):
        self._prepare(training, seed, self._fit_backend)

    @abstractmethod
    def _fit_backend(self, training, seed, mode):
        """Fit the development training view and return observed ResourceUsage."""

    @abstractmethod
    def _sample_backend(self, rows, seed):
        """Return SampleResult from fitted state, with no query data or labels."""

    def sample(self, rows, seed):
        self._ready()
        _count(rows, 1)
        _count(seed)
        result = self._call(self._sample_backend, rows, seed)
        _require(type(result) is SampleResult and result.data.role == "synthetic_train", "invalid_sample")
        self._usage(result.usage, len(self._training.targets))
        _require(len(result.data.targets) == rows, "sample_count_mismatch")
        train = self._training.features
        _require(result.data.features.schema_sha256 == train.schema_sha256 and len(result.data.features.rows[0]) == len(train.rows[0]), "schema_mismatch")
        _require(not set(result.data.features.row_ids).intersection(self._provided_ids), "sample_context_overlap")
        return result
