"""Offline native learners and development-only conventional recipes.

Artifact acquisition is a separate, explicitly authorized executor phase. No
loader here downloads anything. The trusted executor supplies immutable local
files, checks the installed code/environment against those files, denies network
access, and observes resources independently. Callbacks are not a sandbox.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import stat
import time

from research.transfer.adapters.base import (
    AdapterError, AdapterUnavailable, FeatureBatch, LearnerAdapter, PredictionResult, ResourceUsage,
    ResourceLimits, TrainingBatch, _backend_failure, _finite, _hash, _require,
)
from research.transfer.evidence import digest


def recipe_digest(value):
    """Convert immutable recipe tuples to the evidence JSON array vocabulary."""
    return digest(json.loads(json.dumps(value, allow_nan=False)))


def require_native_dependency(identity, package, exact=None):
    """Native loaders require the declared primary distribution."""
    pins = {p[0]: p[1] for p in identity.dependencies}
    if package not in pins or (exact is not None and pins[package] != exact):
        raise AdapterUnavailable("native_dependency_pin_unavailable")


def immutable_parameter(value):
    return (type(value) in {str, bool} or _finite(value) or
            (type(value) is tuple and all(immutable_parameter(v) for v in value)))


def file_sha256(path):
    """Stream a regular final file; diagnostics never contain its path.

    O_NOFOLLOW covers the final component. The executor verifies and freezes
    ancestor directories as part of immutable snapshot admission; this helper
    does not establish filesystem isolation.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise AdapterUnavailable("artifact_unavailable")
            hasher = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
            return hasher.hexdigest()
    except (OSError, TypeError, ValueError):
        raise AdapterUnavailable("artifact_unavailable") from None


def port_source_sha256(module_file):
    """Bind executed first-party wrappers as well as the external code archive.

    The executor verifies installation and prevents source replacement. These
    relative-path byte pins also invalidate a recipe after a wrapper change.
    """
    transfer = Path(__file__).resolve().parents[2]
    files = {Path(module_file).resolve(), Path(__file__).resolve(),
             transfer / "adapters/base.py", transfer / "evidence.py"}
    return {str(p.relative_to(transfer)): file_sha256(p) for p in sorted(files)}


@dataclass(frozen=True, repr=False)
class OfflineArtifacts:
    """Files in a custodian-verified immutable snapshot, in identity pin order.

    Additional checkpoint files (e.g. config.json) are pinned by the recipe.
    The executor must also prevent replacement between verification and load.
    """
    code: Path
    dependencies: tuple
    licenses: tuple
    weights: Path | None = None
    ancillary: tuple = ()  # (local Path, SHA-256), also bound by the recipe

    def verify(self, identity, version_reader=version):
        expected = [identity.code_sha256]
        paths = [self.code]
        for files, pins in ((self.dependencies, identity.dependencies),
                            (self.licenses, identity.licenses)):
            _require(type(files) is tuple and len(files) == len(pins), "artifact_count_mismatch")
            paths.extend(files)
            expected.extend(p[2] for p in pins)
        _require((self.weights is None) == (identity.weights_sha256 is None), "artifact_count_mismatch")
        if self.weights is not None:
            paths.append(self.weights)
            expected.append(identity.weights_sha256)
        _require(type(self.ancillary) is tuple, "invalid_artifacts")
        for path, sha256 in self.ancillary:
            _hash(sha256)
            paths.append(path)
            expected.append(sha256)
        for path, sha256 in zip(paths, expected):
            if file_sha256(path) != sha256:
                raise AdapterUnavailable("artifact_pin_mismatch")
        try:
            matches = all(version_reader(package) == exact for package, exact, _ in identity.dependencies)
        except Exception:
            raise AdapterUnavailable("dependency_unavailable") from None
        if not matches:
            raise AdapterUnavailable("dependency_version_mismatch")


@dataclass(frozen=True, repr=False)
class OfflineRuntime:
    """Trusted executor hooks; factory=None selects the pinned native loader.

    The observer returns measured peaks and the actually consumed context count,
    not an estimate. Production admission cannot use fixture factories/observers.
    """
    artifacts: OfflineArtifacts
    observer: object
    factory: object = None
    version_reader: object = version

    def verify(self, identity):
        _require(type(self.artifacts) is OfflineArtifacts and callable(self.observer), "invalid_runtime")
        _require(self.factory is None or callable(self.factory), "invalid_runtime")
        self.artifacts.verify(identity, self.version_reader)

    def observe(self, event, rows):
        usage = self.observer(event)
        _require(type(usage) is ResourceUsage and usage.consumed_rows == rows, "context_count_mismatch")
        return usage


def validate_task(task, class_order):
    _require(task in {"regression", "classification"}, "invalid_task")
    _require(type(class_order) is tuple and all(type(c) is str for c in class_order), "invalid_classes")
    _require(len(set(class_order)) == len(class_order), "invalid_classes")
    _require((task == "classification") == bool(class_order), "invalid_classes")
    _require(task != "classification" or len(class_order) >= 2, "invalid_classes")


def numeric_features(rows):
    """Numeric first-paper interface; upstream schema owns categorical encoding."""
    _require(all(v is None or _finite(v) for row in rows for v in row), "numeric_features_required")
    import numpy as np
    return np.asarray([[float("nan") if v is None else v for v in row] for row in rows], dtype=np.float64)


def native_targets(training, task, class_order):
    import numpy as np
    if task == "regression":
        _require(all(_finite(y) for y in training.targets), "numeric_targets_required")
        return np.asarray(training.targets, dtype=np.float64)
    _require(all(type(y) is str and y in class_order for y in training.targets), "unknown_target_class")
    # Refuse absent classes prospectively rather than silently renormalizing a
    # native head or manufacturing probabilities for unobserved classes.
    _require(set(training.targets) == set(class_order), "context_class_missing")
    return np.asarray([class_order.index(y) for y in training.targets], dtype=np.int64)


def output_rows(values, rows, width):
    import numpy as np
    array = np.asarray(values)
    _require(array.shape == (rows, width) and array.dtype.kind in "fiu", "invalid_native_output")
    _require(bool(np.isfinite(array).all()), "invalid_native_output")
    return tuple(tuple(float(v) for v in row) for row in array)


class ObservedLearner(LearnerAdapter):
    """Shared fit/predict plumbing; outputs are native original-unit predictions."""
    def __init__(self, identity, admission, recipe, runtime):
        super().__init__(identity, admission)
        _require(type(runtime) is OfflineRuntime, "invalid_runtime")
        _require(identity.recipe_sha256 == recipe.sha256, "recipe_pin_mismatch")
        _require(identity.precision == recipe.precision, "precision_mismatch")
        _require(identity.query_policy in {"batch", "transductive"}, "query_policy_mismatch")
        self._recipe, self._runtime, self._model = recipe, runtime, None

    @property
    def recipe(self):
        return self._recipe

    def _prepare_backend(self, training, seed, mode):
        self._model = None
        self._runtime.verify(self.identity)
        x = numeric_features(training.features.rows)
        y = native_targets(training, self.recipe.task, self.recipe.class_order)
        model = self._create_backend(seed)
        self._fit_native(model, x, y)
        usage = self._runtime.observe("prepare", len(y))
        self._model = model
        return usage

    def _fit_native(self, model, x, y):
        model.fit(x, y)

    def _raw_predict(self, x):
        return self._model.predict_proba(x) if self.recipe.task == "classification" else self._model.predict(x)

    def _native_classes(self):
        return self._model.classes_

    def _predict_backend(self, query, seed):
        import numpy as np
        x = numeric_features(query.rows)
        values = self._raw_predict(x)
        count = len(self._training.targets)
        usage = self._runtime.observe("predict", count)
        if self.recipe.task == "classification":
            classes = np.asarray(self._native_classes())
            _require(classes.shape == (len(self.recipe.class_order),) and
                     classes.dtype.kind in "iu" and set(classes.tolist()) == set(range(len(classes))),
                     "native_class_mismatch")
            values = np.asarray(values)
            _require(values.shape == (len(x), len(classes)), "invalid_native_output")
            values = values[:, np.argsort(classes)]
            kind, width, order = "probability", len(classes), self.recipe.class_order
        else:
            values = np.asarray(values)
            _require(values.shape in {(len(x),), (len(x), 1)}, "invalid_native_output")
            values = values.reshape(len(x), 1)
            kind, width, order = "point", 1, ()
        return PredictionResult(query.row_ids, output_rows(values, len(x), width), kind,
                                count, usage.memory_bytes, usage.gpu_bytes, order,
                                embedding_unavailable="unsupported")


@dataclass(frozen=True, repr=False)
class ConventionalLearnerRecipe:
    """Selected on development data, then identical for real/synthetic fits.

    Linear is a diagnostic. CatBoost, histogram boosting and MLP are distinct
    architecture groups, not three variants of the same checkpoint family.
    Parameters are immutable tuples; seed is supplied by each invocation.
    """
    family: str
    task: str
    parameters: tuple
    selection_sha256: str | None = None
    class_order: tuple = ()
    precision: str = "float64"

    def __post_init__(self):
        validate_task(self.task, self.class_order)
        _require(self.family in {"catboost", "histogram_boosting", "mlp", "linear"}, "invalid_family")
        _require(self.precision == ("float32" if self.family == "catboost" else "float64"), "precision_mismatch")
        _require(type(self.parameters) is tuple, "invalid_parameters")
        keys = []
        for key, value in self.parameters:
            _require(type(key) is str and immutable_parameter(value), "invalid_parameters")
            keys.append(key)
        _require(len(set(keys)) == len(keys), "invalid_parameters")
        reserved = {"random_state", "random_seed", "thread_count", "n_jobs", "task_type",
                    "verbose", "allow_writing_files", "early_stopping", "eval_set"}
        _require(not set(keys) & reserved, "reserved_parameter")
        if self.selection_sha256 is not None:
            _hash(self.selection_sha256)

    @property
    def model_name(self):
        return {"catboost": "catboost", "histogram_boosting": "independent_boosting",
                "mlp": "independent_neural", "linear": "linear"}[self.family]

    @property
    def sha256(self):
        return recipe_digest({"format": "dope-conventional-recipe-v1",
                              "adapter_source_sha256": port_source_sha256(__file__), **asdict(self)})

    @property
    def preprocessing_sha256(self):
        return digest({"format": "dope-conventional-input-v1", "family": self.family,
                       "features": "numeric_missing_nan", "target": "raw",
                       "fit_only": True, "scaling": self.family in {"linear", "mlp"}})

    @property
    def output_sha256(self):
        return recipe_digest({"format": "dope-conventional-output-v1", "task": self.task,
                       "class_order": self.class_order, "absent_class": "refuse", "units": "raw"})


def select_conventional_recipe(candidates, training, validation, fit_and_score, limits):
    """One declared development search; retain its receipt with all fit costs.

    The trusted executor returns (original-unit validation loss, ResourceUsage)
    per candidate and enforces hard isolation/deadlines. This function checks
    the cumulative wall budget and returns the winning immutable recipe plus
    a receipt; protected query labels are never an input to learner.predict.
    """
    _require(type(candidates) is tuple and bool(candidates), "invalid_candidates")
    _require(type(limits) is ResourceLimits and callable(fit_and_score), "invalid_tuning_runtime")
    _require(all(type(batch) is TrainingBatch and batch.role == "development_train"
                 for batch in (training, validation)), "development_only_selection")
    _require(not set(training.features.row_ids) & set(validation.features.row_ids), "development_overlap")
    _require(training.features.schema_sha256 == validation.features.schema_sha256, "schema_mismatch")
    _require(type(candidates[0]) is ConventionalLearnerRecipe, "invalid_candidates")
    first = candidates[0]
    _require(all(type(c) is ConventionalLearnerRecipe and c.selection_sha256 is None and
                 (c.family, c.task, c.class_order) == (first.family, first.task, first.class_order)
                 for c in candidates), "invalid_candidates")
    start, attempts = time.monotonic(), []
    for candidate in candidates:
        _require(time.monotonic() - start < limits.wall_seconds, "wall_budget_exceeded")
        try:
            loss, usage = fit_and_score(candidate, training, validation)
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                raise KeyboardInterrupt() from None
            raise _backend_failure(error) from None
        _require(_finite(loss) and loss >= 0 and type(usage) is ResourceUsage, "invalid_tuning_result")
        _require(usage.consumed_rows == len(training.targets), "context_count_mismatch")
        _require(usage.memory_bytes <= limits.memory_bytes and usage.gpu_bytes <= limits.gpu_bytes,
                 "resource_budget_exceeded")
        _require(time.monotonic() - start <= limits.wall_seconds, "wall_budget_exceeded")
        attempts.append({"recipe_sha256": candidate.sha256, "loss": float(loss), "usage": asdict(usage)})
    winner = min(range(len(attempts)), key=lambda i: (attempts[i]["loss"], i))
    selection = {"format": "dope-development-selection-v1", "training_sha256": training.sha256,
                 "validation_sha256": validation.sha256,
                 "candidates": [{"recipe_sha256": a["recipe_sha256"], "loss": a["loss"]} for a in attempts],
                 "winner": winner, "limits": asdict(limits)}
    selection_sha256 = digest(selection)
    receipt = {"format": "dope-development-tuning-v1", "training_sha256": training.sha256,
               "validation_sha256": validation.sha256, "attempts": attempts,
               "winner": winner, "limits": asdict(limits), "selection_sha256": selection_sha256,
               "elapsed_seconds": time.monotonic() - start}
    return replace(candidates[winner], selection_sha256=selection_sha256), receipt


class ConventionalAdapter(ObservedLearner):
    def __init__(self, identity, admission, recipe, runtime):
        _require(type(recipe) is ConventionalLearnerRecipe, "invalid_recipe")
        _require(identity.name == recipe.model_name and identity.mode == "scratch", "mode_mismatch")
        _require(recipe.family == "linear" or recipe.selection_sha256 is not None, "development_selection_required")
        _require(identity.preprocessing_sha256 == recipe.preprocessing_sha256 and
                 identity.output_aggregation_sha256 == recipe.output_sha256, "transform_pin_mismatch")
        super().__init__(identity, admission, recipe, runtime)

    def _create_backend(self, seed):
        recipe, kwargs = self.recipe, dict(self.recipe.parameters)
        if self._runtime.factory is not None:
            return self._runtime.factory(family=recipe.family, task=recipe.task, seed=seed, parameters=kwargs)
        classification = recipe.task == "classification"
        if recipe.family == "catboost":
            require_native_dependency(self.identity, "catboost", "1.2.10")
            from catboost import CatBoostClassifier, CatBoostRegressor
            cls = CatBoostClassifier if classification else CatBoostRegressor
            return cls(**kwargs, random_seed=seed, thread_count=1, verbose=False,
                       allow_writing_files=False, task_type="CPU")
        require_native_dependency(self.identity, "scikit-learn", "1.7.2")
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if recipe.family == "histogram_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
            return cls(**kwargs, random_state=seed, early_stopping=False)
        if recipe.family == "mlp":
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            cls = MLPClassifier if classification else MLPRegressor
            estimator = cls(**kwargs, random_state=seed, early_stopping=False)
        else:
            from sklearn.linear_model import LinearRegression, LogisticRegression
            estimator = LogisticRegression(**kwargs, random_state=seed, n_jobs=1) if classification else LinearRegression(**kwargs, n_jobs=1)
        return make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), estimator)

    def _fit_native(self, model, x, y):
        if self._runtime.factory is not None:
            return model.fit(x, y)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            model.fit(x, y)

    def _raw_predict(self, x):
        if self._runtime.factory is not None:
            return super()._raw_predict(x)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return super()._raw_predict(x)


def cpu_smoke_lock():
    """Repository-owned reduced panel; a lock is not foundation/GPU admission."""
    try:
        return json.loads((Path(__file__).resolve().parents[2] / "locks/cpu-smoke.json").read_text())
    except (ValueError, TypeError, KeyError, RuntimeError, OSError, ArithmeticError):
        raise AdapterUnavailable("cpu_smoke_lock_unavailable") from None


def _cpu_smoke_model(name, task, parameters, seed, lock):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    classification = task != "regression"
    if name == "decision_tree":
        cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
        return cls(**parameters, random_state=seed)
    estimator = (LogisticRegression(**parameters, random_state=seed, n_jobs=1) if classification
                 else Ridge(**parameters, random_state=seed))
    stages = [SplineTransformer(**lock["spline"])] if name == "additive_spline_surrogate" else []
    return make_pipeline(*stages, StandardScaler(), estimator)


def cpu_smoke_predict(request):
    """W00 native-call kernel after buffer admission, not a standalone wire ABI.

    Typed training/query objects carry X_train/y_train and X_query only. D02
    provenance/rights admission and process isolation remain executor duties.
    """
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        started, lock = time.monotonic(), cpu_smoke_lock()
        fields = {"estimator", "parameters", "task", "seed", "training", "query", "counts", "lock_sha256"}
        _require(type(request) is dict and set(request) == fields, "cpu_smoke_request_invalid")
        request = dict(request)
        _require(platform.system() == "Linux" and platform.python_version().startswith(lock["python"].rsplit(".", 1)[0] + "."), "cpu_smoke_runtime_mismatch")
        _require(all(version(p) == v for p, v in lock["dependencies"].items()), "cpu_smoke_runtime_mismatch")
        _require(request["lock_sha256"] == digest(lock), "cpu_smoke_lock_mismatch")
        name, task, parameters = request["estimator"], request["task"], request["parameters"]
        _require(type(name) is str and name in lock["estimators"] and type(task) is str, "cpu_smoke_estimator_unpinned")
        grid = lock["estimators"][name]["task_grids"].get(task, [])
        _require(type(parameters) is dict and digest(parameters) in [digest(p) for p in grid], "cpu_smoke_recipe_unpinned")
        parameters = request["parameters"] = dict(parameters)
        _require(type(request["seed"]) is int and request["seed"] in lock["seeds"], "cpu_smoke_seed_unpinned")
        train, query, counts = request["training"], request["query"], request["counts"]
        _require(type(train) is TrainingBatch and type(query) is FeatureBatch, "cpu_smoke_input_invalid")
        n, width = len(train.targets), len(train.features.rows[0])
        _require(type(counts) is dict and set(counts) == set(lock["count_fields"]) and
                 all(type(v) is int and v == n for v in counts.values()), "cpu_smoke_count_mismatch")
        counts = request["counts"] = dict(counts)
        _require(n <= lock["max_train_rows"] and len(query.rows) <= lock["max_query_rows"] and
                 width <= lock["max_columns"] and len(query.rows[0]) == width, "cpu_smoke_shape_invalid")
        _require(train.features.schema_sha256 == query.schema_sha256 and
                 not set(train.features.row_ids) & set(query.row_ids), "cpu_smoke_input_mismatch")
        _require(all(_finite(v) for row in (*train.features.rows, *query.rows) for v in row) and
                 all(_finite(y) for y in train.targets), "cpu_smoke_nonfinite_input")
        x, y, q = (np.asarray(values, dtype=np.float64) for values in
                   (train.features.rows, train.targets, query.rows))
        classes = lock["class_order"][task]
        if classes:
            _require(set(y.tolist()) == set(range(len(classes))), "cpu_smoke_class_mismatch")
        model = _cpu_smoke_model(name, task, parameters, request["seed"], lock)
        with threadpool_limits(limits=1):
            model.fit(x, y)
            values = model.predict_proba(q) if classes else np.asarray(model.predict(q)).reshape(len(q), 1)
        values = np.asarray(values, dtype=np.float64)
        _require(values.shape == (len(q), len(classes) or 1) and bool(np.isfinite(values).all()), "cpu_smoke_output_invalid")
        if classes:
            codes = np.asarray(model.classes_)
            _require(codes.tolist() == list(range(len(classes))) and bool(((values >= 0) & (values <= 1)).all()) and
                     bool((np.abs(values.sum(axis=1) - 1) <= 1e-7).all()), "cpu_smoke_probability_invalid")
            values = np.clip(values, lock["probability_clip"], 1 - lock["probability_clip"])
            values /= values.sum(axis=1, keepdims=True)
        elapsed, memory = time.monotonic() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        _require(elapsed <= lock["resources"]["wall_seconds"] and memory <= lock["resources"]["memory_bytes"], "cpu_smoke_budget_exceeded")
        pins = {k: v for k, v in request.items() if k not in {"training", "query"}}
        pins.update(training_sha256=train.sha256, query_sha256=query.sha256,
                    implementation_sha256=file_sha256(__file__))
        return {"format": "dope-cpu-smoke-response-v1", "qualification_scope": "cpu_smoke_only", "qualified": False,
                "request_sha256": digest(pins), "counts": dict(counts), "row_ids": query.row_ids,
                "class_order": tuple(classes), "kind": "probability" if classes else "point",
                "values": output_rows(values, len(q), len(classes) or 1),
                "usage": {"wall_seconds": elapsed, "memory_bytes": memory, "gpu_bytes": 0}}
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise KeyboardInterrupt() from None
        raise AdapterError("cpu_smoke_refused") from None
