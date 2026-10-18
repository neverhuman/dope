"""Bounded offline challenger contracts; fixture evidence grants no native claim.

W00 owns transport/isolation. This module uses the existing feature-only ABI,
requires full context conservation, and never acquires packages or checkpoints.
Native admission additionally needs an exact closure and custodian evidence.
"""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re

from research.transfer.adapters.base import (
    AdapterUnavailable, FeatureBatch, TrainingBatch, _count, _finite, _hash, _require,
)
from research.transfer.adapters.auditors.conventional import (
    ObservedLearner, OfflineRuntime, immutable_parameter, port_source_sha256,
    recipe_digest, validate_task,
)
from research.transfer.evidence import digest


PACKAGES = {"catboost": "catboost", "lightgbm": "lightgbm", "xgboost": "xgboost",
            "realmlp": "pytabkit", "knn": "scikit-learn", "kernel": "scikit-learn"}
PARAMETERS = {"catboost": {"iterations": 500, "depth": 8},
              "lightgbm": {"n_estimators": 500, "num_leaves": 63},
              "xgboost": {"n_estimators": 500, "max_depth": 8},
              "realmlp": {"n_epochs": 256}, "knn": {"n_neighbors": 31},
              "kernel": {"C": 100.0}}
LEGACY_KEYS = {"name", "status", "identity_sha256", "eligibility_sha256",
               "native_reference_sha256", "license_status"}
METADATA_KEYS = {"pretraining_corpus_disclosure", "qualification_state"}
PANEL_KEYS = {"panel_edition", "architecture_family", "checkpoint_sha256",
              "teacher_lineage_overlap", "roles_by_workload", "rights_status"}
PANEL_ROLE_KEYS = {"compiler_teacher", "selection", "required_confirmatory",
                   "challenger_only", "native_recipe_sensitivity"}
QUALIFICATIONS = {"fixture_verified_only", "native_runtime_qualified"}


def validate_disclosure(value):
    _require(type(value) is dict and set(value) == {"state", "caveat_required"},
             "invalid_pretraining_disclosure")
    state = value["state"]
    _require(type(state) is str and state in {
        "known_none", "overlapping", "unknown", "not_applicable"}, "invalid_pretraining_disclosure")
    _require(type(value["caveat_required"]) is bool, "invalid_pretraining_disclosure")
    _require(state not in {"unknown", "overlapping"} or value["caveat_required"],
             "pretraining_caveat_required")


def validate_model_registry(value):
    """Validate CR-10 metadata and return a copy for the closed prior reader.

    Callers must retain the original metadata with scientific claim evidence.
    The projected registry supplies execution admission only.
    """
    _require(type(value) is dict and set(value) == {"format", "kind", "entries"}, "invalid_registry")
    _require(value["format"] == "dope-adapter-registry-v1" and value["kind"] == "learner",
             "invalid_registry")
    _require(type(value["entries"]) is list and bool(value["entries"]), "invalid_registry")
    names, entries = set(), []
    for entry in value["entries"]:
        _require(type(entry) is dict and set(entry) == LEGACY_KEYS | METADATA_KEYS | PANEL_KEYS,
                 "invalid_registry")
        name = entry["name"]
        _require(type(name) is str and re.fullmatch(r"[a-z0-9_]+", name) and name not in names,
                 "invalid_registry")
        names.add(name)
        validate_disclosure(entry["pretraining_corpus_disclosure"])
        state = entry["qualification_state"]
        _require(type(state) is str and state in QUALIFICATIONS, "invalid_qualification")
        _require(type(entry["status"]) is str and entry["status"] in {
            "unresolved", "reviewed", "unavailable"}, "invalid_registry")
        _require(type(entry["license_status"]) is str and entry["license_status"] in {
            "unresolved", "compatible", "incompatible"}, "invalid_registry")
        for key in LEGACY_KEYS - {"name", "status", "license_status"}:
            if entry[key] is not None:
                _hash(entry[key])
            _require(entry["status"] != "reviewed" or entry[key] is not None, "invalid_registry")
        _require(state != "native_runtime_qualified" or
                 (entry["status"] == "reviewed" and entry["license_status"] == "compatible"),
                 "native_evidence_required")
        _require(type(entry["panel_edition"]) is str and bool(entry["panel_edition"]) and
                 type(entry["architecture_family"]) is str and bool(entry["architecture_family"]) and
                 type(entry["teacher_lineage_overlap"]) is str and
                 entry["teacher_lineage_overlap"] in {"known_none", "overlapping", "unknown"} and
                 type(entry["rights_status"]) is str and
                 entry["rights_status"] in {"reviewed", "unresolved", "restricted"}, "invalid_registry")
        if entry["checkpoint_sha256"] is not None:
            _hash(entry["checkpoint_sha256"])
        roles = entry["roles_by_workload"]
        _require(type(roles) is dict and bool(roles), "invalid_registry")
        for role_flags in roles.values():
            _require(type(role_flags) is dict and set(role_flags) == PANEL_ROLE_KEYS and
                     all(type(flag) is bool for flag in role_flags.values()), "invalid_registry")
        entries.append({key: entry[key] for key in sorted(LEGACY_KEYS)})
    return {"format": value["format"], "kind": "learner", "entries": entries}


def _package(value):
    _require(type(value) is dict and set(value) == {
        "name", "version", "artifact", "sha256", "url"}, "invalid_package_pin")
    for key in ("name", "artifact", "url"):
        _require(type(value[key]) is str and bool(value[key]), "invalid_package_pin")
    _require(type(value["version"]) is str and re.fullmatch(r"\d+(?:\.\d+)+(?:\+[a-z0-9.]+)?",
                                                          value["version"]), "unpinned_runtime")
    _hash(value["sha256"])


def _entry(entry):
    keys = {"name", "family_group", "tasks", "mode", "precision", "package", "checkpoint_sha256",
            "dependency_closure", "parameter_grid", "budget", "seed_schedule",
            "qualification_state", "native_evidence", "pretraining_corpus_disclosure"}
    _require(type(entry) is dict and set(entry) == keys, "invalid_challenger_lock")
    name = entry["name"]
    _require(type(name) is str and name in PACKAGES, "invalid_family")
    _require(type(entry["family_group"]) is str and bool(entry["family_group"]), "invalid_family")
    _require(entry["tasks"] == ["regression", "classification"] and entry["mode"] == "scratch",
             "unsupported_task")
    precision = "float32" if name in {"catboost", "realmlp"} else "float64"
    _require(entry["precision"] == precision and entry["checkpoint_sha256"] is None,
             "checkpoint_pin_mismatch")
    _package(entry["package"])
    _require(entry["package"]["name"] == PACKAGES[name], "primary_package_mismatch")
    validate_disclosure(entry["pretraining_corpus_disclosure"])
    _require(entry["pretraining_corpus_disclosure"]["state"] == "known_none",
             "scratch_pretraining_mismatch")
    seeds, grid, budget = entry["seed_schedule"], entry["parameter_grid"], entry["budget"]
    _require(type(seeds) is list and 1 <= len(seeds) <= 3, "invalid_seed_schedule")
    for seed in seeds:
        _count(seed)
    _require(len(set(seeds)) == len(seeds), "invalid_seed_schedule")
    _require(type(grid) is list and 1 <= len(grid) <= 8, "invalid_parameter_grid")
    for parameters in grid:
        _require(type(parameters) is dict and set(parameters) == set(PARAMETERS[name]),
                 "invalid_parameter_grid")
        for key, maximum in PARAMETERS[name].items():
            v = parameters[key]
            _require(_finite(v) and 0 < v <= maximum and (key == "C" or type(v) is int),
                     "invalid_parameter_grid")
    _require(len({digest(p) for p in grid}) == len(grid), "invalid_parameter_grid")
    _require(type(budget) is dict and set(budget) == {"max_candidates", "max_total_fits",
        "wall_seconds", "memory_bytes", "gpu_bytes", "threads"}, "invalid_budget")
    for key in budget:
        _count(budget[key])
    _require(budget["max_candidates"] == len(grid) and
             budget["max_total_fits"] == len(grid) * len(seeds) and
             0 < budget["wall_seconds"] <= 600 and 0 < budget["memory_bytes"] <= 2**31 and
             budget["gpu_bytes"] == 0 and budget["threads"] == 1, "invalid_budget")
    closure = entry["dependency_closure"]
    if closure is not None:
        _require(type(closure) is list and bool(closure), "invalid_dependency_closure")
        for package in closure:
            _package(package)
        _require(entry["package"] in closure and len({p["name"] for p in closure}) == len(closure),
                 "invalid_dependency_closure")
    state, evidence = entry["qualification_state"], entry["native_evidence"]
    _require(type(state) is str and state in QUALIFICATIONS, "invalid_qualification")
    if state == "fixture_verified_only":
        _require(evidence is None, "invalid_qualification")
    else:
        _require(closure is not None and type(evidence) is dict and set(evidence) == {
            "identity_sha256", "dependency_closure_sha256", "native_parity_sha256",
            "resource_sha256", "license_review_sha256", "contamination_review_sha256"},
                 "native_evidence_required")
        for pin in evidence.values():
            _hash(pin)
        _require(evidence["dependency_closure_sha256"] == digest(closure), "closure_pin_mismatch")


def validate_challenger_lock(value):
    """Validate the new closed section while retaining historical lock records."""
    _require(type(value) is dict and value.get("format") == "dope-other-auditors-lock-v1",
             "invalid_challenger_lock")
    for key in ("mitra", "tabicl", "conventional"):
        section = value.get(key)
        _require(type(section) is dict and "pretraining_corpus_disclosure" in section and
                 section.get("qualification_state") == "fixture_verified_only", "invalid_challenger_lock")
        validate_disclosure(section["pretraining_corpus_disclosure"])
    checkpoints = value["mitra"].get("checkpoints")
    _require(type(checkpoints) is dict and set(checkpoints) == {"regression", "classification"},
             "invalid_challenger_lock")
    for checkpoint in [*checkpoints.values(), value["tabicl"].get("checkpoint")]:
        _require(type(checkpoint) is dict and "pretraining_corpus_disclosure" in checkpoint,
                 "invalid_challenger_lock")
        validate_disclosure(checkpoint["pretraining_corpus_disclosure"])
    panel = value.get("challengers")
    _require(type(panel) is dict and set(panel) == {
        "format", "entries", "excluded_families", "package_pin_verification"} and
        panel["format"] == "dope-challenger-lock-v1" and
        panel["package_pin_verification"] == "public_index_metadata_only", "invalid_challenger_lock")
    _require(type(panel["entries"]) is list and bool(panel["entries"]), "invalid_challenger_lock")
    names = []
    for entry in panel["entries"]:
        _entry(entry)
        names.append(entry["name"])
    _require(len(set(names)) == len(names) and set(names) == set(PACKAGES), "invalid_challenger_lock")
    _require(type(panel["excluded_families"]) is dict and
             set(panel["excluded_families"]) == {"resnet_tabular", "survival_competing_risks"} and
             all(type(v) is str and bool(v) for v in panel["excluded_families"].values()),
             "invalid_challenger_lock")
    return json.loads(json.dumps(value, allow_nan=False))


def load_challenger_lock():
    try:
        path = Path(__file__).resolve().parents[2] / "locks/other-auditors.json"
        return validate_challenger_lock(json.loads(path.read_bytes()))
    except (OSError, ValueError):
        raise AdapterUnavailable("challenger_lock_unavailable") from None


@dataclass(frozen=True, repr=False)
class ChallengerRecipe:
    family: str
    task: str
    parameters: tuple
    class_order: tuple = ()
    precision: str = "float64"

    def __post_init__(self):
        validate_task(self.task, self.class_order)
        _require(type(self.family) is str and self.family in PACKAGES, "invalid_family")
        _require(type(self.parameters) is tuple, "invalid_parameters")
        for item in self.parameters:
            _require(type(item) is tuple and len(item) == 2 and type(item[0]) is str and
                     immutable_parameter(item[1]), "invalid_parameters")
        _require(len(dict(self.parameters)) == len(self.parameters), "invalid_parameters")

    @property
    def sha256(self):
        return recipe_digest({"format": "dope-challenger-recipe-v1",
                              "adapter_source_sha256": port_source_sha256(__file__), **asdict(self)})

    @property
    def preprocessing_sha256(self):
        return digest({"format": "dope-challenger-input-v1", "family": self.family,
                       "numeric_only": True, "target": "raw", "fit_only": True})

    @property
    def output_sha256(self):
        return recipe_digest({"format": "dope-challenger-output-v1", "task": self.task,
                              "class_order": self.class_order, "units": "raw"})


class ChallengerAdapter(ObservedLearner):
    """Identical locked recipe on real/synthetic contexts; no query-time fits.

    Factories are explicit fixtures. Package presence and successful fixture
    predictions never upgrade native qualification. Executor isolation and
    independent resource observation remain mandatory for native use.
    """
    def __init__(self, identity, admission, recipe, runtime, lock=None):
        lock = load_challenger_lock() if lock is None else validate_challenger_lock(lock)
        _require(type(recipe) is ChallengerRecipe, "invalid_recipe")
        entry = next(e for e in lock["challengers"]["entries"] if e["name"] == recipe.family)
        _require(identity.name == recipe.family and identity.mode == "scratch" and
                 identity.context_policy == "all" and identity.query_policy == "batch",
                 "challenger_policy_mismatch")
        _require(recipe.precision == entry["precision"] and
                 digest(dict(recipe.parameters)) in {digest(p) for p in entry["parameter_grid"]},
                 "recipe_lock_mismatch")
        _require(identity.preprocessing_sha256 == recipe.preprocessing_sha256 and
                 identity.output_aggregation_sha256 == recipe.output_sha256, "transform_pin_mismatch")
        budget = entry["budget"]
        _require(identity.resources.wall_seconds <= budget["wall_seconds"] and
                 identity.resources.memory_bytes <= budget["memory_bytes"] and
                 identity.resources.gpu_bytes == 0, "resource_budget_exceeded")
        _require(type(runtime) is OfflineRuntime, "invalid_runtime")
        self._entry, self._lock_sha256 = entry, digest(lock)
        self._fit_attempts = 0
        super().__init__(identity, admission, recipe, runtime)

    def _prepare_backend(self, training, seed, mode):
        _require(seed in self._entry["seed_schedule"], "seed_schedule_mismatch")
        _require(self._fit_attempts < self._entry["budget"]["max_total_fits"], "fit_budget_exceeded")
        self._fit_attempts += 1
        package = self._entry["package"]
        required = (package["name"], package["version"], package["sha256"])
        if required not in self.identity.dependencies or self.identity.code_sha256 != package["sha256"]:
            raise AdapterUnavailable("runtime_pin_mismatch")
        if self._runtime.factory is None:
            closure = self._entry["dependency_closure"]
            if closure is None:
                raise AdapterUnavailable("native_dependency_closure_unavailable")
            pins = {(p["name"], p["version"], p["sha256"]) for p in closure}
            if set(self.identity.dependencies) != pins:
                raise AdapterUnavailable("runtime_pin_mismatch")
        return self._quiet(super()._prepare_backend, training, seed, mode)

    def predict(self, query, seed):
        _require(type(query) is FeatureBatch, "query_features_required")
        _require(type(seed) is int and seed in self._entry["seed_schedule"], "seed_schedule_mismatch")
        return super().predict(query, seed)

    def _quiet(self, operation, *args):
        # The worker executor also owns native FD diagnostics; never publish them.
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            return operation(*args)

    def _fit_native(self, model, x, y):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return self._quiet(model.fit, x, y)

    def _raw_predict(self, x):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return self._quiet(super()._raw_predict, x)

    def receipt(self):
        receipt = super().receipt()
        n = len(self._training.targets)
        evidence = self._entry["native_evidence"]
        qualified = (self._runtime.factory is None and evidence is not None and
                     evidence["identity_sha256"] == self.identity.sha256)
        receipt.update(lock_sha256=self._lock_sha256, requested_rows=n, produced_rows=n,
                       provided_rows=self._provided_rows, consumed_rows=n, qualified=qualified,
                       fit_attempts=self._fit_attempts, row_count_scope="learner_fit_context",
                       qualification_state="native_runtime_qualified" if qualified else "fixture_verified_only",
                       pretraining_corpus_disclosure=dict(self._entry["pretraining_corpus_disclosure"]))
        return receipt

    def cache_key(self, request_sha256, seed, requested_rows):
        return digest({"invocation_sha256": super().cache_key(request_sha256, seed, requested_rows),
                       "lock_sha256": self._lock_sha256})

    def worker_request(self, query, seed, *, worker_image_sha256, prepared_sha256,
                       split_sha256, primary_n, sample_multiplier=1):
        """Build W00's bounded envelope after fit, without query labels.

        The custodian supplies image/prepared/split pins and independently
        validates responses with W00. Absence of that ABI fails closed. Scratch
        models use an explicit no-checkpoint commitment, never a weight claim.
        """
        self._ready()
        _require(type(query) is FeatureBatch, "query_features_required")
        _require(type(seed) is int and seed in self._entry["seed_schedule"], "seed_schedule_mismatch")
        _require(type(sample_multiplier) is int and sample_multiplier in {1, 4}, "invalid_sample_budget")
        for pin in (worker_image_sha256, prepared_sha256, split_sha256):
            _hash(pin)
        n = len(self._training.targets)
        _count(primary_n, 1)
        _require(n == primary_n * sample_multiplier, "row_count_mismatch")
        try:
            from research.transfer.adapters.worker_abi import build_request
        except ImportError:
            raise AdapterUnavailable("worker_abi_unavailable") from None
        training = self._training
        classification = self.recipe.task == "classification"
        if classification:
            training = TrainingBatch(training.features, tuple(self.recipe.class_order.index(y)
                                     for y in training.targets), training.role)
        pins = dict(model_sha256=self.identity.sha256, panel_sha256=self._lock_sha256,
                    worker_image_sha256=worker_image_sha256, prepared_sha256=prepared_sha256,
                    split_sha256=split_sha256, dependencies_sha256=recipe_digest(self.identity.dependencies),
                    checkpoint_sha256=digest({"format": "dope-scratch-checkpoint-v1", "weights": None}),
                    preprocessing_sha256=self.recipe.preprocessing_sha256, recipe_sha256=self.recipe.sha256,
                    runtime_version=self._entry["package"]["version"], precision=self.recipe.precision, device="cpu")
        resources = dict(threads=1, wall_ms=max(1, int(self.identity.resources.wall_seconds * 1000)),
                         memory_bytes=self.identity.resources.memory_bytes, vram_bytes=0,
                         disk_bytes=0, network=False)
        policy = dict(primary_n=primary_n, sample_multiplier=sample_multiplier,
                      context_cap=self.identity.context_cap, truncation="none", subsampling="none",
                      predict_time_fit=False, query_policy="batch", query_batch_size=self.identity.query_batch_size,
                      order_sensitive=self.identity.order_sensitive)
        output = dict(kind="probability" if classification else "point",
                      class_order=[digest({"class": c}) for c in self.recipe.class_order],
                      units="probability" if classification else "target_original")
        return build_request(pins, resources, policy, "predict", seed, training=training,
                             query=query, output=output, fit_seed=self._fit_seed)

    def _create_backend(self, seed):
        family, parameters = self.recipe.family, dict(self.recipe.parameters)
        classification = self.recipe.task == "classification"
        if self._runtime.factory is not None:
            return self._runtime.factory(family=family, task=self.recipe.task, seed=seed, parameters=parameters)
        if family == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor
            cls = CatBoostClassifier if classification else CatBoostRegressor
            return cls(**parameters, random_seed=seed, thread_count=1, verbose=False,
                       allow_writing_files=False, task_type="CPU")
        if family == "lightgbm":
            from lightgbm import LGBMClassifier, LGBMRegressor
            cls = LGBMClassifier if classification else LGBMRegressor
            return cls(**parameters, random_state=seed, n_jobs=1, verbosity=-1,
                       deterministic=True, force_col_wise=True, device_type="cpu")
        if family == "xgboost":
            from xgboost import XGBClassifier, XGBRegressor
            cls = XGBClassifier if classification else XGBRegressor
            return cls(**parameters, random_state=seed, n_jobs=1, verbosity=0,
                       tree_method="hist", device="cpu")
        if family == "realmlp":
            from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
            cls = RealMLP_TD_Classifier if classification else RealMLP_TD_Regressor
            return cls(**parameters, random_state=seed, n_threads=1, device="cpu", verbosity=0)
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if family == "knn":
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            cls = KNeighborsClassifier if classification else KNeighborsRegressor
            model = cls(**parameters, n_jobs=1)
        else:
            from sklearn.svm import SVC, SVR
            model = SVC(**parameters, kernel="rbf", probability=True, random_state=seed) if classification else SVR(**parameters, kernel="rbf")
        return make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), model)
