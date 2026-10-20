"""Explicit adapter binding; imports SDKs only through existing native adapters."""
from dataclasses import dataclass, is_dataclass
from importlib import metadata
import inspect
from pathlib import Path
import sys

from research.transfer.adapters.base import AdapterAdmission, ModelIdentity, LearnerAdapter
from research.transfer.adapters.auditors.conventional import OfflineRuntime, file_sha256
from .pilot_learning_contract import LEARNERS, identity, pin, require
from .pilot_native_seed import OWNER_GRANT_SHA256, ORIGINAL_STREAM_HEX

NAMES = {"kumo_tabular": ("kumo_tabular", "frozen_context"),
         "tabicl": ("tabicl_v2", "frozen_context"), "mitra": ("mitra_v2", "fine_tuning"),
         "gradient_boosting": ("independent_boosting", "scratch"),
         "neural": ("independent_neural", "scratch"),
         "null_reference": ("regression_mean_null", "scratch")}


@dataclass(frozen=True, repr=False)
class Registration:
    role: str
    model: ModelIdentity
    admission: AdapterAdmission
    recipe: object
    runtime: OfflineRuntime | None
    runtime_sha256: str
    operation_qualification_sha256: str
    module_pins: tuple  # (exact loaded module name, expected source bytes SHA-256)
    fixture_factory: object = None

    def __post_init__(self):
        require(self.role in (*LEARNERS, "null_reference") and type(self.model) is ModelIdentity)
        require(type(self.admission) is AdapterAdmission and self.admission.kind == "learner")
        require(self.admission.identity_sha256 == self.model.sha256)
        require((self.model.name, self.model.mode) == NAMES[self.role])
        require(is_dataclass(self.recipe) and self.recipe.__dataclass_params__.frozen)
        require(self.recipe.sha256 == self.model.recipe_sha256)
        pin(self.runtime_sha256)
        pin(self.operation_qualification_sha256)
        require(type(self.module_pins) is tuple and bool(self.module_pins))
        require(len({name for name, _ in self.module_pins}) == len(self.module_pins))
        for name, value in self.module_pins:
            require(type(name) is str and bool(name))
            pin(value)

    @property
    def sha256(self):
        return identity({"role": self.role, "model": self.model.sha256,
                         "admission": (self.admission.eligibility_sha256, self.admission.native_reference_sha256),
                         "recipe": self.recipe.sha256, "runtime": self.runtime_sha256,
                         "qualification": self.operation_qualification_sha256,
                         "module_pins": self.module_pins, "fixture": self.fixture_factory is not None})

    def build(self, fixture):
        if fixture:
            require(callable(self.fixture_factory), "fixture_adapter_required")
            result = self.fixture_factory(self.model, self.admission)
            require(isinstance(result, LearnerAdapter) and result.identity.sha256 == self.model.sha256)
            return result
        require(self.fixture_factory is None and type(self.runtime) is OfflineRuntime)
        require(self.runtime.factory is None and self.runtime.version_reader is metadata.version)
        self.runtime.verify(self.model)
        # Verify installed exact distribution RECORD bytes, not only an archive or version string.
        for package, version, expected in self.model.dependencies:
            dist = metadata.distribution(package)
            records = [entry for entry in (dist.files or ()) if str(entry).endswith(".dist-info/RECORD")]
            require(dist.version == version and len(records) == 1, "installed_runtime_unavailable")
            require(file_sha256(dist.locate_file(records[0])) == expected, "installed_runtime_changed")
        if self.role == "kumo_tabular":
            from research.transfer.adapters.auditors.kumo_tabular import KumoTabularAdapter as Adapter
        elif self.role == "tabicl":
            from research.transfer.adapters.auditors.tabicl import TabICLAdapter as Adapter
        elif self.role == "mitra":
            from research.transfer.adapters.auditors.mitra import MitraAdapter as Adapter
            require(self.model.query_policy == "transductive" and self.recipe.task == "regression")
        elif self.role == "null_reference":
            from .pilot_constant_null import ConstantNullAdapter as Adapter
        else:
            from research.transfer.adapters.auditors.conventional import ConventionalAdapter as Adapter
            require(self.recipe.family == {"gradient_boosting": "histogram_boosting", "neural": "mlp"}[self.role])
            require(self.model.weights_sha256 is None and self.recipe.selection_sha256 is not None)
        require(self.recipe.task == "regression")
        sdk = {"kumo_tabular": "sdm", "tabicl": "tabicl", "mitra": "mitra_finetune",
               "gradient_boosting": "sklearn", "neural": "sklearn", "null_reference": "sklearn.dummy"}[self.role]
        require({sdk, Adapter.__module__} <= set(dict(self.module_pins)), "runtime_origin_pins_missing")
        return Adapter(self.model, self.admission, self.recipe, self.runtime)

    def origins(self, adapter, fixture):
        module = inspect.getmodule(type(adapter))
        require(module is not None and getattr(module, "__file__", None))
        names = dict(self.module_pins)
        if fixture:
            names = {module.__name__: file_sha256(module.__file__)}
        else:
            require(module.__name__ in names)
        result = []
        for name, expected in names.items():
            loaded = sys.modules.get(name)
            require(loaded is not None and getattr(loaded, "__file__", None), "module_origin_missing")
            actual = file_sha256(loaded.__file__)
            require(actual == expected, "module_origin_changed")
            result.append({"module": name, "origin": str(Path(loaded.__file__).absolute()), "sha256": actual})
        return result  # Private only. Paths never appear in redacted stage summaries.


@dataclass(frozen=True)
class OperationAdmission:
    kind: str  # fixture or development, never confirmation
    plan_sha256: str
    registry_sha256: str
    owner_receipt_sha256: str
    resource_receipt_sha256: str
    development_exposed: tuple
    null_reference_sha256: str

    def __post_init__(self):
        require(self.kind in ("fixture", "development"))
        require(self.development_exposed == LEARNERS)
        for name in ("plan_sha256", "registry_sha256", "owner_receipt_sha256", "resource_receipt_sha256", "null_reference_sha256"):
            pin(getattr(self, name))


def registry_digest(registry):
    require(type(registry) is dict and set(registry) == set(LEARNERS), "missing_required_learner")
    require(all(type(registry[name]) is Registration and registry[name].role == name for name in LEARNERS))
    return identity({name: registry[name].sha256 for name in LEARNERS})


def authorize(plan, registry, null_reference, admission, verify_operation):
    require(type(admission) is OperationAdmission and callable(verify_operation))
    require(admission.plan_sha256 == plan.sha256 and admission.registry_sha256 == registry_digest(registry))
    require(type(null_reference) is Registration and null_reference.role == "null_reference",
            "null_reference_required")
    require(plan.null_reference_sha256 == admission.null_reference_sha256 == null_reference.sha256,
            "null_reference_binding_mismatch")
    entries = (*registry.values(), null_reference)
    require(all(entry.model.environment_sha256 == plan.environment_sha256 for entry in entries))
    fixture = admission.kind == "fixture"
    require(all((entry.fixture_factory is not None) == fixture for entry in entries))
    if not fixture:
        require(plan.owner_recipe_decision_sha256 == OWNER_GRANT_SHA256 and
                plan.native_seed.original_stream_hex == ORIGINAL_STREAM_HEX, "owner_fixed_inputs_mismatch")
        fixed = {
            "gradient_boosting": {"max_iter": 64, "max_leaf_nodes": 15, "min_samples_leaf": 10,
                                  "l2_regularization": 1.0, "loss": "squared_error"},
            "neural": {"hidden_layer_sizes": (64, 32), "activation": "relu", "solver": "adam",
                       "alpha": 0.0001, "batch_size": 32, "learning_rate_init": 0.001,
                       "max_iter": 100, "tol": 0.0, "n_iter_no_change": 101, "shuffle": True}}
        for role in fixed:
            require(identity(dict(registry[role].recipe.parameters)) == identity(fixed[role]),
                    "fixed_owner_recipe_parameters_mismatch")
        for role in ("gradient_boosting", "neural"):
            require(registry[role].recipe.selection_sha256 == plan.owner_recipe_decision_sha256,
                    "fixed_owner_recipe_decision_mismatch")
    # This REQUIRED trusted-executor callback verifies actual owner/resource/role evidence.
    # Hashes and dataclass construction are not authority and are not a fallback verifier.
    require(verify_operation(admission) is None, "operation_not_admitted")
    return fixture
