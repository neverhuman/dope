"""Offline admission for TabSyn, TabDiff, and TabDDPM.

The vendor runtime is not executed. A fitted sample is a schema probe, and
every author schedule that does not fit the published budget stays visible.
"""

from dataclasses import dataclass
import hashlib
import json

from research.transfer.adapters.base import (
    AdapterError, GeneratorAdapter, ModelIdentity, ResourceLimits, ResourceUsage, SampleResult,
    TrainingBatch, FeatureBatch,
)
from research.transfer.evidence import digest


_LANES = frozenset({"resource_matched", "common_transfer", "unrestricted_quality"})
_CLASSES = {}


def _require(condition, code):
    if not condition:
        raise AdapterError(code)


@dataclass(frozen=True)
class NativeRecipe:
    """One named schedule. Scaled and author-full schedules stay distinct."""

    name: str
    schedule: tuple
    full_schedule: bool
    lane: str
    within_budget: bool
    selection: str
    sampling_dependencies: tuple

    def as_dict(self):
        return {
            "name": self.name,
            "schedule": {key: value for key, value in self.schedule},
            "full_schedule": self.full_schedule,
            "lane": self.lane,
            "within_budget": self.within_budget,
            "selection": self.selection,
            "sampling_dependencies": list(self.sampling_dependencies),
        }


def _recipe(name, raw):
    _require(type(raw) is dict, "invalid_recipe")
    schedule = raw.get("schedule")
    dependencies = raw.get("sampling_dependencies")
    _require(type(schedule) is dict and schedule, "invalid_recipe")
    _require(all(type(key) is str and type(value) is int for key, value in schedule.items()), "invalid_recipe")
    _require(type(raw.get("full_schedule")) is bool and raw.get("lane") in _LANES, "invalid_recipe")
    _require(type(raw.get("within_budget")) is bool and type(raw.get("selection")) is str, "invalid_recipe")
    _require(type(dependencies) is list and all(type(item) is str and item for item in dependencies), "invalid_recipe")
    return NativeRecipe(name, tuple(sorted(schedule.items())), raw["full_schedule"], raw["lane"],
                        raw["within_budget"], raw["selection"], tuple(dependencies))


def _identity(name, spec, recipe):
    resources = spec["resources"]
    code_revision = spec["code_revision"]
    return ModelIdentity(
        name=name, mode="scratch", code_revision=code_revision, code_sha256=spec["code_sha256"],
        weights_sha256=None,
        dependencies=(("author_source", code_revision, spec["code_sha256"]),),
        licenses=(tuple(spec["license"]),),
        preprocessing_sha256=hashlib.sha256(spec["native_transform"].encode()).hexdigest(),
        recipe_sha256=digest(recipe.as_dict()),
        output_aggregation_sha256=hashlib.sha256(b"synthetic rows match the fitted schema").hexdigest(),
        environment_sha256=hashlib.sha256(b"vendor runtime not executed").hexdigest(),
        precision=spec["precision"], context_cap=spec["context_cap"], context_policy="all",
        query_policy="batch", query_batch_size=1024, order_sensitive=True,
        determinism="seeded_best_effort",
        resources=ResourceLimits(resources["wall_seconds"], resources["memory_bytes"], resources["gpu_bytes"]))


def load_lock(text):
    """Read the generator lock. The returned object does not admit a model by itself."""
    raw = json.loads(text)
    _require(type(raw) is dict and raw.get("format") == "dope-neural-generator-lock-v1", "invalid_lock")
    generators = raw.get("generators")
    _require(type(generators) is dict and set(generators) == set(_CLASSES), "invalid_lock")
    loaded = {}
    for name, spec in generators.items():
        recipes = {recipe_name: _recipe(recipe_name, recipe) for recipe_name, recipe in spec["recipes"].items()}
        loaded[name] = {"spec": spec, "recipes": recipes,
                        "recorded_attempts": tuple(spec["recorded_attempts"])}
    return loaded


def recipe_identity(lock, name, recipe_name):
    """Identity of one locked recipe. The registry still has to admit it."""
    _require(name in lock and recipe_name in lock[name]["recipes"], "invalid_generator")
    entry = lock[name]
    return _identity(name, entry["spec"], entry["recipes"][recipe_name])


def neural_adapter(lock, name, recipe_name, admission):
    """Bind one locked recipe. Admission comes from the custodian, not this lock."""
    _require(name in _CLASSES and recipe_name in lock[name]["recipes"], "invalid_generator")
    entry = lock[name]
    recipe = entry["recipes"][recipe_name]
    identity = recipe_identity(lock, name, recipe_name)
    references = tuple(item for key, item in sorted(entry["recipes"].items()) if key != recipe_name)
    return _CLASSES[name](identity, admission, recipe, references, entry["recorded_attempts"])


class NeuralGenerator(GeneratorAdapter):
    """Schema, sample, export, and reload without starting the author process."""

    scaled_only = False

    def __init__(self, identity, admission, recipe, references, recorded_attempts):
        super().__init__(identity, admission)
        self.recipe = recipe
        self.references = references
        self.recorded_attempts = recorded_attempts
        self.attempts = []
        self._inference = None

    def _attempt(self, recipe, role, status):
        return {"generator": self.identity.name, "recipe": recipe.name, "role": role, "status": status,
                "full_schedule": recipe.full_schedule, "lane": recipe.lane,
                "within_budget": recipe.within_budget, "schedule": recipe.as_dict()["schedule"],
                "selection": recipe.selection, "sampling_dependencies": list(recipe.sampling_dependencies),
                "vendor_executed": False}

    def fit(self, training, seed):
        self.attempts = [dict(item, generator=self.identity.name, vendor_executed=False)
                         for item in self.recorded_attempts]
        self.attempts.append(self._attempt(self.recipe, "requested", "requested"))
        self.attempts.extend(self._attempt(recipe, "reference", "visible") for recipe in self.references)
        if self.scaled_only and self.recipe.full_schedule:
            raise AdapterError("scaled_called_full_schedule")
        if not self.recipe.within_budget:
            raise AdapterError("out_of_budget_reference")
        super().fit(training, seed)

    def _fit_backend(self, training, seed, mode):
        for item in self.attempts:
            if item.get("role") == "requested":
                item["status"] = "fitted"
        self._inference = self.export_state(training)
        return ResourceUsage(len(training.targets), 128, 0)

    def export_state(self, training):
        return {"format": "dope-neural-inference-state-v1", "vendor_executed": False,
                "generator": self.identity.name, "recipe": self.recipe.name,
                "full_schedule": self.recipe.full_schedule, "schedule": self.recipe.as_dict()["schedule"],
                "schema_sha256": training.features.schema_sha256,
                "width": len(training.features.rows[0]), "model_sha256": self.identity.sha256,
                "sampling_dependencies": list(self.recipe.sampling_dependencies)}

    def _sample_backend(self, rows, seed):
        state = self._inference
        width = state["width"]
        synthetic = []
        for index in range(rows):
            synthetic.append(hashlib.sha256(f"{self.identity.name}-{seed}-{index}".encode()).hexdigest())
        features = FeatureBatch(tuple(synthetic), tuple((0.0,) * width for _ in range(rows)), state["schema_sha256"])
        data = TrainingBatch(features, tuple(0.0 for _ in range(rows)), "synthetic_train")
        return SampleResult(data, ResourceUsage(len(self._training.targets), 128, 0))

    def export(self):
        self._ready()
        _require(self._inference is not None and self._inference["vendor_executed"] is False, "vendor_execution_unmeasured")
        return dict(self._inference)

    def reload(self, state):
        """Replace the inference probe. A scaled adapter still cannot carry a full schedule."""
        self._ready()
        _require(type(state) is dict and state.get("format") == "dope-neural-inference-state-v1", "invalid_state")
        _require(state.get("vendor_executed") is False, "vendor_execution_unmeasured")
        _require(state.get("generator") == self.identity.name and state.get("recipe") == self.recipe.name, "recipe_mismatch")
        _require(state.get("model_sha256") == self.identity.sha256, "recipe_mismatch")
        if self.scaled_only and state.get("full_schedule") is True:
            raise AdapterError("scaled_called_full_schedule")
        _require(state.get("schema_sha256") == self._training.features.schema_sha256, "schema_mismatch")
        _require(state.get("schedule") == self.recipe.as_dict()["schedule"], "recipe_mismatch")
        self._inference = dict(state)


class TabSynAdapter(NeuralGenerator):
    scaled_only = True


class TabDiffAdapter(NeuralGenerator):
    pass


class TabDDPMAdapter(NeuralGenerator):
    pass


_CLASSES.update(tabsyn=TabSynAdapter, tabdiff=TabDiffAdapter, tabddpm=TabDDPMAdapter)
