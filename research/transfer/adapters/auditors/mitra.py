"""Mitra-v2: frozen classification and explicitly charged regression tuning.

Stock AutoGluon regression has a scalar head incompatible with v2's 1,000-bin
checkpoint. The official mitra-finetune route performs one bagged fit per
prediction call. It is fine_tuning/transductive, never frozen_context. Point
predictions are its native raw-bin mean; no inverse-after-mean approximation.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

from research.transfer.adapters.base import AdapterUnavailable, _count, _hash, _require
from research.transfer.adapters.auditors.conventional import (
    ObservedLearner, validate_task, require_native_dependency, port_source_sha256, recipe_digest as digest,
)


AUTOGLUON_REVISION = "152343642ce605c95333b2b1e27e5d47675a5c13"
FINETUNE_REVISION = "b4701e8148dc33b00ed15d7086ff59816957cde4"
LOCK = Path(__file__).resolve().parents[2] / "locks/other-auditors.json"


@dataclass(frozen=True, repr=False)
class MitraRecipe:
    task: str
    config_sha256: str
    estimators: int = 1
    bag_folds: int = 8
    class_order: tuple = ()
    device: str = "cpu"
    precision: str = "bfloat16"

    def __post_init__(self):
        validate_task(self.task, self.class_order)
        _hash(self.config_sha256)
        _count(self.estimators, 1)
        _require(self.bag_folds == 8 and self.device in {"cpu", "cuda"} and
                 self.precision == "bfloat16", "unsupported_native_recipe")
        _require(self.task != "regression" or self.estimators == 1, "unsupported_native_recipe")
        _require(len(self.class_order) <= 10, "native_class_cap_exceeded")

    @property
    def code_revision(self):
        return AUTOGLUON_REVISION if self.task == "classification" else FINETUNE_REVISION

    @property
    def sha256(self):
        return digest({"format": "dope-mitra-recipe-v1", "revision": self.code_revision,
                       "adapter_source_sha256": port_source_sha256(__file__),
                       "autogluon_revision": AUTOGLUON_REVISION, "fine_tune_steps": 0 if self.task == "classification" else 50,
                       "regression_route": "MitraFinetune.predict_mean_one_bagged_fit_per_call",
                       "in_process": True, "validation": "native_training_only",
                       "classification_shuffle_classes": True, "classification_shuffle_features": True,
                       "classification_random_transforms": False, "classification_random_mirrors": False,
                       **asdict(self)})

    @property
    def preprocessing_sha256(self):
        return digest({"format": "dope-mitra-input-v1", "revision": self.code_revision,
                       "features": "numeric_missing_nan", "target": "raw_or_contiguous_class_codes",
                       "support_cap": 8192, "fit_only": True})

    @property
    def output_sha256(self):
        return digest({"format": "dope-mitra-output-v1", "task": self.task,
                       "class_order": self.class_order, "absent_class": "refuse",
                       "units": "native_raw_bin_mean_or_probability"})


class MitraAdapter(ObservedLearner):
    def __init__(self, identity, admission, recipe, runtime):
        _require(type(recipe) is MitraRecipe and identity.name == "mitra_v2", "invalid_recipe")
        if recipe.task == "regression" and identity.mode == "frozen_context":
            raise AdapterUnavailable("frozen_v2_regression_unavailable")
        mode = "frozen_context" if recipe.task == "classification" else "fine_tuning"
        _require(identity.mode == mode and identity.code_revision == recipe.code_revision, "mode_revision_mismatch")
        _require(identity.context_cap <= 8192 and identity.query_batch_size <= 1024, "native_budget_exceeded")
        _require(identity.order_sensitive, "order_policy_mismatch")
        _require(recipe.task != "regression" or identity.query_policy == "transductive", "query_policy_mismatch")
        _require(identity.preprocessing_sha256 == recipe.preprocessing_sha256 and
                 identity.output_aggregation_sha256 == recipe.output_sha256, "transform_pin_mismatch")
        super().__init__(identity, admission, recipe, runtime)
        self._attempts = []

    @property
    def attempts(self):
        return tuple(dict(a) for a in self._attempts)

    def _create_backend(self, seed):
        # The pinned finetune package reads numerous MITRA_* switches. Its
        # defaults are code-pinned; custom overrides need a separate recipe.
        if any(name.startswith("MITRA_") for name in os.environ):
            raise AdapterUnavailable("environment_override_unavailable")
        weights = self._runtime.artifacts.weights
        _require(weights is not None, "missing_weights")
        config = Path(weights).with_name("config.json")
        _require(self._runtime.artifacts.ancillary == ((config, self.recipe.config_sha256),), "config_pin_mismatch")
        if self.recipe.task == "classification":
            kwargs = dict(hf_model=str(Path(weights).parent), device=self.recipe.device,
                          n_estimators=self.recipe.estimators, fine_tune=False, fine_tune_steps=0,
                          shuffle_classes=True, shuffle_features=True, use_random_transforms=False,
                          random_mirror_regression=False, random_mirror_x=False, seed=seed, verbose=False)
        else:
            kwargs = dict(checkpoint_dir=str(Path(weights).parent), problem_type="regression",
                          device=self.recipe.device, random_state=seed, in_process=True,
                          num_bag_folds=self.recipe.bag_folds, time_limit=self.identity.resources.wall_seconds)
        if self._runtime.factory is not None:
            return self._runtime.factory(**kwargs)
        require_native_dependency(self.identity, "autogluon.tabular", "1.6.1")
        pins = json.loads(LOCK.read_text())["mitra"]["checkpoints"][self.recipe.task]
        _require(Path(weights).name == "model.safetensors" and
                 self.identity.weights_sha256 == pins["files"]["model.safetensors"]["sha256"], "checkpoint_task_mismatch")
        if self.recipe.task == "classification":
            from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
            return MitraClassifier(**kwargs)
        require_native_dependency(self.identity, "mitra-finetune")
        from mitra_finetune import MitraFinetune
        return MitraFinetune(**kwargs)

    def _fit_native(self, model, x, y):
        if self.recipe.task == "classification":
            model.fit(X=x, y=y, time_limit=self.identity.resources.wall_seconds)
        else:
            model.fit(x, y)

    def _native_classes(self):
        # Pinned MitraClassifier returns ascending contiguous target codes;
        # unlike sklearn classifiers it exposes no classes_ attribute.
        return tuple(range(len(self.recipe.class_order)))

    def _raw_predict(self, x):
        if self.recipe.task == "classification":
            return self._model.predict_proba(x)
        attempt = {"event": "predict", "requested_bagged_fits": 1,
                   "requested_child_fits": self.recipe.bag_folds, "status": "started"}
        self._attempts.append(attempt)
        try:
            result = self._model.predict(x, output_type="mean")
        except BaseException:
            attempt["status"] = "failed"
            raise
        attempt["status"] = "returned"
        return result
