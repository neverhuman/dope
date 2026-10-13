"""TabICLv2 public sklearn API, verified local checkpoints, no auto-download.

The native fitted encoder/scaler restores original target units. This port
adds no target normalization, context subsampling, or query reordering.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from research.transfer.adapters.base import _count, _require
from research.transfer.adapters.auditors.conventional import (
    ObservedLearner, validate_task, require_native_dependency, port_source_sha256, recipe_digest as digest,
)


CODE_REVISION = "10719e21fd72b8945d4a4f9ae077e350ba8650d9"
LOCK = Path(__file__).resolve().parents[2] / "locks/other-auditors.json"


@dataclass(frozen=True, repr=False)
class TabICLRecipe:
    task: str
    estimators: int = 8
    estimator_batch_size: int = 1
    norm_methods: tuple = ("none", "power")
    feature_shuffle: str = "latin"
    class_shuffle: str = "shift"
    kv_cache: bool = False
    class_order: tuple = ()
    device: str = "cpu"
    precision: str = "float32"

    def __post_init__(self):
        validate_task(self.task, self.class_order)
        _count(self.estimators, 1)
        _count(self.estimator_batch_size, 1)
        _require(self.estimator_batch_size <= self.estimators, "invalid_ensemble_batch")
        _require(type(self.norm_methods) is tuple and bool(self.norm_methods) and
                 set(self.norm_methods) <= {"none", "power", "quantile", "quantile_rtdl", "robust"}, "invalid_normalization")
        _require(self.feature_shuffle in {"latin", "shift", "random", "none"} and
                 self.class_shuffle in {"latin", "shift", "random", "none"}, "invalid_order_recipe")
        _require(type(self.kv_cache) is bool, "invalid_cache_recipe")
        _require(self.device in {"cpu", "cuda"} and self.precision == "float32", "unsupported_device_precision")

    @property
    def sha256(self):
        return digest({"format": "dope-tabicl-recipe-v1", "revision": CODE_REVISION,
                       "adapter_source_sha256": port_source_sha256(__file__),
                       "n_jobs": 1, "use_amp": False, "use_fa3": False, "offload_mode": False,
                       "allow_auto_download": False, "outlier_threshold": 4.0,
                       "softmax_temperature": 0.9, "average_logits": True,
                       "support_many_classes": True, **asdict(self)})

    @property
    def preprocessing_sha256(self):
        return digest({"format": "dope-tabicl-input-v1", "revision": CODE_REVISION,
                       "features": "numeric_missing_nan", "target": "raw_native_fit_only"})

    @property
    def output_sha256(self):
        return digest({"format": "dope-tabicl-output-v1", "task": self.task,
                       "class_order": self.class_order, "absent_class": "refuse",
                       "units": "native_original_units"})


class TabICLAdapter(ObservedLearner):
    def __init__(self, identity, admission, recipe, runtime):
        _require(type(recipe) is TabICLRecipe, "invalid_recipe")
        _require(identity.name == "tabicl_v2" and identity.mode == "frozen_context", "mode_mismatch")
        _require(identity.code_revision == CODE_REVISION, "code_revision_mismatch")
        _require(identity.preprocessing_sha256 == recipe.preprocessing_sha256 and
                 identity.output_aggregation_sha256 == recipe.output_sha256, "transform_pin_mismatch")
        _require(identity.order_sensitive, "order_policy_mismatch")
        super().__init__(identity, admission, recipe, runtime)

    def _create_backend(self, seed):
        name = "tabicl-classifier-v2-20260212.ckpt" if self.recipe.task == "classification" else "tabicl-regressor-v2-20260212.ckpt"
        kwargs = dict(n_estimators=self.recipe.estimators, norm_methods=list(self.recipe.norm_methods),
                      feat_shuffle_method=self.recipe.feature_shuffle, outlier_threshold=4.0,
                      batch_size=self.recipe.estimator_batch_size, kv_cache=self.recipe.kv_cache,
                      model_path=str(self._runtime.artifacts.weights), allow_auto_download=False,
                      checkpoint_version=name, device=self.recipe.device, use_amp=False,
                      use_fa3=False, offload_mode=False, n_jobs=1, random_state=seed, verbose=False)
        if self.recipe.task == "classification":
            kwargs.update(class_shuffle_method=self.recipe.class_shuffle, softmax_temperature=0.9,
                          average_logits=True, support_many_classes=True)
        if self._runtime.factory is not None:
            return self._runtime.factory(**kwargs)
        require_native_dependency(self.identity, "tabicl", "2.0.3")
        pins = json.loads(LOCK.read_text())["tabicl"]["checkpoint"]
        _require(self.identity.weights_sha256 == pins["files"][name]["sha256"], "checkpoint_task_mismatch")
        _require(not self._runtime.artifacts.ancillary, "unexpected_artifacts")
        from tabicl import TabICLClassifier, TabICLRegressor
        cls = TabICLClassifier if self.recipe.task == "classification" else TabICLRegressor
        return cls(**kwargs)

    def _fit_native(self, model, x, y):
        model.fit(x, y)
        _require(model.n_samples_in_ == len(y), "context_count_mismatch")
