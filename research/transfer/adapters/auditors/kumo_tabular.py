"""Pinned NVIDIA SDM Kumo-Tabular; raw targets and original-unit outputs.

The offline loader mirrors the pinned SDK's checkpoint assignment using public
PyTorch state loading. An architecture is constructed on meta, and strict
verified weight loading is mandatory before use. Missing weights produce typed
unavailability. Official checkpoint parity/license admission remains a separate
custodian gate.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from research.transfer.adapters.base import PredictionResult, _count, _require
from research.transfer.adapters.auditors.conventional import (
    ObservedLearner, output_rows, validate_task, require_native_dependency,
    port_source_sha256, recipe_digest as digest,
)


CODE_REVISION = "d50589692dbbbad4852eb1d9627f3b40c2673b63"
QUANTILES = tuple(f"q{i:03d}" for i in range(1, 1000))
LOCK = Path(__file__).resolve().parents[2] / "locks/kumo-tabular.json"


@dataclass(frozen=True, repr=False)
class KumoRecipe:
    task: str
    size: str = "large"
    estimators: int = 16
    estimator_batch_size: int = 1
    output: str = "quantile_grid_mean"
    class_order: tuple = ()
    device: str = "cpu"
    precision: str = "float32"

    def __post_init__(self):
        validate_task(self.task, self.class_order)
        _require(self.size in {"small", "medium", "large"}, "invalid_model_size")
        _count(self.estimators, 1)
        _count(self.estimator_batch_size, 1)
        _require(self.estimator_batch_size <= self.estimators, "invalid_ensemble_batch")
        _require(self.device in {"cpu", "cuda"} and self.precision == "float32", "unsupported_device_precision")
        _require(self.output in ({"probability"} if self.task == "classification" else
                                {"quantiles", "quantile_grid_mean", "median"}), "invalid_output_recipe")

    @property
    def sha256(self):
        return digest({"format": "dope-kumo-recipe-v1", "sdk_revision": CODE_REVISION,
                       "adapter_source_sha256": port_source_sha256(__file__),
                       "native_recipe": "KumoTabular.default_recipe", **asdict(self)})

    @property
    def preprocessing_sha256(self):
        return digest({"format": "dope-kumo-input-v1", "features": "numeric_missing_nan",
                       "target": "original_units_or_declared_class_codes",
                       "regression_target_dtype": "float32_native_head",
                       "fit_only": True, "sdk_revision": CODE_REVISION, "feature_cap": 500})

    @property
    def output_sha256(self):
        return digest({"format": "dope-kumo-output-v1", "output": self.output,
                       "grid": QUANTILES if self.task == "regression" else self.class_order,
                       "inverse": "sdk_per_estimator_before_aggregation",
                       "absent_class": "refuse"})


class KumoTabularAdapter(ObservedLearner):
    def __init__(self, identity, admission, recipe, runtime):
        _require(type(recipe) is KumoRecipe, "invalid_recipe")
        _require(identity.name == "kumo_tabular" and identity.mode == "frozen_context", "mode_mismatch")
        _require(identity.code_revision == CODE_REVISION, "code_revision_mismatch")
        _require(identity.preprocessing_sha256 == recipe.preprocessing_sha256 and
                 identity.output_aggregation_sha256 == recipe.output_sha256, "transform_pin_mismatch")
        _require(identity.order_sensitive, "order_policy_mismatch")
        super().__init__(identity, admission, recipe, runtime)

    def prepare_context(self, training, seed):
        self.prepare(training, seed)

    def predict_original_units(self, query, seed):
        return self.predict(query, seed)

    def _create_backend(self, seed):
        kwargs = dict(task=self.recipe.task, size=self.recipe.size, device=self.recipe.device)
        self._context_seed = seed
        if self._runtime.factory is not None:
            return self._runtime.factory(**kwargs)
        require_native_dependency(self.identity, "structured-data-models")
        require_native_dependency(self.identity, "torch")
        pins = json.loads(LOCK.read_text())
        task_file = "classifier.pt" if self.recipe.task == "classification" else "regressor.pt"
        expected = pins["checkpoint"]["files"][f"{self.recipe.size}/{task_file}"]["sha256"]
        _require(self.identity.weights_sha256 == expected, "checkpoint_task_mismatch")
        _require(not self._runtime.artifacts.ancillary, "unexpected_artifacts")
        import torch
        from sdm import Task
        from sdm.models import KumoTabular
        task = Task(self.recipe.task)
        model = KumoTabular(task=task, size=self.recipe.size, pretrained=False, device="meta")
        state = torch.load(self._runtime.artifacts.weights, map_location=self.recipe.device, weights_only=True)
        model.models[task].load_state_dict(state, strict=True, assign=True)
        model.eval()
        return model

    def _fit_native(self, model, x, y):
        # The native recipe silently selects 500 columns. Refuse wider views so
        # this port never advertises a full-feature reference after truncation.
        _require(x.shape[1] <= 500, "native_feature_cap_exceeded")
        if self._runtime.factory is not None:
            generator = self._context_seed
            target = y.reshape(-1, 1)
        else:
            import torch
            from sdm import Stype, TableTensor
            x = torch.as_tensor(x, dtype=torch.float64, device=self.recipe.device)
            if self.recipe.task == "classification":
                import pandas as pd
                target = TableTensor.from_pandas(pd.DataFrame({"target": y}),
                                                  {"target": Stype.categorical}, device=self.recipe.device)
            else:
                # The pinned regression y_lin head has float32 weights. Raw
                # units are preserved, while its declared precision is applied
                # before the native fit-only target recipe.
                target = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.recipe.device)
                _require(bool(torch.isfinite(target).all()), "native_target_precision_exceeded")
            generator = torch.Generator(device=self.recipe.device).manual_seed(self._context_seed)
        model.fit(x=x, y=target, num_estimators=self.recipe.estimators,
                  estimator_batch_size=self.recipe.estimator_batch_size, generator=generator)

    def _predict_backend(self, query, seed):
        from research.transfer.adapters.auditors.conventional import numeric_features
        x = numeric_features(query.rows)
        if self._runtime.factory is None:
            import torch
            from sdm import Stype
            x = torch.as_tensor(x, dtype=torch.float64, device=self.recipe.device)
            native = self._model.predict(x=x)
            columns = tuple(native.columns[Stype.numerical])
            values = native.numerical.detach().cpu().numpy()
        else:
            # Executor test/reference factories expose the same named output.
            columns, values = self._model.predict(x=x)
            columns = tuple(columns)
        count, n = len(self._training.targets), len(query.rows)
        usage = self._runtime.observe("predict", count)
        if self.recipe.task == "classification":
            expected = tuple(str(i) for i in range(len(self.recipe.class_order)))
            _require(len(columns) == len(expected) and set(columns) == set(expected), "native_class_mismatch")
            values = output_rows(values, n, len(expected))
            values = tuple(tuple(row[columns.index(c)] for c in expected) for row in values)
            kind, order = "probability", self.recipe.class_order
        else:
            _require(columns == QUANTILES, "native_quantile_grid_mismatch")
            values = output_rows(values, n, 999)
            _require(all(all(a <= b for a, b in zip(row, row[1:])) for row in values), "invalid_quantile_order")
            if self.recipe.output == "quantile_grid_mean":
                # A declared finite-grid summary, not an exact integral mean.
                values = tuple((math.fsum(row) / len(row),) for row in values)
            elif self.recipe.output == "median":
                values = tuple((row[499],) for row in values)
            kind = "distribution" if self.recipe.output == "quantiles" else "point"
            order = ()
        return PredictionResult(query.row_ids, values, kind, count, usage.memory_bytes,
                                usage.gpu_bytes, order, embedding_unavailable="unsupported")
