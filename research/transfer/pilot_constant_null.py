"""Fit-only original-unit regression null using the pinned native SDK."""
from dataclasses import asdict, dataclass
from research.transfer.adapters.base import _require
from research.transfer.adapters.auditors.conventional import (
    ObservedLearner, port_source_sha256, recipe_digest, require_native_dependency,
)


@dataclass(frozen=True)
class ConstantNullRecipe:
    task: str = "regression"
    strategy: str = "mean"
    class_order: tuple = ()
    precision: str = "float64"

    def __post_init__(self):
        _require((self.task, self.strategy, self.class_order, self.precision) ==
                 ("regression", "mean", (), "float64"), "invalid_constant_null_recipe")

    @property
    def sha256(self):
        return recipe_digest({"format": "dope-constant-null-recipe-v1", **asdict(self),
                              "adapter_source_sha256": port_source_sha256(__file__)})

    @property
    def preprocessing_sha256(self):
        return recipe_digest({"format": "dope-constant-null-input-v1", "fit_only": True,
                              "features": "numeric_missing_nan", "targets": "original_real_context"})

    @property
    def output_sha256(self):
        return recipe_digest({"format": "dope-constant-null-output-v1", "units": "raw", "kind": "point"})


class ConstantNullAdapter(ObservedLearner):
    def __init__(self, identity, admission, recipe, runtime):
        _require(type(recipe) is ConstantNullRecipe, "invalid_constant_null_recipe")
        _require(identity.name == "regression_mean_null" and identity.mode == "scratch" and
                 identity.weights_sha256 is None and identity.context_policy == "all" and
                 identity.query_policy == "batch", "invalid_constant_null_identity")
        _require(identity.preprocessing_sha256 == recipe.preprocessing_sha256 and
                 identity.output_aggregation_sha256 == recipe.output_sha256, "transform_pin_mismatch")
        super().__init__(identity, admission, recipe, runtime)

    def _create_backend(self, seed):
        # The fixture path is admitted only by a fixture Registration. Production
        # Registration rejects factories and verifies exact runtime/module bytes.
        if self._runtime.factory is not None:
            return self._runtime.factory(strategy="mean")
        require_native_dependency(self.identity, "scikit-learn", "1.7.2")
        from sklearn.dummy import DummyRegressor
        return DummyRegressor(strategy="mean")
