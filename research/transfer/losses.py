"""Original-unit losses. Query evidence is private evaluator input, never a sidecar.

No target ranking or final-query scale fitting occurs here. Classification uses
its own explicit contract, rather than inheriting the regression contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real


class LossSpace(str, Enum):
    ORIGINAL_MSE = "original_mse"
    LOG_LOSS = "log_loss"
    BRIER = "brier_sum_over_classes"


class DegenerateScale(str, Enum):
    CONSTANT_OR_NEAR_CONSTANT = "constant_or_near_constant"


def finite_number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("expected a finite numeric value")
    return float(value)


def _mean(values) -> float:
    values = tuple(values)
    if not values:
        raise ValueError("empty loss evidence")
    try:
        return finite_number(math.fsum(v / len(values) for v in values))
    except OverflowError:
        raise ValueError("nonfinite loss mean") from None


@dataclass(frozen=True)
class TrainingScale:
    value: float | None
    status: DegenerateScale | None = None

    def __post_init__(self):
        if self.value is None:
            if not isinstance(self.status, DegenerateScale):
                raise ValueError("undefined scale requires a disposition")
        elif finite_number(self.value) <= 0 or self.status is not None:
            raise ValueError("scale must be positive and nondegenerate")


def training_variance(fit_targets, *, minimum_variance: float = 0.0) -> TrainingScale:
    """Population MSE scale from fit rows; threshold must be declared in raw units."""
    minimum_variance = finite_number(minimum_variance)
    if minimum_variance < 0:
        raise ValueError("negative scale threshold")
    targets = tuple(finite_number(v) for v in fit_targets)
    center = _mean(targets)
    try:
        variance = _mean((v - center) ** 2 for v in targets)
    except OverflowError:
        raise ValueError("nonfinite training scale") from None
    if variance <= minimum_variance:
        return TrainingScale(None, DegenerateScale.CONSTANT_OR_NEAR_CONSTANT)
    return TrainingScale(variance)


@dataclass(frozen=True)
class PairedLosses:
    """Exactly aligned query pairs; repr deliberately omits protected evidence."""

    query_ids: tuple[str, ...] = field(repr=False)
    cluster_ids: tuple[str, ...] = field(repr=False)
    real: tuple[float, ...] = field(repr=False)
    synthetic: tuple[float, ...] = field(repr=False)
    null: tuple[float, ...] = field(repr=False)
    loss_space: LossSpace = LossSpace.ORIGINAL_MSE

    def __post_init__(self):
        for name in ("query_ids", "cluster_ids", "real", "synthetic", "null"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.query_ids)
        if not n or any(len(v) != n for v in (self.cluster_ids, self.real, self.synthetic, self.null)):
            raise ValueError("query evidence lengths differ or are empty")
        if len(set(self.query_ids)) != n or any(not isinstance(v, str) or not v for v in (*self.query_ids, *self.cluster_ids)):
            raise ValueError("invalid or duplicate query identity")
        if not isinstance(self.loss_space, LossSpace):
            raise ValueError("unsupported loss space")
        for name in ("real", "synthetic", "null"):
            values = tuple(finite_number(v) for v in getattr(self, name))
            if any(v < 0 for v in values):
                raise ValueError("losses must be nonnegative")
            object.__setattr__(self, name, values)

    @classmethod
    def regression(cls, query_ids, cluster_ids, raw_targets, real_predictions,
                   synthetic_predictions, null_predictions, *, loss_space=LossSpace.ORIGINAL_MSE):
        if loss_space != LossSpace.ORIGINAL_MSE:
            raise ValueError("regression requires original-unit MSE")
        targets = tuple(finite_number(v) for v in raw_targets)
        losses = []
        for prediction in (real_predictions, synthetic_predictions, null_predictions):
            prediction = tuple(finite_number(v) for v in prediction)
            if len(prediction) != len(targets):
                raise ValueError("prediction lengths differ")
            try:
                losses.append(tuple(finite_number((y - p) ** 2) for y, p in zip(targets, prediction)))
            except OverflowError:
                raise ValueError("nonfinite regression loss") from None
        return cls(tuple(query_ids), tuple(cluster_ids), *losses)

    @property
    def means(self) -> tuple[float, float, float]:
        return tuple(_mean(v) for v in (self.real, self.synthetic, self.null))

    def regression_errors(self) -> dict:
        if self.loss_space != LossSpace.ORIGINAL_MSE:
            raise ValueError("regression summary requires MSE")
        return {name: {"mse": _mean(v), "rmse": math.sqrt(_mean(v)),
                       "mae": _mean(math.sqrt(x) for x in v)}
                for name, v in zip(("real", "synthetic", "null"), (self.real, self.synthetic, self.null))}


@dataclass(frozen=True)
class ClassificationContract:
    classes: tuple[str, ...]
    loss_space: LossSpace
    probability_floor: float

    def __post_init__(self):
        object.__setattr__(self, "classes", tuple(self.classes))
        if any(not isinstance(c, str) or not c for c in self.classes) or len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("class mapping must be unique")
        if self.loss_space not in (LossSpace.LOG_LOSS, LossSpace.BRIER):
            raise ValueError("classification loss contract required")
        if not 0 < finite_number(self.probability_floor) < 1 / len(self.classes):
            raise ValueError("invalid probability floor")

    def losses(self, labels, probabilities) -> tuple[float, ...]:
        labels, probabilities = tuple(labels), tuple(probabilities)
        if len(labels) != len(probabilities) or not labels:
            raise ValueError("classification evidence lengths differ")
        result = []
        for label, row in zip(labels, probabilities):
            if label not in self.classes:
                raise ValueError("class outside frozen mapping")
            row = tuple(finite_number(p) for p in row)
            if len(row) != len(self.classes) or any(not 0 <= p <= 1 for p in row) or not math.isclose(math.fsum(row), 1, abs_tol=1e-12, rel_tol=0):
                raise ValueError("invalid class probabilities")
            index = self.classes.index(label)
            if self.loss_space == LossSpace.LOG_LOSS:
                # Floor all classes then renormalize, a convention frozen here.
                floored = tuple(max(p, self.probability_floor) for p in row)
                result.append(-math.log(floored[index] / math.fsum(floored)))
            else:
                result.append(math.fsum((p - int(i == index)) ** 2 for i, p in enumerate(row)))
        return tuple(result)

    def paired(self, query_ids, cluster_ids, labels, real, synthetic, null) -> PairedLosses:
        labels = tuple(labels)
        return PairedLosses(tuple(query_ids), tuple(cluster_ids), self.losses(labels, real),
                            self.losses(labels, synthetic), self.losses(labels, null), self.loss_space)
