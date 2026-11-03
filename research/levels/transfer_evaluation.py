"""Prospective transfer evaluation in one frozen real-FIT coordinate system.

The bridge uses frozen-v3 REAL-FIT feature preparation and an unclipped affine target
map. It is deliberately NOT NC1. Historical v3 replay remains in its runner.
Only hashes and structural metadata cross the public receipt boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable

import numpy as np

EVALUATION_PROTOCOL = "dope-dvs-v1"
LEGACY_REPLAY_PROTOCOL = "canonical-v3-historical-v1"
RECEIVER_PREPARATION = "legacy-real-fit-affine-target-v1"


class EvaluationInputError(ValueError):
    """Stable source-free public failure code."""


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise EvaluationInputError(code)


def _pin(value: Any) -> str:
    _require(type(value) is str and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "invalid_evaluation_pin")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _arrays(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        a, b = np.array(x, dtype=np.float64, copy=True), np.array(y, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError):
        raise EvaluationInputError("invalid_evaluation_matrix") from None
    _require(a.ndim == 2 and b.ndim == 1 and len(a) == len(b)
             and len(a) > 0 and a.shape[1] > 0, "invalid_evaluation_shape")
    _require(bool(np.isfinite(a).all() and np.isfinite(b).all()), "nonfinite_evaluation_matrix")
    a.setflags(write=False)
    b.setflags(write=False)
    return a, b


def _array_digest(x: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    for array in (x, y):
        h.update(json.dumps(array.shape).encode())
        h.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return h.hexdigest()


@dataclass(frozen=True)
class PreparedEvaluationView:
    space_id: str
    fit_scope: str
    feature_groups: tuple[int, ...]
    task: str
    role: str
    x: np.ndarray
    y: np.ndarray
    content_sha256: str
    class_count: int | None = None

    @classmethod
    def create(cls, x: Any, y: Any, *, space_id: str, fit_scope: str,
               feature_groups: tuple[int, ...], task: str, role: str,
               class_count: int | None = None) -> "PreparedEvaluationView":
        _pin(space_id)
        _pin(fit_scope)
        _require(task in ("regression", "classification", "multiclass"), "invalid_evaluation_task")
        _require(role in ("real_fit", "synthetic_fit", "real_test"), "invalid_evaluation_role")
        a, b = _arrays(x, y)
        _require(type(feature_groups) is tuple and len(feature_groups) == a.shape[1]
                 and all(type(g) is int and g >= 0 for g in feature_groups), "invalid_evaluation_groups")
        if task != "regression":
            _require(type(class_count) is int and class_count >= 2, "invalid_evaluation_classes")
            _require(bool(((b == np.rint(b)) & (b >= 0) & (b < class_count)).all()), "invalid_evaluation_classes")
        else:
            _require(class_count is None, "invalid_evaluation_classes")
        return cls(space_id, fit_scope, feature_groups, task, role, a, b, _array_digest(a, b), class_count)

    def validate(self) -> None:
        # Revalidate direct dataclass construction/replacement at the boundary.
        PreparedEvaluationView.create(self.x, self.y, space_id=self.space_id,
            fit_scope=self.fit_scope, feature_groups=self.feature_groups,
            task=self.task, role=self.role, class_count=self.class_count)
        _require(_array_digest(self.x, self.y) == self.content_sha256,
                 "evaluation_view_mutated")
        _require(bool(np.isfinite(self.x).all() and np.isfinite(self.y).all()),
                 "nonfinite_evaluation_matrix")


class FrozenRealFitReceiver:
    """One frozen-v3 REAL-FIT feature fit with an unclipped regression target map.

    The private fitted object is never returned or serialized publicly. The
    constructor copies its state, and every apply verifies the state identity.
    """
    def __init__(self, fitted: Any, y_fit: Any, *, task: str, fit_scope: str,
                 feature_groups: tuple[int, ...] | None = None):
        import copy
        self._fitted = copy.deepcopy(fitted)
        self.task, self.fit_scope = task, _pin(fit_scope)
        _require(task in ("regression", "classification", "multiclass"), "invalid_evaluation_task")
        self._target_lo, self._target_scale = 0.0, 1.0
        if task == "regression":
            try:
                y = np.asarray(y_fit, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                raise EvaluationInputError("invalid_evaluation_target") from None
            _require(y.ndim == 1 and len(y) > 0 and bool(np.isfinite(y).all()),
                     "invalid_evaluation_target")
            self._target_lo = float(np.min(y))
            scale = float(np.max(y) - self._target_lo)
            _require(np.isfinite(scale), "nonfinite_evaluation_target_range")
            self._target_scale = scale if scale > 0 else 1.0
        self.class_count = (None if task == "regression" else
                            (2 if task == "classification" else len(fitted.target_scaler.class_labels)))
        self.feature_groups = feature_groups or tuple(range(len(fitted.kinds)))
        self._state_sha256 = self._state_digest()
        self.space_id = _digest({"preparation": RECEIVER_PREPARATION,
                                 "state": self._state_sha256, "fit_scope": self.fit_scope,
                                 "task": task, "groups": self.feature_groups})

    def _state_digest(self) -> str:
        try:
            return _digest({"feature_transform": asdict(self._fitted),
                            "target_lo": self._target_lo, "target_scale": self._target_scale})
        except (TypeError, ValueError, OverflowError):
            raise EvaluationInputError("invalid_evaluation_transform") from None

    @classmethod
    def fit(cls, x_fit: Any, y_fit: Any, *, task: str, fit_scope: str) -> "FrozenRealFitReceiver":
        from .scaling import FitOnlyTableTransform
        _pin(fit_scope)
        try:
            fitted = FitOnlyTableTransform.fit(x_fit, y_fit, task=task)
        except (TypeError, ValueError, OverflowError, IndexError):
            raise EvaluationInputError("evaluation_fit_failed") from None
        return cls(fitted, y_fit, task=task, fit_scope=fit_scope)

    def prepare(self, x: Any, y: Any, *, role: str,
                input_space: str = "raw") -> PreparedEvaluationView:
        _require(input_space == "raw" and not isinstance(x, PreparedEvaluationView),
                 "evaluation_double_transform")
        _require(self._state_digest() == self._state_sha256, "evaluation_transform_mutated")
        try:
            a = self._fitted.transform_features(x)
            if self.task == "regression":
                b = (np.asarray(y, dtype=np.float64) - self._target_lo) / self._target_scale
            else:
                b = self._fitted.transform_target(y, strict_unknown=True)
        except (TypeError, ValueError, OverflowError, IndexError):
            raise EvaluationInputError("evaluation_apply_failed") from None
        return PreparedEvaluationView.create(a, b, space_id=self.space_id,
                                             fit_scope=self.fit_scope,
                                             feature_groups=self.feature_groups,
                                             task=self.task, role=role, class_count=self.class_count)

    def metadata(self) -> dict[str, Any]:
        return {"preparation": RECEIVER_PREPARATION, "space_id": self.space_id,
                "fit_scope": self.fit_scope, "feature_groups_sha256": _digest(self.feature_groups),
                "target_tail_policy": "unclipped_affine" if self.task == "regression" else "frozen_classes"}


def evaluate_prepared_transfer(real_fit: PreparedEvaluationView,
                               synth_fit: PreparedEvaluationView,
                               real_test: PreparedEvaluationView, *,
                               oracle: Callable[..., Any], scorer: Callable[..., Any],
                               protocol: str = EVALUATION_PROTOCOL) -> dict[str, Any]:
    """Score already-prepared arms without any fitting of common coordinates.

    Oracle/scorer own learner-local training. Neither callback receives raw rows
    or a common-transform fitting callback. NC1 callers construct views using
    their qualified card's space/FIT/group identities rather than this bridge.
    """
    _require(protocol == EVALUATION_PROTOCOL, "unsupported_evaluation_protocol")
    expected = ("real_fit", "synthetic_fit", "real_test")
    for view, role in zip((real_fit, synth_fit, real_test), expected):
        _require(isinstance(view, PreparedEvaluationView) and view.role == role,
                 "invalid_evaluation_role")
        view.validate()
        _require((view.space_id, view.fit_scope, view.feature_groups, view.task, view.class_count)
                 == (real_fit.space_id, real_fit.fit_scope, real_fit.feature_groups, real_fit.task, real_fit.class_count),
                 "evaluation_coordinate_mismatch")
    reference = oracle(real_fit.x, real_fit.y, real_test.x, real_test.y)
    result = scorer(reference, synth_fit.x, synth_fit.y, real_test.x, real_test.y)
    return {"protocol": protocol, "space_id": real_fit.space_id,
            "fit_scope": real_fit.fit_scope, "real_test_sha256": real_test.content_sha256,
            "result": result}
