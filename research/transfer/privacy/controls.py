"""Bounded local shadow attackers, matched distance controls and attribute inference.

These consume explicitly supplied views. They fetch no data, fit no generator,
and grant no custody or release authority. The learned shadow classifier is a
baseline, with no claim of LiRA/DOMIAS implementation or author parity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

from .attacks import (AttackCalibration, DisclosurePolicy, DisclosureThreatModel,
                      PrivacyError, ScoreView, is_pin, number, pin, require)
from .release_profile import local_empirical_profile


@dataclass(frozen=True, slots=True, repr=False)
class RowView:
    rows: tuple
    ids: tuple
    units: tuple
    role: str

    def __post_init__(self):
        require(type(self.role) is str and self.role in {"auxiliary_train", "released_rows", "attack_training",
                                                       "attack_development", "attack_evaluation"}, "invalid_role")
        _matrix(self.rows)
        _identities(self.ids, self.units, len(self.rows))

    @property
    def sha256(self):
        return pin(asdict(self))


def _matrix(rows):
    require(type(rows) is tuple and 0 < len(rows) <= 100_000, "invalid_rows")
    require(all(type(row) is tuple for row in rows), "invalid_rows")
    width = len(rows[0])
    require(0 < width <= 64 and all(len(row) == width for row in rows), "invalid_width")
    require(all(number(x) and abs(x) <= 1e12 for row in rows for x in row), "invalid_rows")
    return np.asarray(rows, dtype=np.float64)


def _identities(ids, units, n):
    require(type(ids) is tuple and type(units) is tuple and len(ids) == len(units) == n, "invalid_ids")
    require(all(is_pin(x) for x in ids + units) and len(set(ids)) == n, "invalid_ids")


def _disjoint(left, right):
    require(not set(left.ids).intersection(right.ids), "attack_observation_overlap")
    require(not set(left.units).intersection(right.units), "attack_unit_overlap")


def _labels(labels, n):
    require(type(labels) is tuple and len(labels) == n and all(type(x) is bool for x in labels), "invalid_labels")
    require(any(labels) and not all(labels), "missing_class")


@dataclass(frozen=True, slots=True, repr=False)
class DistanceRecipe:
    kinds: tuple
    scales: tuple
    max_pairs: int = 1_000_000

    def __post_init__(self):
        require(type(self.kinds) is tuple and type(self.scales) is tuple and 0 < len(self.kinds) == len(self.scales) <= 64,
                "invalid_distance_recipe")
        require(all(type(k) is str and k in {"numeric", "nominal"} for k in self.kinds), "invalid_distance_recipe")
        require(all(number(x) and 1e-12 <= x <= 1e12 for x in self.scales), "invalid_distance_recipe")
        require(type(self.max_pairs) is int and 0 < self.max_pairs <= 10_000_000, "invalid_distance_budget")

    @property
    def sha256(self):
        return pin({"format": "dope-matched-nearest-v1", **asdict(self)})


def _nearest(query, reference, recipe):
    require(query.shape[1] == reference.shape[1] == len(recipe.kinds), "invalid_width")
    require(len(query) * len(reference) <= recipe.max_pairs, "distance_budget_exceeded")
    nearest = []
    # One query at a time bounds workspace independently of the pair count.
    for row in query:
        distances = np.zeros(len(reference), dtype=np.float64)
        for col, (kind, scale) in enumerate(zip(recipe.kinds, recipe.scales)):
            distances += ((reference[:, col] - row[col]) / scale) ** 2 if kind == "numeric" else (reference[:, col] != row[col])
        nearest.append(float(np.sqrt(np.min(distances) / len(recipe.kinds))))
    return tuple(nearest)


@dataclass(frozen=True, slots=True, repr=False)
class FeatureView:
    rows: tuple
    ids: tuple
    units: tuple
    role: str
    feature_recipe_sha256: str
    input_sha256: str

    def __post_init__(self):
        require(type(self.role) is str and self.role in {"attack_training", "attack_development", "attack_evaluation"}, "invalid_role")
        _matrix(self.rows)
        _identities(self.ids, self.units, len(self.rows))
        require(is_pin(self.feature_recipe_sha256) and is_pin(self.input_sha256), "invalid_feature_identity")

    @property
    def sha256(self):
        return pin(asdict(self))


def membership_features(query, released, auxiliary, recipe, sample_budget, reference_rows):
    """Type-aware nearest distances and matched population-distance ratio.

    The fixed 1e-12 floor handles exact copies. This is a nearest-distance
    diagnostic; it is not a fitted population-density attack.
    """
    require(type(query) is RowView and query.role in {"attack_training", "attack_development", "attack_evaluation"}, "invalid_query")
    require(type(released) is RowView and released.role == "released_rows", "released_rows_required")
    require(type(auxiliary) is RowView and auxiliary.role == "auxiliary_train", "auxiliary_required")
    require(type(recipe) is DistanceRecipe, "invalid_distance_recipe")
    require(type(sample_budget) is int and type(reference_rows) is int and sample_budget == reference_rows,
            "unmatched_reference_counts")
    require(len(released.rows) == sample_budget and len(auxiliary.rows) == reference_rows, "sample_budget_mismatch")
    _disjoint(query, auxiliary)
    q, syn, aux = _matrix(query.rows), _matrix(released.rows), _matrix(auxiliary.rows)
    ds, dr = _nearest(q, syn, recipe), _nearest(q, aux, recipe)
    rows = tuple((-math.log(max(s, 1e-12)), math.log(max(r, 1e-12)) - math.log(max(s, 1e-12)))
                 for s, r in zip(ds, dr))
    identity = pin({"query": query.sha256, "released": released.sha256, "auxiliary": auxiliary.sha256,
                    "recipe": recipe.sha256, "sample_budget": sample_budget, "reference_rows": reference_rows})
    return FeatureView(rows, query.ids, query.units, query.role, recipe.sha256, identity)


@dataclass(frozen=True, slots=True, repr=False)
class ShadowMembershipAttack:
    coefficients: tuple
    intercept: float
    center: tuple
    scale: tuple
    training_ids: tuple
    training_units: tuple
    training_sha256: str
    feature_recipe_sha256: str
    recipe_sha256: str
    runtime: str

    def __post_init__(self):
        require(all(type(x) is tuple for x in (self.coefficients, self.center, self.scale)), "invalid_shadow_state")
        require(0 < len(self.coefficients) == len(self.center) == len(self.scale) <= 64, "invalid_shadow_state")
        require(all(number(x) for x in self.coefficients + self.center + self.scale + (self.intercept,))
                and all(x > 0 for x in self.scale), "invalid_shadow_state")
        _identities(self.training_ids, self.training_units, len(self.training_ids))
        require(all(is_pin(x) for x in (self.training_sha256, self.feature_recipe_sha256, self.recipe_sha256)),
                "invalid_shadow_state")
        require(type(self.runtime) is str and 0 < len(self.runtime) <= 64, "invalid_shadow_state")

    @classmethod
    def fit(cls, training, membership, recipe_sha256):
        require(type(training) is FeatureView and training.role == "attack_training", "shadow_training_required")
        require(len(training.rows) <= 10_000, "shadow_fit_budget_exceeded")
        require(is_pin(recipe_sha256), "invalid_recipe")
        _labels(membership, len(training.rows))
        x = _matrix(training.rows)
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale == 0] = 1.
        try:
            with threadpool_limits(limits=1), warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model = LogisticRegression(C=1., solver="liblinear", max_iter=200, tol=1e-8, random_state=0)
                model.fit((x - center) / scale, membership)
        except (ValueError, ArithmeticError, ConvergenceWarning):
            raise PrivacyError("shadow_fit_failed") from None
        coefficients = tuple(float(v) for v in model.coef_[0])
        intercept = float(model.intercept_[0])
        require(all(number(v) for v in coefficients + (intercept,)), "shadow_fit_failed")
        training_hash = pin({"features": training.sha256, "memberships": membership})
        return cls(coefficients, intercept, tuple(float(v) for v in center), tuple(float(v) for v in scale),
                   training.ids, training.units, training_hash, training.feature_recipe_sha256,
                   recipe_sha256, sklearn.__version__)

    @property
    def sha256(self):
        return pin({"format": "dope-shadow-logistic-v1", "C": 1., "solver": "liblinear", "iterations": 200,
                    "tol": 1e-8, "seed": 0, **asdict(self)})

    def scores(self, query):
        require(type(query) is FeatureView and query.role in {"attack_development", "attack_evaluation"}, "attack_query_required")
        require(query.feature_recipe_sha256 == self.feature_recipe_sha256, "feature_recipe_mismatch")
        require(not set(query.ids).intersection(self.training_ids), "shadow_observation_overlap")
        require(not set(query.units).intersection(self.training_units), "shadow_unit_overlap")
        x = _matrix(query.rows)
        require(x.shape[1] == len(self.coefficients), "invalid_width")
        values = ((x - self.center) / self.scale) @ np.asarray(self.coefficients) + self.intercept
        require(np.isfinite(values).all(), "invalid_scores")
        return tuple(float(v) for v in values)


def _score_view(features, membership, scores, recipe):
    _labels(membership, len(features.rows))
    groups = [[i for i, value in enumerate(membership) if value == wanted] for wanted in (True, False)]
    members, nonmembers = groups
    return ScoreView(tuple(scores[i] for i in members), tuple(scores[i] for i in nonmembers),
                     tuple(features.ids[i] for i in members), tuple(features.ids[i] for i in nonmembers),
                     tuple(features.units[i] for group in groups for i in group), features.role, recipe, features.input_sha256)


def run_shadow_mia(training, training_membership, development, development_membership,
                   evaluation, evaluation_membership, recipe_sha256, policy, threat):
    """Fit only on shadow features; freeze direction/threshold before final scores."""
    require(type(development) is FeatureView and development.role == "attack_development", "development_required")
    require(type(evaluation) is FeatureView and evaluation.role == "attack_evaluation", "evaluation_required")
    require(type(policy) is DisclosurePolicy, "invalid_policy")
    require(type(threat) is DisclosureThreatModel, "invalid_threat")
    _disjoint(development, evaluation)
    require(type(training) is FeatureView and training.role == "attack_training", "shadow_training_required")
    _disjoint(training, development)
    _disjoint(training, evaluation)
    require(training.feature_recipe_sha256 == development.feature_recipe_sha256 == evaluation.feature_recipe_sha256,
            "feature_recipe_mismatch")
    require(len(training.rows[0]) == len(development.rows[0]) == len(evaluation.rows[0]), "invalid_width")
    _labels(development_membership, len(development.rows))
    _labels(evaluation_membership, len(evaluation.rows))
    model = ShadowMembershipAttack.fit(training, training_membership, recipe_sha256)
    dev = _score_view(development, development_membership, model.scores(development), model.sha256)
    calibration = AttackCalibration.freeze(dev, policy)
    final = _score_view(evaluation, evaluation_membership, model.scores(evaluation), model.sha256)
    return calibration.evaluate(final, threat)


def control_check(positive, negative, minimum_detection, maximum_null_advantage):
    require(number(minimum_detection) and 0 < minimum_detection < 1, "invalid_control_margin")
    require(number(maximum_null_advantage) and 0 < maximum_null_advantage < 1, "invalid_control_margin")
    matched = (positive["members"] == negative["members"] and positive["nonmembers"] == negative["nonmembers"]
               and positive["policy_sha256"] == negative["policy_sha256"]
               and positive["recipe_sha256"] == negative["recipe_sha256"])
    resolved = matched and positive["tpr_bounds"] is not None and negative["advantage_bounds"] is not None
    powerful = resolved and positive["tpr_bounds"][0] >= minimum_detection
    clean_null = resolved and negative["advantage_bounds"][1] <= maximum_null_advantage
    return {"matched": bool(matched), "positive_detected": bool(powerful), "negative_compatible": bool(clean_null),
            "status": "inconclusive", "reason": "control_provenance_unqualified",
            "release_profile": local_empirical_profile()}


def attribute_increment(artifact, auxiliary, artifact_capacity, auxiliary_capacity, artifact_aux, baseline_aux):
    require(all(is_pin(x) for x in (artifact_capacity, auxiliary_capacity, artifact_aux, baseline_aux)), "invalid_attribute_identity")
    require(artifact_capacity == auxiliary_capacity and artifact_aux == baseline_aux, "unmatched_attribute_baseline")
    require(type(artifact) is tuple and type(auxiliary) is tuple and len(artifact) == len(auxiliary) and bool(artifact), "invalid_attribute_data")
    require(all(type(x) is bool for x in artifact + auxiliary), "invalid_attribute_data")
    return sum(int(a) - int(b) for a, b in zip(artifact, auxiliary)) / len(artifact)


def attribute_inference(released, auxiliary, evaluation, hidden, recipe):
    """Matched ridge attackers predict a hidden numeric column from the others.

    Both have the same regularization, public units, reference count and query
    features. Evaluation truth is read only after both fits/predictions.
    """
    require(type(released) is RowView and released.role == "released_rows", "released_rows_required")
    require(type(auxiliary) is RowView and auxiliary.role == "auxiliary_train", "auxiliary_required")
    require(type(evaluation) is RowView and evaluation.role == "attack_evaluation", "evaluation_required")
    require(type(recipe) is DistanceRecipe and all(k == "numeric" for k in recipe.kinds), "numeric_attribute_required")
    require(len(released.rows) == len(auxiliary.rows) <= 10_000, "unmatched_reference_counts")
    _disjoint(auxiliary, evaluation)
    s, a, q = _matrix(released.rows), _matrix(auxiliary.rows), _matrix(evaluation.rows)
    require(s.shape[1] == a.shape[1] == q.shape[1] == len(recipe.kinds), "invalid_width")
    require(type(hidden) is int and 0 <= hidden < q.shape[1] and q.shape[1] > 1, "invalid_hidden_column")
    columns = [i for i in range(q.shape[1]) if i != hidden]
    scales = np.asarray(recipe.scales)
    predictions = []
    try:
        with threadpool_limits(limits=1):
            for reference in (s, a):
                model = Ridge(alpha=1., solver="cholesky")
                model.fit(reference[:, columns] / scales[columns], reference[:, hidden] / scales[hidden])
                predictions.append(model.predict(q[:, columns] / scales[columns]) * scales[hidden])
    except (ValueError, ArithmeticError):
        raise PrivacyError("attribute_fit_failed") from None
    losses = [float(np.mean((q[:, hidden] - prediction) ** 2)) for prediction in predictions]
    require(all(number(x) for x in losses), "attribute_fit_failed")
    return {"format": "dope-attribute-measurement", "version": 1, "status": "inconclusive",
            "reason": "attribute_custody_and_margin_unqualified", "recipe_sha256": recipe.sha256,
            "capacity_sha256": pin({"algorithm": "ridge", "alpha": 1., "solver": "cholesky", "runtime": sklearn.__version__}),
            "auxiliary_sha256": auxiliary.sha256, "released_sha256": released.sha256,
            "evaluation_sha256": evaluation.sha256, "reference_rows": len(a), "queries": len(q),
            "independent_units": len(set(evaluation.units)), "artifact_mse": losses[0], "auxiliary_mse": losses[1],
            "incremental_mse_improvement": losses[1] - losses[0], "uncertainty": None,
            "formal_dp": "unverified", "legal_route": "unverified", "release_authorized": False,
            "release_profile": local_empirical_profile()}


def holdout_nndr(released, fit_reference, holdout_reference, recipe):
    """Matched-size release-to-fit / release-to-independent-holdout distances.

    Zero holdout distance has no identified ratio. Exact duplicates and natural
    discreteness are exposed as counts, without changing historical copy gates.
    """
    require(type(released) is RowView and released.role == "released_rows", "released_rows_required")
    require(type(fit_reference) is RowView and fit_reference.role == "attack_training", "fit_reference_required")
    require(type(holdout_reference) is RowView and holdout_reference.role == "attack_evaluation", "holdout_reference_required")
    require(type(recipe) is DistanceRecipe, "invalid_distance_recipe")
    require(len(fit_reference.rows) == len(holdout_reference.rows), "unmatched_reference_counts")
    _disjoint(fit_reference, holdout_reference)
    rows = _matrix(released.rows)
    fit = _nearest(rows, _matrix(fit_reference.rows), recipe)
    holdout = _nearest(rows, _matrix(holdout_reference.rows), recipe)
    ratios = [a / b for a, b in zip(fit, holdout) if b > 0]
    require(all(number(x) for x in ratios), "distance_ratio_unrepresentable")
    return {"format": "dope-holdout-nndr", "version": 1, "status": "inconclusive",
            "reason": "distance_control_custody_unqualified", "recipe_sha256": recipe.sha256,
            "released_sha256": released.sha256, "fit_sha256": fit_reference.sha256,
            "holdout_sha256": holdout_reference.sha256,
            "sample_rows": len(rows), "reference_rows": len(fit_reference.rows),
            "exact_fit_matches": sum(x == 0 for x in fit), "zero_holdout_distances": sum(x == 0 for x in holdout),
            "identified_ratios": len(ratios), "median_ratio": float(np.median(ratios)) if ratios else None,
            "formal_dp": "unverified", "legal_route": "unverified", "release_authorized": False,
            "release_profile": local_empirical_profile()}
