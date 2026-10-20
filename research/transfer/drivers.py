"""Permutation importance on a declared real query population.

Absolute target correlation is a historical proxy and is not this evidence.
Tied, undefined, and low-signal comparisons do not become perfect or zero
agreement. Importance is not a causal effect.
"""

from __future__ import annotations

import math

import numpy as np


from .losses import finite_number


def positive_importances(values):
    """Truncate noisy negative loss increases before normalizing group mass."""
    from .tdl.common import require, vector
    values = vector(values)
    require(bool(values), "invalid_driver_evidence")
    return [max(0.0, value) for value in values]


def mass_agreement(real, other):
    from .tdl.common import require, sum_checked
    real, other = positive_importances(real), positive_importances(other)
    require(len(real) == len(other), "importance_shape_mismatch")
    rt, ot = sum_checked(real), sum_checked(other)
    require(rt > 0.0, "degenerate_real_importance")
    if ot == 0.0:
        return 0.0
    l1 = sum_checked([abs(r / rt - o / ot) for r, o in zip(real, other)])
    return min(1.0, max(0.0, 1.0 - 0.5 * l1))


def paired_margin(real_synthetic, real_real):
    """D80 endpoint on matched query/group/perturbation evidence, before inference."""
    from .tdl.common import unit
    return unit(real_synthetic) - 0.8 * unit(real_real)


def _columns(columns):
    columns = tuple(tuple(column) for column in columns)
    if not columns:
        raise ValueError("invalid query shape")
    width = len(columns[0])
    if width < 2 or any(len(column) != width for column in columns):
        raise ValueError("invalid query shape")
    for column in columns:
        for value in column:
            finite_number(value)
    return columns


def pearson(left, right):
    raw_left = np.asarray(left)
    raw_right = np.asarray(right)
    if (
        raw_left.ndim != 1
        or raw_right.ndim != 1
        or raw_left.shape != raw_right.shape
        or raw_left.size < 2
        or raw_left.dtype == bool
        or raw_right.dtype == bool
    ):
        raise ValueError("invalid association")
    left_list = raw_left.tolist()
    right_list = raw_right.tolist()
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (*left_list, *right_list)):
        raise ValueError("invalid association")
    left_arr = np.asarray(left_list, float)
    right_arr = np.asarray(right_list, float)
    if not np.isfinite(left_arr).all() or not np.isfinite(right_arr).all():
        raise ValueError("nonfinite association")
    left_std = float(np.std(left_arr))
    right_std = float(np.std(right_arr))
    if not math.isfinite(left_std) or not math.isfinite(right_std):
        raise ValueError("nonfinite association")
    if left_std == 0 or right_std == 0:
        return None
    corr = float(np.corrcoef(left_arr, right_arr)[0, 1])
    if not math.isfinite(corr):
        raise ValueError("nonfinite association")
    return max(-1.0, min(1.0, corr))


def ordinal_association(column, target, *, scale: str):
    if scale == "nominal":
        raise ValueError("no ordinal association for a nominal column")
    if scale != "ordinal":
        raise ValueError("unknown scale")
    return pearson(column, target)


def historical_target_correlation(column, target) -> dict:
    value = pearson(column, target)
    return {"path": "historical-proxy-v0", "evidence_kind": "absolute_target_correlation",
            "learner_id": None, "query_population": None,
            "value": None if value is None else abs(value)}


def permutation_importance(loss, columns, groups, *, learner_id: str, query_population: str,
                           seed: int, repeats: int = 16) -> dict:
    if not isinstance(learner_id, str) or not learner_id or not isinstance(query_population, str) or not query_population:
        raise ValueError("driver evidence requires a learner and query population")
    if type(seed) is not int or not 0 <= seed < (1 << 64) or type(repeats) is not int or repeats < 1:
        raise ValueError("invalid permutation budget")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("invalid logical group")
    columns = _columns(columns)
    baseline = finite_number(loss(columns))
    rng = np.random.default_rng(seed)
    scores = {}
    for name, indices in groups.items():
        indices = tuple(indices)
        if not isinstance(name, str) or not name or not indices or len(set(indices)) != len(indices):
            raise ValueError("invalid logical group")
        if any(type(index) is not int or index < 0 or index >= len(columns) for index in indices):
            raise ValueError("invalid logical group")
        increases = []
        for _ in range(repeats):
            order = rng.permutation(len(columns[0]))
            permuted = list(columns)
            for index in indices:
                permuted[index] = tuple(columns[index][int(row)] for row in order)
            value = finite_number(loss(tuple(permuted)))
            increases.append(finite_number(value - baseline))
        scores[name] = finite_number(float(np.mean(increases)))
    return {"path": "drivers-v1", "evidence_kind": "permutation_importance",
            "learner_id": learner_id, "query_population": query_population,
            "baseline_loss": baseline, "importance": scores}


def importance_agreement(left: dict, right: dict, *, low_signal: float = 1e-8) -> dict:
    if isinstance(low_signal, bool) or not isinstance(low_signal, (int, float)) or not math.isfinite(low_signal) or low_signal < 0:
        raise ValueError("invalid low_signal threshold")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("invalid driver evidence")
    for side in (left, right):
        path = side.get("path")
        kind = side.get("evidence_kind")
        if (path is not None and path != "drivers-v1") or (kind is not None and kind != "permutation_importance"):
            raise ValueError("historical proxy or non-permutation evidence is not driver agreement")
        if not isinstance(side.get("learner_id"), str) or not side["learner_id"]:
            raise ValueError("driver agreement requires learner_id")
        if not isinstance(side.get("query_population"), str) or not side["query_population"]:
            raise ValueError("driver agreement requires query_population")
        if not isinstance(side.get("importance"), dict):
            raise ValueError("invalid driver importance mapping")
    if left["query_population"] != right["query_population"]:
        raise ValueError("driver query populations differ")
    keys = tuple(left["importance"])
    if keys != tuple(right["importance"]) or not keys:
        raise ValueError("importance groups differ")
    pair = ([left["importance"][key] for key in keys], [right["importance"][key] for key in keys])
    for values in pair:
        for value in values:
            if isinstance(value, bool) or (value is not None and not isinstance(value, (int, float))):
                raise ValueError("invalid importance value")

    def state(values):
        if any(value is None or not math.isfinite(value) for value in values):
            return "undefined"
        if max(abs(value) for value in values) < low_signal:
            return "low_signal"
        if len(set(values)) < 2:
            return "tied"
        return "defined"

    left_state, right_state = state(pair[0]), state(pair[1])
    if left_state != "defined" or right_state != "defined":
        return {"status": left_state if left_state != "defined" else right_state, "agreement": None}

    def ranks(values):
        order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            avg_rank = (start + end - 1) / 2.0
            for idx in order[start:end]:
                result[idx] = avg_rank
            start = end
        return result

    left_rank, right_rank = ranks(pair[0]), ranks(pair[1])
    agreement = pearson(left_rank, right_rank)
    return {"status": "defined" if agreement is not None else "undefined", "agreement": agreement}
