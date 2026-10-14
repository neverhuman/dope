"""Permutation importance on a declared real query population.

Absolute target correlation is a historical proxy and is not this evidence.
Tied, undefined, and low-signal comparisons do not become perfect or zero
agreement. Importance is not a causal effect.
"""

from __future__ import annotations

import math

import numpy as np


def _columns(columns):
    columns = tuple(tuple(column) for column in columns)
    if not columns:
        raise ValueError("invalid query shape")
    width = len(columns[0])
    if width < 2 or any(len(column) != width for column in columns):
        raise ValueError("invalid query shape")
    return columns


def pearson(left, right):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("invalid association")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite association")
    if float(np.std(left)) == 0 or float(np.std(right)) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


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
    if type(seed) is not int or seed < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("invalid permutation budget")
    columns = _columns(columns)
    baseline = float(loss(columns))
    if not math.isfinite(baseline):
        raise ValueError("nonfinite loss")
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
            value = float(loss(tuple(permuted)))
            if not math.isfinite(value):
                raise ValueError("nonfinite loss")
            increases.append(value - baseline)
        scores[name] = float(np.mean(increases))
    return {"path": "drivers-v1", "evidence_kind": "permutation_importance",
            "learner_id": learner_id, "query_population": query_population,
            "baseline_loss": baseline, "importance": scores}


def importance_agreement(left: dict, right: dict, *, low_signal: float = 1e-8) -> dict:
    keys = tuple(left["importance"])
    if keys != tuple(right["importance"]) or not keys:
        raise ValueError("importance groups differ")
    pair = ([left["importance"][key] for key in keys], [right["importance"][key] for key in keys])

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
        order = sorted(range(len(values)), key=values.__getitem__)
        result = [0] * len(values)
        for rank, index in enumerate(order):
            result[index] = rank
        return result

    left_rank, right_rank = ranks(pair[0]), ranks(pair[1])
    agreement = pearson(left_rank, right_rank)
    return {"status": "defined" if agreement is not None else "undefined", "agreement": agreement}
