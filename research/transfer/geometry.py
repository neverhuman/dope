"""Separate geometry summaries. Closeness qualifies neither transfer nor privacy.

Kumo coordinates and DOPE coordinates are different spaces. A per-table refit
does not create a shared geometry claim. Unbiased MMD-squared is unclipped.
"""

from __future__ import annotations

from types import MappingProxyType
import math

import numpy as np


SPACES = ("row_activation", "pooled_embedding", "prediction_fingerprint", "refit_statistics", "action_representation")


def coordinate_distance(left, right, *, space: str, encoder_id: str, other_space: str | None = None,
                        other_encoder: str | None = None) -> float:
    if space not in SPACES:
        raise ValueError("unknown coordinate space")
    if not encoder_id:
        raise ValueError("missing encoder")
    if other_space is not None and other_space != space:
        raise ValueError("no direct distance across coordinate systems")
    if other_encoder is not None and other_encoder != encoder_id:
        raise ValueError("no direct distance across encoders")
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if left.shape != right.shape or left.size == 0 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("invalid coordinates")
    return float(np.linalg.norm(left - right))


def _rbf(left, right, sigma):
    diff = left[:, None, :] - right[None, :, :]
    return np.exp(-np.sum(diff * diff, axis=2) / (2 * sigma * sigma))


def mmd2_unbiased(left, right, *, sigma: float) -> float:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    sigma = float(sigma)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or min(left.shape[0], right.shape[0]) < 2:
        raise ValueError("invalid clouds")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("invalid kernel scale")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite cloud")
    left_kernel = _rbf(left, left, sigma)
    right_kernel = _rbf(right, right, sigma)
    cross = _rbf(left, right, sigma)
    np.fill_diagonal(left_kernel, 0.0)
    np.fill_diagonal(right_kernel, 0.0)
    m, n = left.shape[0], right.shape[0]
    return float(left_kernel.sum() / (m * (m - 1)) + right_kernel.sum() / (n * (n - 1)) - 2 * cross.mean())


def ecdf(values, *, space: str) -> dict:
    if space not in SPACES:
        raise ValueError("unknown coordinate space")
    values = tuple(sorted(float(value) for value in values))
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("invalid ecdf")
    n = len(values)
    return {"space": space, "values": values, "probabilities": tuple((index + 1) / n for index in range(n))}


def other_entity_row_count(reference, query, *, reference_entities, query_entities):
    """Count reference rows with a different entity. Coordinates are not distances.

    The result is not a nearest-neighbor statistic and not a privacy result.
    """
    reference = np.asarray(reference, float)
    query = np.asarray(query, float)
    reference_entities = tuple(reference_entities)
    query_entities = tuple(query_entities)
    if reference.shape[0] != len(reference_entities) or query.shape[0] != len(query_entities):
        raise ValueError("entity cardinality mismatch")
    if reference.shape[0] != query.shape[0] or reference.shape[1] != query.shape[1] or reference.shape[0] < 2:
        raise ValueError("unequal reference cardinality")
    same_rows = reference.shape == query.shape and np.array_equal(reference, query)
    counts = []
    for index, entity in enumerate(query_entities):
        eligible = 0
        for other_index, other in enumerate(reference_entities):
            if other == entity:
                continue
            if same_rows and other_index == index:
                continue
            eligible += 1
        counts.append(eligible)
    return MappingProxyType({
        "estimand": "other_entity_row_count",
        "counts": tuple(counts),
        "nearest_neighbor": False,
        "privacy_result": False,
    })


def null_cloud(rows: int, width: int, seed: int):
    if type(rows) is not int or type(width) is not int or rows < 2 or width < 1 or type(seed) is not int or seed < 0:
        raise ValueError("invalid null cloud")
    return np.random.default_rng(seed).normal(size=(rows, width))


def fit_minmax(rows) -> tuple:
    rows = np.asarray(rows, float)
    if rows.size == 0 or not np.isfinite(rows).all():
        raise ValueError("invalid scaler rows")
    return (float(rows.min()), float(rows.max()))


def shared_geometry_claim(scalers, *, frozen_id=None) -> dict:
    """Equal fit ranges are not a shared scaler. One supplied frozen identity is."""
    scalers = tuple(scalers)
    if not scalers:
        raise ValueError("missing scalers")
    if frozen_id is not None:
        if type(frozen_id) is not str or not frozen_id:
            raise ValueError("invalid frozen scaler")
        if all(scaler == frozen_id for scaler in scalers):
            return {"shared_geometry": True, "reason": "common_frozen_scaler"}
    if len(set(scalers)) == 1:
        return {"shared_geometry": False, "reason": "equal_ranges"}
    return {"shared_geometry": False, "reason": "per_table_refit"}


def real_real_calibration(*, orientation: str, scale: float, floor: float) -> dict:
    if orientation not in ("higher_closer", "higher_farther"):
        raise ValueError("missing calibration orientation")
    scale, floor = float(scale), float(floor)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(floor) or floor < 0:
        raise ValueError("invalid calibration scale")
    return {"orientation": orientation, "scale": scale, "floor": floor,
            "qualifies_transfer": False, "qualifies_privacy": False}


def closeness_qualifies(distance: float) -> dict:
    if not math.isfinite(float(distance)) or float(distance) < 0:
        raise ValueError("invalid distance")
    return {"transfer": False, "privacy": False, "formal_dp": False, "hipaa": False}


def held_out_status(learner_id: str, consumed) -> dict:
    if not learner_id:
        raise ValueError("missing learner")
    blocked = set(consumed) & {"embeddings", "labels", "losses"}
    if blocked:
        return {"learner_id": learner_id, "held_out": False, "reason": "selection_exposed",
                "consumed": tuple(sorted(blocked))}
    return {"learner_id": learner_id, "held_out": True, "reason": "no_recorded_use", "consumed": ()}
