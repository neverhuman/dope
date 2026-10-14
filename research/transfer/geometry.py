"""Separate geometry summaries. Closeness qualifies neither transfer nor privacy.

Kumo coordinates and DOPE coordinates are different spaces. A per-table refit
does not create a shared geometry claim. Unbiased MMD-squared is unclipped.
"""

from __future__ import annotations

from numbers import Real
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


def _pairwise_norms(left, right):
    diff = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def energy_distance_unbiased(left, right) -> float:
    """Unbiased energy distance. The diagonal is excluded and the value may be negative.

    The number is not a privacy result and does not qualify transfer.
    """
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or min(left.shape[0], right.shape[0]) < 2:
        raise ValueError("invalid clouds")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite cloud")
    cross = _pairwise_norms(left, right)
    left_d = _pairwise_norms(left, left)
    right_d = _pairwise_norms(right, right)
    np.fill_diagonal(left_d, 0.0)
    np.fill_diagonal(right_d, 0.0)
    m, n = left.shape[0], right.shape[0]
    return float(2 * cross.mean() - left_d.sum() / (m * (m - 1)) - right_d.sum() / (n * (n - 1)))


def sliced_wasserstein(left, right, *, projections: int, seed: int) -> dict:
    """Mean 1D Wasserstein-1 over one frozen set of directions.

    Equal sample counts are required. The result does not qualify transfer or privacy.
    """
    if type(projections) is not int or projections < 1 or type(seed) is not int or seed < 0:
        raise ValueError("invalid sliced wasserstein")
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or left.shape[1] < 1:
        raise ValueError("invalid clouds")
    if min(left.shape[0], right.shape[0]) < 1:
        raise ValueError("invalid clouds")
    if left.shape[0] != right.shape[0]:
        raise ValueError("unequal sliced cardinality")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite cloud")
    directions = np.random.default_rng(seed).normal(size=(projections, left.shape[1]))
    norms = np.linalg.norm(directions, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("invalid sliced wasserstein")
    directions = directions / norms[:, None]
    scores = []
    for direction in directions:
        left_projected = np.sort(left @ direction)
        right_projected = np.sort(right @ direction)
        scores.append(float(np.mean(np.abs(left_projected - right_projected))))
    return {"estimand": "sliced_wasserstein", "projections": projections, "seed": seed,
            "value": float(np.mean(scores)), "privacy_result": False,
            "qualifies_transfer": False, "formal_dp": False}


def ecdf(values, *, space: str) -> dict:
    if space not in SPACES:
        raise ValueError("unknown coordinate space")
    values = tuple(sorted(float(value) for value in values))
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("invalid ecdf")
    n = len(values)
    return {"space": space, "values": values, "probabilities": tuple((index + 1) / n for index in range(n))}


def _frozen_columns(frozen_scaler):
    if not isinstance(frozen_scaler, dict):
        raise ValueError("missing frozen scaler")
    frozen_id = frozen_scaler.get("frozen_id")
    columns = frozen_scaler.get("columns")
    if type(frozen_id) is not str or not frozen_id or not isinstance(columns, tuple) or not columns:
        raise ValueError("missing frozen scaler")
    if any(not isinstance(column, dict) for column in columns):
        raise ValueError("invalid frozen scaler")
    for column in columns:
        kind = column.get("kind")
        if kind == "continuous":
            minimum, maximum = column.get("minimum"), column.get("maximum")
            if type(minimum) is not float or type(maximum) is not float or not math.isfinite(minimum) or not math.isfinite(maximum):
                raise ValueError("invalid frozen scaler")
            if minimum == maximum:
                raise ValueError("constant_column")
            if minimum > maximum:
                raise ValueError("invalid frozen scaler")
        elif kind == "nominal":
            levels = column.get("levels")
            if not isinstance(levels, tuple) or not levels or any(type(level) is not str or not level for level in levels):
                raise ValueError("invalid frozen scaler")
            if len(set(levels)) < 2:
                raise ValueError("constant_column")
            if len(set(levels)) != len(levels):
                raise ValueError("invalid frozen scaler")
        else:
            raise ValueError("invalid frozen scaler")
    return frozen_id, columns


def _row_tuples(values):
    if isinstance(values, np.ndarray):
        if values.ndim != 2 or values.size == 0:
            raise ValueError("invalid geometry matrix")
        return tuple(tuple(row) for row in values.tolist())
    rows = tuple(tuple(row) for row in values)
    if not rows:
        raise ValueError("invalid geometry matrix")
    return rows


def _scale_rows(rows, columns):
    parsed = []
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("frozen scaler width mismatch")
        scaled = []
        for value, column in zip(row, columns):
            if column["kind"] == "continuous":
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError("invalid geometry matrix")
                number = float(value)
                if math.isnan(number):
                    raise ValueError("missing_value")
                if not math.isfinite(number):
                    raise ValueError("nonfinite_geometry")
                width = column["maximum"] - column["minimum"]
                scaled.append(("continuous", (number - column["minimum"]) / width))
            else:
                if value is None or value == "":
                    raise ValueError("missing_value")
                if type(value) is not str:
                    raise ValueError("invalid geometry matrix")
                if value not in column["levels"]:
                    raise ValueError("nominal_outside_frozen_levels")
                scaled.append(("nominal", value))
        parsed.append(tuple(scaled))
    return tuple(parsed)


def _mixed_distance(left, right):
    total = 0.0
    for (kind, left_value), (_, right_value) in zip(left, right):
        if kind == "continuous":
            gap = left_value - right_value
            total += gap * gap
        elif left_value != right_value:
            total += 1.0
    return total ** 0.5


def nearest_other_entity_distance(reference, query, *, reference_entities, query_entities, frozen_scaler):
    """Distance under one frozen scaler, excluding the same entity.

    Continuous columns use the frozen min/max. Nominal columns add a mismatch
    of one. The result is a neighbor distance, not an other-entity row count
    and not a privacy result. A per-table refit cannot supply the scaler.
    """
    frozen_id, columns = _frozen_columns(frozen_scaler)
    reference = _scale_rows(_row_tuples(reference), columns)
    query = _scale_rows(_row_tuples(query), columns)
    reference_entities = tuple(reference_entities)
    query_entities = tuple(query_entities)
    if len(reference) != len(reference_entities) or len(query) != len(query_entities):
        raise ValueError("entity cardinality mismatch")
    same_rows = reference == query
    distances, tie_counts, unique, statuses = [], [], [], []
    for index, entity in enumerate(query_entities):
        eligible = []
        for other_index, other in enumerate(reference_entities):
            if other == entity:
                continue
            if same_rows and other_index == index:
                continue
            eligible.append(_mixed_distance(query[index], reference[other_index]))
        if not eligible:
            distances.append(None)
            tie_counts.append(0)
            unique.append(False)
            statuses.append("missing_neighbor")
            continue
        nearest = min(eligible)
        ties = sum(distance == nearest for distance in eligible)
        distances.append(nearest)
        tie_counts.append(ties)
        unique.append(ties == 1)
        statuses.append("measured" if ties == 1 else "tie")
    return MappingProxyType({
        "estimand": "nearest_other_entity_distance",
        "frozen_id": frozen_id,
        "distances": tuple(distances),
        "tie_counts": tuple(tie_counts),
        "unique_neighbor": tuple(unique),
        "statuses": tuple(statuses),
        "nearest_neighbor": True,
        "privacy_result": False,
        "row_count": False,
    })


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


def shared_null_realizations(rows: int, width: int, seed: int) -> MappingProxyType:
    """One standard-normal cloud for real-real, real-synthetic, and synthetic-synthetic.

    The scale is the null scale. It is not a real-real calibration scale.
    The cloud is not a neighbor distance, a privacy result, or a transfer qualification.
    """
    cloud = null_cloud(rows, width, seed)
    return MappingProxyType({
        "estimand": "shared_null_realization",
        "seed": seed,
        "rows": rows,
        "width": width,
        "scale": 1.0,
        "roles": ("real_real", "real_synthetic", "synthetic_synthetic"),
        "cloud": cloud,
        "shared": True,
        "nearest_neighbor": False,
        "row_count": False,
        "privacy_result": False,
        "qualifies_transfer": False,
    })


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
