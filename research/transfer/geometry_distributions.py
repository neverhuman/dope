"""Euclidean cloud distributions. Pair counts are not independent units.

The geometry-state digest body is not built here. Closeness qualifies neither
transfer nor privacy.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import SPACES, _pairwise_norms


def _finite_cloud(rows):
    rows = np.asarray(rows, float)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < 1 or not np.isfinite(rows).all():
        raise ValueError("invalid clouds")
    return rows


def _upper_euclidean(cloud):
    matrix = _pairwise_norms(cloud, cloud)
    index = np.triu_indices(cloud.shape[0], k=1)
    return tuple(sorted(float(value) for value in matrix[index]))


def euclidean_role_distributions(real, synthetic, *, space: str, encoder_id: str) -> dict:
    """Euclidean distances for real-real, real-synthetic, and synthetic-synthetic.

    The two clouds have the same row count and the same width. Each call is one
    space and one encoder. Pair counts are not row counts.
    """
    if space not in SPACES:
        raise ValueError("unknown coordinate space")
    if type(encoder_id) is not str or not encoder_id:
        raise ValueError("missing encoder")
    real = _finite_cloud(real)
    synthetic = _finite_cloud(synthetic)
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("invalid clouds")
    if real.shape[0] != synthetic.shape[0]:
        raise ValueError("unequal cloud cardinality")
    cross = _pairwise_norms(real, synthetic)
    distances = {
        "real_real": _upper_euclidean(real),
        "real_synthetic": tuple(sorted(float(value) for value in cross.ravel())),
        "synthetic_synthetic": _upper_euclidean(synthetic),
    }
    row_count = int(real.shape[0])
    return {
        "estimand": "euclidean_distance_distribution",
        "space": space,
        "encoder_id": encoder_id,
        "roles": ("real_real", "real_synthetic", "synthetic_synthetic"),
        "row_count": row_count,
        "pair_counts": {role: len(values) for role, values in distances.items()},
        "distances": distances,
        "distance_definition": "euclidean",
        "independent_units": None,
        "band": None,
        "qualifies_transfer": False,
        "privacy_result": False,
        "formal_dp": False,
        "hipaa": False,
    }


def dependence_aware_ecdf(distances, *, space: str, row_count: int, fit_count: int) -> dict:
    """ECDF of one distance sample. A confidence band stays unmeasured.

    fit_count is the independent-unit count. A pair count larger than the row
    count is not that unit.
    """
    if space not in SPACES:
        raise ValueError("unknown coordinate space")
    if type(row_count) is not int or row_count < 2:
        raise ValueError("invalid ecdf")
    if type(fit_count) is not int or fit_count < 1:
        raise ValueError("invalid ecdf")
    values = tuple(sorted(float(value) for value in distances))
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("invalid ecdf")
    if fit_count == len(values) and len(values) > row_count:
        raise ValueError("dependent_pairs_are_not_units")
    if fit_count > row_count:
        raise ValueError("invalid ecdf")
    count = len(values)
    return {
        "estimand": "dependence_aware_ecdf",
        "space": space,
        "values": values,
        "probabilities": tuple((index + 1) / count for index in range(count)),
        "distance_count": count,
        "row_count": row_count,
        "fit_count": fit_count,
        "independent_units": fit_count,
        "pair_count_is_independent": False,
        "band": None,
        "band_status": "unmeasured",
        "qualifies_transfer": False,
        "privacy_result": False,
        "formal_dp": False,
    }


def euclidean_cloud_report(real, synthetic, *, space: str, encoder_id: str, fit_count: int) -> dict:
    """One matched cloud pair and its three dependence-aware ECDFs.

    The real-real curve is the reference. No role receives a measured band.
    """
    distributions = euclidean_role_distributions(real, synthetic, space=space, encoder_id=encoder_id)
    row_count = distributions["row_count"]
    ecdfs = {
        role: dependence_aware_ecdf(values, space=space, row_count=row_count, fit_count=fit_count)
        for role, values in distributions["distances"].items()
    }
    return {
        "estimand": "euclidean_cloud_report",
        "space": space,
        "encoder_id": encoder_id,
        "roles": distributions["roles"],
        "row_count": row_count,
        "pair_counts": distributions["pair_counts"],
        "distances": distributions["distances"],
        "ecdfs": ecdfs,
        "reference_role": "real_real",
        "reference_probabilities": ecdfs["real_real"]["probabilities"],
        "band_status": "unmeasured",
        "qualifies_transfer": False,
        "privacy_result": False,
        "formal_dp": False,
        "hipaa": False,
    }
