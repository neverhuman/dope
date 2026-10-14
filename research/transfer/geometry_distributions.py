"""Euclidean cloud distributions. Pair counts are not independent units.

The geometry-state digest body is not built here. Closeness qualifies neither
transfer nor privacy.

Development ECDF band protocol `dope-ecdf-band-development` version 1, not
statistics-v1 and not N13b. The estimand is the equal-weight mean of
per-cluster empirical distribution functions on a caller-declared ascending
grid. Each declared fit is one group. Bootstrap draws resample whole clusters
with replacement and never resample pairwise distances. The band is the
pointwise 0.025 and 0.975 linear quantiles. It is not a simultaneous band.
The seed and draw count are part of the result. Fewer than two clusters, or
clusters with identical curves, stay unmeasured rather than a zero-width
interval. The draw count follows the existing bootstrap floor of 99.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .geometry import SPACES, _pairwise_norms


def _finite_cloud(rows):
    try:
        rows = np.asarray(rows, float)
    except (TypeError, ValueError):
        raise ValueError("invalid clouds") from None
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
    space and one encoder. Pair counts are not row counts. The full pair matrix
    is for fixture-scale clouds; a campaign panel needs a declared block before
    use. This function does not build that block.
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


ECDF_BAND_DEVELOPMENT = {
    "format": "dope-ecdf-band-development",
    "version": 1,
    "estimand": "equal_weight_mean_of_cluster_ecdfs",
    "group_weight": "one_per_declared_fit",
    "grid": "caller_declared_finite_ascending",
    "resample": "whole_clusters_with_replacement",
    "band": "pointwise_percentile",
    "quantile_method": "linear",
    "pointwise_tails": (0.025, 0.975),
    "simultaneous": False,
    "minimum_clusters": 2,
    "minimum_draws": 99,
    "qualifies": False,
    "n13b": False,
    "frozen_for_confirmation": False,
}


def dependence_aware_ecdf(distances, *, space: str, row_count: int, fit_count: int | None = None) -> dict:
    """Descriptive ECDF of one flat distance sample.

    A caller fit_count is not an independent-unit count. The band stays
    unmeasured. Passing the pair count as that integer is refused.
    """
    if space not in SPACES:
        raise ValueError("unknown coordinate space")
    if type(row_count) is not int or row_count < 2:
        raise ValueError("invalid ecdf")
    if fit_count is not None and (type(fit_count) is not int or fit_count < 1):
        raise ValueError("invalid ecdf")
    values = tuple(sorted(float(value) for value in distances))
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("invalid ecdf")
    if fit_count == len(values) and len(values) > row_count:
        raise ValueError("dependent_pairs_are_not_units")
    count = len(values)
    return {
        "estimand": "dependence_aware_ecdf",
        "space": space,
        "values": values,
        "probabilities": tuple((index + 1) / count for index in range(count)),
        "distance_count": count,
        "row_count": row_count,
        "caller_fit_count": fit_count,
        "fit_count_verified": False,
        "independent_units": None,
        "unit_status": "unverified",
        "pair_count_is_independent": False,
        "band": None,
        "band_status": "unmeasured",
        "qualifies_transfer": False,
        "privacy_result": False,
        "formal_dp": False,
        "hipaa": False,
        "n13b": False,
    }


def euclidean_cloud_report(real, synthetic, *, space: str, encoder_id: str, fit_count: int | None = None) -> dict:
    """One matched cloud pair and three descriptive ECDFs.

    The real-real curve is the reference. caller fit_count is not verified,
    and no role receives a measured band.
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
        "caller_fit_count": fit_count,
        "fit_count_verified": False,
        "independent_units": None,
        "unit_status": "unverified",
        "band_status": "unmeasured",
        "qualifies_transfer": False,
        "privacy_result": False,
        "formal_dp": False,
        "hipaa": False,
    }


def _hex_digest(value) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _declared_grid(grid):
    if isinstance(grid, (str, bytes, np.ndarray)) or not isinstance(grid, (tuple, list)) or not grid:
        raise ValueError("invalid ecdf")
    values = []
    for point in grid:
        if type(point) is not float or not math.isfinite(point):
            raise ValueError("invalid ecdf")
        values.append(point)
    values = tuple(values)
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("invalid ecdf")
    return values


def _curve_at(distances, grid):
    ordered = np.asarray(distances, float)
    points = np.asarray(grid, float)
    return np.searchsorted(ordered, points, side="right") / ordered.size


def _read_clusters(clusters, *, encoder_id: str):
    if isinstance(clusters, (str, bytes, np.ndarray)) or not isinstance(clusters, (tuple, list)) or not clusters:
        raise ValueError("query is not a fit cluster")
    records = []
    fit_ids = []
    entities = []
    digests = []
    shape = None
    for cluster in clusters:
        if isinstance(cluster, (str, bytes, np.ndarray)) or not isinstance(cluster, Mapping):
            raise ValueError("query is not a fit cluster")
        fit_id = cluster.get("fit_id")
        lineage_id = cluster.get("lineage_id")
        entity_id = cluster.get("entity_id")
        if (
            type(fit_id) is not str or not fit_id
            or type(lineage_id) is not str or not lineage_id
            or type(entity_id) is not str or not entity_id
        ):
            raise ValueError("invalid cluster")
        if cluster.get("encoder_id") not in (None, encoder_id):
            raise ValueError("inconsistent encoder")
        real = _finite_cloud(cluster.get("real"))
        synthetic = _finite_cloud(cluster.get("synthetic"))
        if real.shape != synthetic.shape:
            raise ValueError("unequal cloud cardinality")
        if shape is None:
            shape = real.shape
        elif real.shape != shape:
            raise ValueError("unequal cloud cardinality")
        digest = cluster.get("state_digest")
        records.append({
            "fit_id": fit_id,
            "lineage_id": lineage_id,
            "entity_id": entity_id,
            "real": real,
            "synthetic": synthetic,
        })
        fit_ids.append(fit_id)
        entities.append(entity_id)
        digests.append(digest)
    if len(set(fit_ids)) != len(fit_ids):
        raise ValueError("duplicate fit id")
    if len(set(entities)) != len(entities):
        raise ValueError("duplicate entity")
    if any(digest is not None for digest in digests):
        if any(not _hex_digest(digest) for digest in digests) or len(set(digests)) != 1:
            raise ValueError("inconsistent frozen state")
        frozen_state = "declared_unverified"
    else:
        frozen_state = "undeclared"
    records.sort(key=lambda record: record["fit_id"])
    return records, frozen_state


def _role_curve(record, role, grid):
    real, synthetic = record["real"], record["synthetic"]
    if role == "real_real":
        distances = _upper_euclidean(real)
    elif role == "synthetic_synthetic":
        distances = _upper_euclidean(synthetic)
    elif role == "real_synthetic":
        distances = tuple(sorted(float(value) for value in _pairwise_norms(real, synthetic).ravel()))
    else:
        raise ValueError("invalid ecdf")
    return _curve_at(distances, grid)


def grouped_euclidean_bands(clusters, *, space: str, encoder_id: str, grid, seed: int, draws: int) -> dict:
    """Pointwise ECDF bands from whole declared fit clusters.

    The result follows dope-ecdf-band-development version 1. It does not
    qualify transfer, privacy, formal DP, or HIPAA, and it is not N13b.
    """
    if space not in SPACES:
        raise ValueError("unknown coordinate space")
    if type(encoder_id) is not str or not encoder_id:
        raise ValueError("missing encoder")
    if type(seed) is not int or seed < 0 or type(draws) is not int or draws < ECDF_BAND_DEVELOPMENT["minimum_draws"]:
        raise ValueError("invalid bootstrap budget")
    grid = _declared_grid(grid)
    records, frozen_state = _read_clusters(clusters, encoder_id=encoder_id)
    roles = ("real_real", "real_synthetic", "synthetic_synthetic")
    curves = {
        role: np.vstack([_role_curve(record, role, grid) for record in records])
        for role in roles
    }
    roles_result = {}
    if len(records) < ECDF_BAND_DEVELOPMENT["minimum_clusters"]:
        for role in roles:
            roles_result[role] = {"status": "insufficient_clusters", "curve": None, "lower": None, "upper": None}
    else:
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, len(records), size=(draws, len(records)))
        for role, matrix in curves.items():
            point = matrix.mean(axis=0)
            if np.all(np.isclose(matrix, matrix[:1])):
                roles_result[role] = {"status": "degenerate_clusters", "curve": None, "lower": None, "upper": None}
                continue
            resampled = matrix[picks].mean(axis=1)
            lower, upper = np.quantile(
                resampled, ECDF_BAND_DEVELOPMENT["pointwise_tails"], axis=0, method="linear",
            )
            roles_result[role] = {
                "status": "pointwise_band",
                "curve": tuple(float(value) for value in point),
                "lower": tuple(float(value) for value in lower),
                "upper": tuple(float(value) for value in upper),
            }
    return {
        "estimand": ECDF_BAND_DEVELOPMENT["estimand"],
        "protocol": dict(ECDF_BAND_DEVELOPMENT),
        "space": space,
        "encoder_id": encoder_id,
        "roles": roles,
        "cluster_count": len(records),
        "independent_units": len(records),
        "row_count": int(records[0]["real"].shape[0]),
        "fit_ids": tuple(record["fit_id"] for record in records),
        "grid": grid,
        "seed": seed,
        "draws": draws,
        "resample": ECDF_BAND_DEVELOPMENT["resample"],
        "band_meaning": "pointwise",
        "simultaneous": False,
        "frozen_state": frozen_state,
        "reference_role": "real_real",
        "roles_result": roles_result,
        "qualifies_transfer": False,
        "privacy_result": False,
        "formal_dp": False,
        "hipaa": False,
        "n13b": False,
    }
