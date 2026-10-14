"""Euclidean cloud distributions. Pair counts are not independent units.

The geometry-state digest body is not built here. Closeness qualifies neither
transfer nor privacy.

Development ECDF band protocol `dope-ecdf-band-development` version 1, not
statistics-v1 and not N13b. The estimand is the equal-weight mean of
per-cluster empirical distribution functions on a caller-declared ascending
grid. Each declared fit is one group, and that fit must have its own lineage.
A repeated lineage is refused. A nested within-lineage resample is not declared.
Bootstrap draws resample whole fits with replacement and never resample
pairwise distances. The band is the
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


DEVELOPMENT_MAX_PAIRS = 1_000_000
PAIR_ALLOCATION_LIMIT = 10_000_000

STEP5_CONTROLS = (
    {"control": "duplicate", "state": "executable"},
    {"control": "reordered", "state": "executable"},
    {"control": "label_shuffled", "state": "executable"},
    {"control": "tail_deleted", "state": "executable"},
    {"control": "collapse", "state": "executable"},
    {"control": "row_count", "state": "executable"},
    {"control": "context", "state": "executable"},
    {"control": "coordinate_permutation", "state": "executable"},
    {"control": "pair_allocation", "state": "executable"},
    {"control": "held_out_relevance", "state": "unmeasured"},
    {"control": "official_kumo_dope_measurements", "state": "unmeasured"},
)

_PLOT_ESTIMANDS = frozenset({
    "euclidean_distance_distribution",
    "euclidean_cloud_report",
    "dependence_aware_ecdf",
    "equal_weight_mean_of_cluster_ecdfs",
    "label_shuffle_control",
    "tail_deletion_control",
    "context_query_control",
})


def _require_pair_budget(left_rows: int, right_rows: int, max_pairs: int) -> None:
    if type(max_pairs) is not int or max_pairs < 1 or max_pairs > PAIR_ALLOCATION_LIMIT:
        raise ValueError("invalid_distance_budget")
    if left_rows * right_rows > max_pairs:
        raise ValueError("distance_budget_exceeded")


def _bounded_pairwise(left, right, max_pairs):
    _require_pair_budget(int(left.shape[0]), int(right.shape[0]), max_pairs)
    return _pairwise_norms(left, right)


def _upper_euclidean(cloud, max_pairs):
    matrix = _bounded_pairwise(cloud, cloud, max_pairs)
    index = np.triu_indices(cloud.shape[0], k=1)
    return tuple(sorted(float(value) for value in matrix[index]))


def _qualification() -> dict:
    return {
        "qualifies_transfer": False,
        "privacy_result": False,
        "formal_dp": False,
        "hipaa": False,
        "n13b": False,
        "held_out": False,
        "held_out_relevance": "unmeasured",
    }


def euclidean_role_distributions(
    real, synthetic, *, space: str, encoder_id: str, max_pairs: int = DEVELOPMENT_MAX_PAIRS,
) -> dict:
    """Euclidean distances for real-real, real-synthetic, and synthetic-synthetic.

    The two clouds have the same row count and the same width. Each call is one
    space and one encoder. Pair counts are not row counts. The allocated matrix
    is rows by rows. A product above max_pairs is refused before that matrix
    is built. Pairs are not subsampled.
    """
    if space not in SPACES:
        raise ValueError("unknown coordinate space")
    if type(encoder_id) is not str or not encoder_id:
        raise ValueError("missing encoder")
    _require_pair_budget(1, 1, max_pairs)
    real = _finite_cloud(real)
    synthetic = _finite_cloud(synthetic)
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("invalid clouds")
    if real.shape[0] != synthetic.shape[0]:
        raise ValueError("unequal cloud cardinality")
    cross = _bounded_pairwise(real, synthetic, max_pairs)
    distances = {
        "real_real": _upper_euclidean(real, max_pairs),
        "real_synthetic": tuple(sorted(float(value) for value in cross.ravel())),
        "synthetic_synthetic": _upper_euclidean(synthetic, max_pairs),
    }
    row_count = int(real.shape[0])
    return {
        "estimand": "euclidean_distance_distribution",
        "space": space,
        "encoder_id": encoder_id,
        "roles": ("real_real", "real_synthetic", "synthetic_synthetic"),
        "row_count": row_count,
        "pair_counts": {role: len(values) for role, values in distances.items()},
        "allocated_pairs": row_count * row_count,
        "max_pairs": max_pairs,
        "distances": distances,
        "distance_definition": "euclidean",
        "independent_units": None,
        "band": None,
        "held_out_relevance": "unmeasured",
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
    "lineage_rule": "refuse_repeated_lineage",
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


def euclidean_cloud_report(
    real, synthetic, *, space: str, encoder_id: str, fit_count: int | None = None,
    max_pairs: int = DEVELOPMENT_MAX_PAIRS,
) -> dict:
    """One matched cloud pair and three descriptive ECDFs.

    The real-real curve is the reference. caller fit_count is not verified,
    and no role receives a measured band. Held-out relevance stays unmeasured.
    """
    distributions = euclidean_role_distributions(
        real, synthetic, space=space, encoder_id=encoder_id, max_pairs=max_pairs,
    )
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
        "allocated_pairs": distributions["allocated_pairs"],
        "max_pairs": max_pairs,
        "held_out_relevance": "unmeasured",
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
    lineages = []
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
        lineages.append(lineage_id)
        entities.append(entity_id)
        digests.append(digest)
    if len(set(fit_ids)) != len(fit_ids):
        raise ValueError("duplicate fit id")
    if len(set(lineages)) != len(lineages):
        raise ValueError("duplicate lineage")
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


def _role_curve(record, role, grid, max_pairs):
    real, synthetic = record["real"], record["synthetic"]
    if role == "real_real":
        distances = _upper_euclidean(real, max_pairs)
    elif role == "synthetic_synthetic":
        distances = _upper_euclidean(synthetic, max_pairs)
    elif role == "real_synthetic":
        distances = tuple(sorted(float(value) for value in _bounded_pairwise(real, synthetic, max_pairs).ravel()))
    else:
        raise ValueError("invalid ecdf")
    return _curve_at(distances, grid)


def grouped_euclidean_bands(
    clusters, *, space: str, encoder_id: str, grid, seed: int, draws: int,
    max_pairs: int = DEVELOPMENT_MAX_PAIRS,
) -> dict:
    """Pointwise ECDF bands from whole declared fit clusters.

    Each accepted fit has its own lineage. A repeated lineage is refused, so
    fits that share a lineage are not resampled as separate units. The result
    follows dope-ecdf-band-development version 1. It does not qualify transfer,
    privacy, formal DP, or HIPAA, and it is not N13b.
    """
    if space not in SPACES:
        raise ValueError("unknown coordinate space")
    if type(encoder_id) is not str or not encoder_id:
        raise ValueError("missing encoder")
    if type(seed) is not int or seed < 0 or type(draws) is not int or draws < ECDF_BAND_DEVELOPMENT["minimum_draws"]:
        raise ValueError("invalid bootstrap budget")
    _require_pair_budget(1, 1, max_pairs)
    grid = _declared_grid(grid)
    records, frozen_state = _read_clusters(clusters, encoder_id=encoder_id)
    roles = ("real_real", "real_synthetic", "synthetic_synthetic")
    curves = {
        role: np.vstack([_role_curve(record, role, grid, max_pairs) for record in records])
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
        "lineage_ids": tuple(record["lineage_id"] for record in records),
        "row_count": int(records[0]["real"].shape[0]),
        "allocated_pairs": int(records[0]["real"].shape[0]) * int(records[0]["real"].shape[0]),
        "max_pairs": max_pairs,
        "fit_ids": tuple(record["fit_id"] for record in records),
        "held_out_relevance": "unmeasured",
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


def _string_ids(values, message):
    if isinstance(values, (str, bytes, np.ndarray)) or not isinstance(values, (tuple, list)) or not values:
        raise ValueError(message)
    if any(type(value) is not str or not value for value in values):
        raise ValueError(message)
    return tuple(values)


def label_shuffle_control(
    real, synthetic, labels, *, space: str, encoder_id: str, seed: int,
    labels_consumed: bool = False, max_pairs: int = DEVELOPMENT_MAX_PAIRS,
) -> dict:
    """A side label vector is not a coordinate. Shuffling it leaves distances unchanged.

    Consumed labels make the geometry selection-exposed. Held-out relevance stays
    unmeasured either way.
    """
    if type(labels_consumed) is not bool or type(seed) is not int or seed < 0:
        raise ValueError("invalid labels")
    labels = _string_ids(labels, "invalid labels")
    real = _finite_cloud(real)
    if len(labels) != real.shape[0]:
        raise ValueError("invalid labels")
    report = euclidean_role_distributions(
        real, synthetic, space=space, encoder_id=encoder_id, max_pairs=max_pairs,
    )
    order = np.random.default_rng(seed).permutation(len(labels))
    shuffled = tuple(labels[int(index)] for index in order)
    if len(shuffled) != len(labels):
        raise ValueError("invalid labels")
    repeated = euclidean_role_distributions(
        real, synthetic, space=space, encoder_id=encoder_id, max_pairs=max_pairs,
    )
    return {
        "estimand": "label_shuffle_control",
        "space": space,
        "encoder_id": encoder_id,
        "seed": seed,
        "distances_unchanged": report["distances"] == repeated["distances"],
        "labels_are_coordinates": False,
        "labels_consumed": labels_consumed,
        "selection_exposed": labels_consumed,
        "distances": report["distances"],
        **_qualification(),
    }


def _farthest_index(cloud):
    center = np.median(cloud, axis=0)
    offset = cloud - center
    return int(np.argmax(np.sqrt(np.sum(offset * offset, axis=1))))


def tail_deletion_control(
    real, synthetic, *, space: str, encoder_id: str, max_pairs: int = DEVELOPMENT_MAX_PAIRS,
) -> dict:
    """Drop each cloud's farthest row and compare the matched distance roles.

    The comparison is a development fixture. It does not measure held-out relevance.
    """
    real = _finite_cloud(real)
    synthetic = _finite_cloud(synthetic)
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("invalid clouds")
    if real.shape[0] != synthetic.shape[0]:
        raise ValueError("unequal cloud cardinality")
    if real.shape[0] < 3:
        raise ValueError("invalid tail control")
    before = euclidean_role_distributions(
        real, synthetic, space=space, encoder_id=encoder_id, max_pairs=max_pairs,
    )
    reduced_real = np.delete(real, _farthest_index(real), axis=0)
    reduced_synthetic = np.delete(synthetic, _farthest_index(synthetic), axis=0)
    after = euclidean_role_distributions(
        reduced_real, reduced_synthetic, space=space, encoder_id=encoder_id, max_pairs=max_pairs,
    )
    before_values = {role: tuple(sorted(set(values))) for role, values in before["distances"].items()}
    after_values = {role: tuple(sorted(set(values))) for role, values in after["distances"].items()}
    values_changed = before_values != after_values
    return {
        "estimand": "tail_deletion_control",
        "space": space,
        "encoder_id": encoder_id,
        "row_count_before": before["row_count"],
        "row_count_after": after["row_count"],
        "pair_count_changed": before["pair_counts"] != after["pair_counts"],
        "distance_values_changed": values_changed,
        "tail_effect": "changed" if values_changed else "unobserved",
        "real_real_before": before["distances"]["real_real"],
        "real_real_after": after["distances"]["real_real"],
        **_qualification(),
    }


def context_query_control(
    context_ids, query_ids, real, synthetic, *, space: str, encoder_id: str,
    include_context: bool = False, max_pairs: int = DEVELOPMENT_MAX_PAIRS,
) -> dict:
    """Query distances exclude context rows. Overlap and pooling are refused.

    Held-out relevance stays unmeasured.
    """
    if type(include_context) is not bool:
        raise ValueError("invalid context")
    if include_context:
        raise ValueError("context is not a query")
    context_ids = _string_ids(context_ids, "invalid context")
    query_ids = _string_ids(query_ids, "invalid context")
    if set(context_ids).intersection(query_ids):
        raise ValueError("query_context_overlap")
    real = _finite_cloud(real)
    if len(query_ids) != real.shape[0]:
        raise ValueError("invalid context")
    report = euclidean_role_distributions(
        real, synthetic, space=space, encoder_id=encoder_id, max_pairs=max_pairs,
    )
    return {
        "estimand": "context_query_control",
        "space": space,
        "encoder_id": encoder_id,
        "context_row_count": len(context_ids),
        "query_row_count": report["row_count"],
        "context_included_in_distances": False,
        "distances": report["distances"],
        **_qualification(),
    }


def geometry_plot_hook(report) -> dict:
    """Status for a geometry figure. Real Kumo and DOPE measurements stay unmeasured."""
    if not isinstance(report, Mapping):
        raise ValueError("invalid geometry report")
    estimand = report.get("estimand")
    space = report.get("space")
    if estimand not in _PLOT_ESTIMANDS or space not in SPACES:
        raise ValueError("invalid geometry report")
    return {
        "format": "dope-geometry-plot-hook-development",
        "version": 1,
        "estimand": estimand,
        "space": space,
        "figure": "unmeasured",
        "kumo_measurements": "unmeasured",
        "dope_measurements": "unmeasured",
        "official_asset": False,
        **_qualification(),
    }
