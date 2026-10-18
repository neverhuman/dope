"""Bounded numerical whitebox probes; no protected sources or bundle files opened.

Draft B01 interface: owner supplies the complete, already decoded R(A,H) numeric
parameters below, not an opaque checkpoint or a claim that its payload is safe.
Gaussian copula and pair-correlation factors are reference probes, not a native
vine decoder or LiRA parity. D02 owns admission/custody; supplied views must be
authorized development data. S00 alone verifies release-axis evidence. These
fixtures and local empirical measurements cannot advance its trusted profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math

import numpy as np
from scipy.special import logsumexp, ndtri

from .attacks import PrivacyError, number, rate_bounds, require
from .controls import DistanceRecipe, RowView, _disjoint, _labels
from .release_profile import local_empirical_profile


PROBES = ("singling_out", "linkability", "attribute_inference", "membership_likelihood_ratio",
          "membership_distance", "membership_parameter_inspection")
CONTROLS = ("copied_rows", "memorized_outlier_endpoint", "exact_lookup_cell")
MAX_CANDIDATES = 4096


@dataclass(frozen=True, slots=True, repr=False)
class WhiteboxBundle:
    """Private positional adapter, with every channel explicitly supplied.

    marginals[j] = ((support, probability), ...); correlations = Gaussian matrix;
    vine_correlations = ((left_index, right_index, rho), ...); interaction_tables
    = ((column_indices, ((cell_values, probability), ...)), ...). Quantizer
    endpoints are one (lower, upper) pair per column. Empty channels are explicit.
    Raw data and parameter values never appear in returned reports or reprs.
    """
    rows: tuple
    marginals: tuple
    correlations: tuple
    vine_correlations: tuple
    interaction_tables: tuple
    quantizer_endpoints: tuple

    def __post_init__(self):
        d = len(self.marginals) if type(self.marginals) is tuple else 0
        require(3 <= d <= 8, "whitebox_width_invalid")
        require(type(self.rows) is tuple and len(self.rows) <= 1024, "whitebox_rows_invalid")
        require(all(type(row) is tuple and len(row) == d and all(_scalar(x) for x in row)
                    for row in self.rows), "whitebox_rows_invalid")
        for marginal in self.marginals:
            require(type(marginal) is tuple and 0 < len(marginal) <= 1024, "whitebox_marginal_invalid")
            require(all(type(p) is tuple and len(p) == 2 and _scalar(p[0]) and number(p[1])
                        and 0 < p[1] <= 1 for p in marginal), "whitebox_marginal_invalid")
            require(len({p[0] for p in marginal}) == len(marginal)
                    and math.isclose(sum(p[1] for p in marginal), 1., abs_tol=1e-9), "whitebox_marginal_invalid")
        require(type(self.correlations) is tuple, "whitebox_correlation_invalid")
        if self.correlations:
            require(len(self.correlations) == d and all(type(r) is tuple and len(r) == d
                    and all(number(x) and abs(x) <= 1 for x in r) for r in self.correlations), "whitebox_correlation_invalid")
            matrix = np.asarray(self.correlations)
            require(np.allclose(matrix, matrix.T, atol=1e-12) and np.allclose(np.diag(matrix), 1.)
                    and np.linalg.eigvalsh(matrix)[0] > 1e-8, "whitebox_correlation_invalid")
        require(type(self.vine_correlations) is tuple and len(self.vine_correlations) <= d * d, "whitebox_vine_invalid")
        for edge in self.vine_correlations:
            require(type(edge) is tuple and len(edge) == 3 and all(type(i) is int and 0 <= i < d for i in edge[:2])
                    and edge[0] != edge[1] and number(edge[2]) and abs(edge[2]) < .999, "whitebox_vine_invalid")
        require(type(self.interaction_tables) is tuple and len(self.interaction_tables) <= 64, "whitebox_table_invalid")
        for table in self.interaction_tables:
            require(type(table) is tuple and len(table) == 2, "whitebox_table_invalid")
            columns, cells = table
            require(type(columns) is tuple and bool(columns) and len(set(columns)) == len(columns)
                    and all(type(i) is int and 0 <= i < d for i in columns), "whitebox_table_invalid")
            require(type(cells) is tuple and 0 < len(cells) <= 1024, "whitebox_table_invalid")
            require(all(type(c) is tuple and len(c) == 2 and type(c[0]) is tuple and len(c[0]) == len(columns)
                        and all(_scalar(x) for x in c[0]) and number(c[1]) and 0 < c[1] <= 1 for c in cells), "whitebox_table_invalid")
            require(len({c[0] for c in cells}) == len(cells)
                    and math.isclose(sum(c[1] for c in cells), 1., abs_tol=1e-9), "whitebox_table_invalid")
        require(type(self.quantizer_endpoints) is tuple and len(self.quantizer_endpoints) in (0, d), "whitebox_endpoint_invalid")
        require(all(type(p) is tuple and len(p) == 2 and all(_scalar(x) for x in p) and p[0] <= p[1]
                    for p in self.quantizer_endpoints), "whitebox_endpoint_invalid")
        require(len(_candidates(self)) <= MAX_CANDIDATES, "whitebox_candidate_budget")


def _scalar(x):
    return number(x) and abs(x) <= 1e12


def _candidates(bundle):
    # Bounded reconstruction recipe: observed rows, marginal modes with one
    # support changed, each lookup cell completed at other marginal modes, and
    # aligned endpoint tuples. This is not exhaustive reconstruction certification.
    mode = tuple(max(m, key=lambda p: p[1])[0] for m in bundle.marginals)
    result = set(bundle.rows) | {mode}
    for j, marginal in enumerate(bundle.marginals):
        for value, _ in marginal:
            result.add(mode[:j] + (value,) + mode[j + 1:])
    for columns, cells in bundle.interaction_tables:
        for values, _ in cells:
            row = list(mode)
            for j, value in zip(columns, values):
                row[j] = value
            result.add(tuple(row))
    if bundle.quantizer_endpoints:
        result.update(tuple(pair[side] for pair in bundle.quantizer_endpoints) for side in (0, 1))
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True, repr=False)
class HarnessPolicy:
    target_fpr: float = .01
    alpha: float = .05
    detection_floor: float = .8
    maximum_tpr: float | None = None
    control_trials: int = 1000
    match_radius: float = .01

    def __post_init__(self):
        require(number(self.target_fpr) and 0 < self.target_fpr <= .1, "harness_fpr_invalid")
        require(number(self.alpha) and 1e-9 <= self.alpha <= .1, "harness_alpha_invalid")
        require(number(self.detection_floor) and 0 < self.detection_floor < 1, "harness_floor_invalid")
        require(self.maximum_tpr is None or (number(self.maximum_tpr) and 0 < self.maximum_tpr < self.detection_floor),
                "harness_margin_invalid")
        require(type(self.control_trials) is int and 1 <= self.control_trials <= 5000, "harness_control_budget")
        require(number(self.match_radius) and 1e-8 <= self.match_radius <= .1, "harness_radius_invalid")


def _distances(row, candidates, columns, recipe):
    return np.sqrt(sum(((candidates[:, j] - row[j]) / recipe.scales[j]) ** 2
                       if recipe.kinds[j] == "numeric" else (candidates[:, j] != row[j]).astype(float)
                       for j in columns) / len(columns))


def _log_density(bundle, row, recipe, radius):
    logs, latent = [], []
    for j, marginal in enumerate(bundle.marginals):
        scale = recipe.scales[j] * radius
        kernels = [math.log(p) - (min(1500., ((row[j] - x) / scale) ** 2) / 2
                   if recipe.kinds[j] == "numeric" else 0. if row[j] == x else 750.) for x, p in marginal]
        logs.append(float(logsumexp(kernels)))
        cdf = sum(p * (1 if x < row[j] else .5 if x == row[j] else 0) for x, p in marginal)
        latent.append(float(ndtri(min(1 - 1e-6, max(1e-6, cdf)))))
    result = sum(logs)
    z = np.asarray(latent)
    if bundle.correlations:
        matrix = np.asarray(bundle.correlations)
        result += -.5 * (np.linalg.slogdet(matrix)[1] + z @ (np.linalg.solve(matrix, z) - z))
    for left, right, rho in bundle.vine_correlations:
        a, b = z[left], z[right]
        result += -.5 * math.log1p(-rho * rho) - (rho * rho * (a*a + b*b) - 2*rho*a*b) / (2*(1-rho*rho))
    for columns, cells in bundle.interaction_tables:
        kernels = [math.log(p) - .5 * sum(min(1500., ((row[j] - value) / (recipe.scales[j] * radius)) ** 2)
                    if recipe.kinds[j] == "numeric" else 0. if row[j] == value else 1500.
                    for j, value in zip(columns, values)) for values, p in cells]
        result += float(logsumexp(kernels)) - sum(logs[j] for j in columns)
    # Recipient-visible copied rows and paired extrema are explicit additional
    # mixture channels, including when no synthetic rows were sampled.
    points = list(bundle.rows)
    if bundle.quantizer_endpoints:
        points.extend(tuple(pair[side] for pair in bundle.quantizer_endpoints) for side in (0, 1))
    if points:
        kernels = [-.5 * sum(min(1500., ((row[j] - point[j]) / (recipe.scales[j] * radius)) ** 2)
                              if recipe.kinds[j] == "numeric" else 0. if row[j] == point[j] else 1500.
                              for j in range(len(row))) for point in points]
        result = float(np.logaddexp(result, logsumexp(kernels) - math.log(len(points))) - math.log(2))
    return float(result)


def _predict(bundle, row, qids, hidden, recipe, radius):
    # Fit/score hypotheses using only QIs; the true sensitive value is never
    # included in the prediction input or in tie breaking.
    support = {x for x, _ in bundle.marginals[hidden]}
    support.update(r[hidden] for r in bundle.rows)
    if bundle.quantizer_endpoints:
        support.update(bundle.quantizer_endpoints[hidden])
    for columns, cells in bundle.interaction_tables:
        if hidden in columns:
            support.update(values[columns.index(hidden)] for values, _ in cells)
    base = [max(m, key=lambda p: p[1])[0] for m in bundle.marginals]
    for j in qids:
        base[j] = row[j]
    def likelihood(value):
        candidate = list(base)
        candidate[hidden] = value
        # Endpoint inspection explicitly reconstructs the paired endpoint tuple.
        endpoints = bundle.quantizer_endpoints
        endpoint_match = endpoints and any(all(row[j] == endpoints[j][side] for j in qids)
                                           and value == endpoints[hidden][side] for side in (0, 1))
        return (bool(endpoint_match), _log_density(bundle, candidate, recipe, radius), -value)
    return max(sorted(support), key=likelihood)


def _reference(rows):
    require(type(rows) is tuple and 0 < len(rows) <= 1024, "harness_reference_budget")
    marginals = []
    for column in zip(*rows):
        counts = Counter(column)
        marginals.append(tuple((x, counts[x] / len(column)) for x in sorted(counts)))
    return WhiteboxBundle(rows, tuple(marginals), (), (), (), ())


def _scores(bundle, reference, rows, recipe, splits, hidden, radius):
    candidates = np.asarray(_candidates(bundle), dtype=float)
    refs = np.asarray(reference.rows, dtype=float)
    qids = splits[0] + splits[1]
    work = len(candidates) + len(refs)
    for model in (bundle, reference):
        cells = sum(len(c) for _, c in model.interaction_tables)
        support = len(model.marginals[hidden]) + len(model.rows) + cells + 2
        work += support * (sum(map(len, model.marginals)) + len(model.rows) + cells + 1)
    require(len(rows) * work <= recipe.max_pairs, "harness_pair_budget")
    occurrences = Counter(rows)
    result = {probe: [] for probe in PROBES}
    for row in rows:
        all_distance = _distances(row, candidates, tuple(range(len(row))), recipe)
        ref_distance = _distances(row, refs, tuple(range(len(row))), recipe)
        left = np.flatnonzero(_distances(row, candidates, splits[0], recipe) <= radius)
        right = np.flatnonzero(_distances(row, candidates, splits[1], recipe) <= radius)
        unique = np.flatnonzero(all_distance <= radius)
        ref_known = ref_distance.min() <= radius
        ref_link_known = _distances(row, refs, qids, recipe).min() <= radius
        guess = _predict(bundle, row, qids, hidden, recipe, radius)
        baseline = _predict(reference, row, qids, hidden, recipe, radius)
        correct = lambda value: (value == row[hidden] if recipe.kinds[hidden] == "nominal"
                                 else abs(value - row[hidden]) <= recipe.scales[hidden] * radius)
        parameter = any(all(abs(row[j] - value) <= recipe.scales[j] * radius for j, value in zip(columns, values))
                        for columns, cells in bundle.interaction_tables for values, _ in cells)
        if bundle.quantizer_endpoints:
            parameter |= any(all(row[j] == bundle.quantizer_endpoints[j][side] for j in range(len(row))) for side in (0, 1))
        density = _log_density(bundle, row, recipe, radius) - _log_density(reference, row, recipe, radius)
        values = (float(len(unique) == 1 and occurrences[tuple(row)] == 1 and not ref_known),
                  float(len(left) == len(right) == 1 and left[0] == right[0] and not ref_link_known),
                  float(correct(guess)) - float(correct(baseline)), density / (1 + abs(density)),
                  float((ref_distance.min() - all_distance.min()) / (ref_distance.min() + all_distance.min() + 1e-12)),
                  float((parameter or all_distance.min() <= radius) and not ref_known))
        require(all(number(x) for x in values), "harness_scores_invalid")
        for probe, value in zip(PROBES, values):
            result[probe].append(value)
    return result


def _controls(recipe, splits, hidden, policy):
    # Each trial is its own planted release, with one target and a distinct
    # nonmember. This avoids counting repeats of one leaked endpoint as power.
    random = np.random.default_rng(71291)
    result = {control: {probe: [[], []] for probe in PROBES} for control in CONTROLS}
    for control in CONTROLS:
        for _ in range(policy.control_trials):
            positive = tuple(float(x) for x in random.uniform(10, 11, len(recipe.kinds)))
            negative = tuple(x + min(4 * scale, 1e11) for x, scale in zip(positive, recipe.scales))
            # Benign marginals reveal only the auxiliary record; each control
            # plants its target in precisely its named release channel.
            marginals = tuple(((x, 1.),) for x in negative)
            bundle = WhiteboxBundle((positive,) if control == "copied_rows" else (), marginals, (), (),
                ((tuple(range(len(positive))), ((positive, 1.),)),) if control == "exact_lookup_cell" else (),
                tuple((lo, hi) for lo, hi in zip(positive, negative)) if control == "memorized_outlier_endpoint" else ())
            scores = _scores(bundle, _reference((negative,)), (positive, negative), recipe, splits, hidden, policy.match_radius)
            for probe in PROBES:
                result[control][probe][0].append(scores[probe][0])
                result[control][probe][1].append(scores[probe][1])
    return result


def _rates(positive, negative, threshold, event_alpha):
    tp, fp = sum(x >= threshold for x in positive), sum(x >= threshold for x in negative)
    lo, hi = rate_bounds(tp, len(positive), event_alpha)
    _, fpr_hi = rate_bounds(fp, len(negative), event_alpha)
    return {"positives": len(positive), "negatives": len(negative), "tpr": tp / len(positive),
            "fpr": fp / len(negative), "tpr_lower": lo, "tpr_upper": hi, "fpr_upper": fpr_hi}


def run_harness(bundle, reference, development, development_membership, evaluation, evaluation_membership,
                recipe, *, qid_splits, sensitive, policy=None):
    """Freeze thresholds on development nonmembers, then controls, then target.

    Independent unit IDs are required across reference/development/evaluation;
    labels establish challenge truth, not verified real-world membership. Counts
    and simultaneous exact bounds refer only to these supplied independent views.
    Defaults withhold screening until an explicit application risk ceiling exists.
    """
    policy = HarnessPolicy() if policy is None else policy
    require(type(policy) is HarnessPolicy and type(bundle) is WhiteboxBundle and type(recipe) is DistanceRecipe, "harness_input_invalid")
    require(type(reference) is RowView and reference.role == "auxiliary_train", "harness_reference_invalid")
    require(type(development) is RowView and development.role == "attack_development", "development_required")
    require(type(evaluation) is RowView and evaluation.role == "attack_evaluation", "evaluation_required")
    d = len(bundle.marginals)
    require(len(recipe.kinds) == d and all(len(v.rows[0]) == d for v in (reference, development, evaluation)), "harness_width_invalid")
    require(type(sensitive) is int and 0 <= sensitive < d, "harness_sensitive_invalid")
    require(type(qid_splits) is tuple and len(qid_splits) == 2 and all(type(s) is tuple and bool(s) for s in qid_splits), "harness_qids_invalid")
    qids = qid_splits[0] + qid_splits[1]
    require(all(type(j) is int and 0 <= j < d and j != sensitive for j in qids)
            and len(set(qids)) == len(qids), "harness_qids_invalid")
    for view, labels in ((development, development_membership), (evaluation, evaluation_membership)):
        _labels(labels, len(view.rows))
        require(len(set(view.units)) == len(view.units), "harness_cluster_inference_required")
    for left, right in ((reference, development), (reference, evaluation), (development, evaluation)):
        _disjoint(left, right)
    ref = _reference(reference.rows)
    dev = _scores(bundle, ref, development.rows, recipe, qid_splits, sensitive, policy.match_radius)
    thresholds = {}
    for probe in PROBES:
        negatives = sorted((x for x, member in zip(dev[probe], development_membership) if not member), reverse=True)
        thresholds[probe] = math.nextafter(negatives[math.floor(policy.target_fpr * len(negatives))], math.inf)
    # Two TPR bounds and one FPR bound for each probe/control/target comparison.
    event_alpha = policy.alpha / (3 * len(PROBES) * (len(CONTROLS) + 1))
    controls = _controls(recipe, qid_splits, sensitive, policy)
    measurements, failed, unresolved = {}, False, False
    for control in CONTROLS:
        measurements[control] = {}
        for probe in PROBES:
            rates = _rates(*controls[control][probe], thresholds[probe], event_alpha)
            rates["detected"] = rates["tpr_lower"] > policy.detection_floor and rates["fpr_upper"] <= policy.target_fpr
            failed |= not rates["detected"]
            unresolved |= rates["fpr_upper"] > policy.target_fpr
            measurements[control][probe] = rates
    report = {"format": "dope-whitebox-disclosure-v1", "version": 1, "controls": measurements, "attacks": {},
              "status": "calibration_failed", "formal_dp_status": "not_dp", "legal_release_status": "unauthorized",
              "release_authorized": False, "custody_hold": "D18_HOLD", "scope": "supplied_numeric_views_and_declared_parameters",
              "policy": {"alpha": policy.alpha, "target_fpr": policy.target_fpr,
                         "detection_floor": policy.detection_floor, "maximum_tpr": policy.maximum_tpr,
                         "control_trials": policy.control_trials},
              "release_profile": local_empirical_profile()}
    if failed:
        report["status"] = "inconclusive_underpowered" if unresolved else "calibration_failed"
        return report
    final = _scores(bundle, ref, evaluation.rows, recipe, qid_splits, sensitive, policy.match_radius)
    for probe in PROBES:
        positives = [x for x, member in zip(final[probe], evaluation_membership) if member]
        negatives = [x for x, member in zip(final[probe], evaluation_membership) if not member]
        report["attacks"][probe] = _rates(positives, negatives, thresholds[probe], event_alpha)
    rates = list(report["attacks"].values())
    if policy.maximum_tpr is None:
        report["status"] = "inconclusive_unqualified"
    elif any(r["fpr_upper"] > policy.target_fpr for r in rates):
        report["status"] = "inconclusive_underpowered"
    elif any(r["tpr_lower"] > policy.maximum_tpr for r in rates):
        report["status"] = "disclosure_detected"
    elif all(r["tpr_upper"] <= policy.maximum_tpr for r in rates):
        report["status"] = "empirical_screened_only"
    else:
        report["status"] = "inconclusive_underpowered"
    return report
