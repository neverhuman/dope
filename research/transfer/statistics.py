"""Family-balanced paired inference with crossed global seeds and query clusters.

Independent axis resampling follows the pigeonhole-bootstrap construction
(Owen, 2007, https://arxiv.org/abs/0712.1111). Finite-sample qualification still
requires a frozen design and separate Monte Carlo validation. No historical
scorer or receipt is rewritten. Missing/unbalanced schedules fail inference
closed; their complete-case point estimate is only a labeled sensitivity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np

from .losses import PairedLosses, TrainingScale, finite_number
from .metrics import Margin


def task_substream(master_seed: int, task_id: str, component_id: str) -> int:
    """Prospective RNG identity includes task and algorithm/component namespace."""
    if type(master_seed) is not int or master_seed < 0 or not task_id or not component_id:
        raise ValueError("invalid random stream identity")
    identity = json.dumps(["transfer-statistics-v1", master_seed, task_id, component_id], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")


@dataclass(frozen=True, repr=False)
class EvaluationRun:
    family_id: str
    task_id: str
    method_id: str
    fit_id: str
    sample_id: str
    learner_seed_id: str
    pairs: PairedLosses | None
    scale: TrainingScale
    margin: Margin
    status: str = "measured"
    global_seed_namespace: str | None = None
    global_seed_id: str | None = None

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in self.key):
            raise ValueError("incomplete run identity")
        if self.status not in ("measured", "technical_failure", "unsupported", "unresolved"):
            raise ValueError("unknown run disposition")
        if (self.status == "measured") != (self.pairs is not None):
            raise ValueError("run disposition disagrees with loss evidence")
        if (self.global_seed_id is None) != (self.global_seed_namespace is None):
            raise ValueError("global randomness requires an explicit coupling namespace")
        if self.global_seed_namespace is not None and (not self.global_seed_namespace or not self.global_seed_id):
            raise ValueError("empty global seed identity")

    @property
    def key(self):
        return (self.family_id, self.task_id, self.method_id, self.fit_id, self.sample_id, self.learner_seed_id)

    def values(self, endpoint: str) -> np.ndarray:
        if self.pairs is None:
            raise ValueError("missing loss evidence")
        if endpoint == "z":
            return np.asarray(self.margin.query_z(self.pairs, self.scale))
        values = np.asarray(self.pairs.synthetic) - np.asarray(self.pairs.real)
        if endpoint == "normalized_gap":
            if self.scale.value is None:
                raise ValueError("degenerate scale cannot acquire normalized precision")
            values = values / self.scale.value
        elif endpoint != "gap":
            raise ValueError("unknown endpoint")
        if not np.isfinite(values).all():
            raise ValueError("nonfinite paired endpoint")
        return values


@dataclass(frozen=True)
class CrossedDesign:
    runs: tuple[EvaluationRun, ...] = field(repr=False)
    required_methods: tuple[str, ...]
    scheduled_keys: tuple[tuple[str, ...], ...] = field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "runs", tuple(self.runs))
        object.__setattr__(self, "required_methods", tuple(self.required_methods))
        object.__setattr__(self, "scheduled_keys", tuple(tuple(k) for k in self.scheduled_keys))
        if not self.runs or not self.required_methods or any(not isinstance(m, str) or not m for m in self.required_methods) or len(set(self.required_methods)) != len(self.required_methods):
            raise ValueError("empty or duplicate method schedule")
        if len({r.key for r in self.runs}) != len(self.runs):
            raise ValueError("duplicate run or aliased fit identity")
        if len(set(self.scheduled_keys)) != len(self.scheduled_keys) or set(self.scheduled_keys) != {r.key for r in self.runs}:
            raise ValueError("run dispositions do not cover the declared schedule")
        tasks, query_sets, policies, fit_couplings = {}, {}, {}, {}
        for run in self.runs:
            if run.method_id not in self.required_methods:
                raise ValueError("unscheduled method")
            if tasks.setdefault(run.task_id, run.family_id) != run.family_id:
                raise ValueError("task has inconsistent family lineage")
            policy_key = (run.task_id, run.method_id)
            policy = (run.scale, run.margin, run.global_seed_namespace)
            if policies.setdefault(policy_key, policy) != policy:
                raise ValueError("task policy or seed coupling changed between runs")
            coupling = (run.global_seed_namespace, run.global_seed_id)
            fit_key = (run.task_id, run.method_id, run.fit_id)
            if fit_couplings.setdefault(fit_key, coupling) != coupling:
                raise ValueError("fit identity has inconsistent global seed")
            if run.pairs is not None:
                signature = (run.pairs.query_ids, run.pairs.cluster_ids, run.pairs.loss_space)
                if query_sets.setdefault(run.task_id, signature) != signature:
                    raise ValueError("paired queries or clusters are not exactly aligned")

    @property
    def counts(self):
        result = Counter(r.status for r in self.runs)
        return {"scheduled_runs": len(self.runs), "dispositions": dict(result),
                "families": len({r.family_id for r in self.runs}), "tasks": len({r.task_id for r in self.runs})}


def paired_family_effect(design: CrossedDesign, *, endpoint: str = "z") -> dict:
    """Equal family/task/fit/sample/learner weights, never query-count weights."""
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    for run in design.runs:
        if run.pairs is not None:
            tree[run.method_id][run.family_id][run.task_id][run.fit_id][run.sample_id].append(float(np.mean(run.values(endpoint))))
    family_effects, points = {}, {}
    for method in design.required_methods:
        family_effects[method] = {}
        for family, tasks in tree[method].items():
            task_values = []
            for fits in tasks.values():
                task_values.append(np.mean([np.mean([np.mean(learners) for learners in samples.values()]) for samples in fits.values()]))
            family_effects[method][family] = float(np.mean(task_values))
        values = list(family_effects[method].values())
        points[method] = float(np.mean(values)) if values else None
    return {**design.counts, "endpoint": endpoint, "estimates": points,
            "family_effects": family_effects, "estimand": "complete_case_sensitivity" if any(r.pairs is None for r in design.runs) else "scheduled_family_mean"}


def _tables(design: CrossedDesign, endpoint: str):
    """Rectangular fit/sample/learner cells per task, retaining named seed axes."""
    grouped = defaultdict(list)
    for run in design.runs:
        grouped[(run.task_id, run.method_id)].append(run)
    tables, namespace_seeds = {}, {}
    for task in sorted({r.task_id for r in design.runs}):
        for method in design.required_methods:
            runs = grouped[(task, method)]
            if not runs or any(r.pairs is None for r in runs):
                raise ValueError("incomplete scheduled evidence")
            lookup = {(r.fit_id, r.sample_id, r.learner_seed_id): r for r in runs}
            fits = sorted({r.fit_id for r in runs})
            samples = sorted({r.sample_id for r in runs})
            learners = sorted({r.learner_seed_id for r in runs})
            if len(lookup) != len(fits) * len(samples) * len(learners):
                raise ValueError("unbalanced scheduled evidence")
            namespace = runs[0].global_seed_namespace
            seeds = [next(r.global_seed_id for r in runs if r.fit_id == f) for f in fits]
            if namespace is not None:
                if len(set(seeds)) != len(seeds):
                    raise ValueError("global seed aliased by multiple fits")
                ordered = tuple(sorted(seeds))
                if namespace_seeds.setdefault(namespace, ordered) != ordered:
                    raise ValueError("incomplete crossed global seed axis")
            array = np.asarray([[[lookup[(f, s, l)].values(endpoint) for l in learners] for s in samples] for f in fits])
            tables[(task, method)] = (array, namespace, seeds, runs[0].pairs)
    return tables, namespace_seeds


def _bootstrap_refusal(point, estimand, draws, status, reason=None):
    """The requested estimand is refused. Complete-case numbers stay sensitivity."""
    payload = {
        "estimand": estimand,
        "status": status,
        "draws": draws,
        "endpoint": point["endpoint"],
        "scheduled_runs": point["scheduled_runs"],
        "dispositions": point["dispositions"],
        "families": point["families"],
        "tasks": point["tasks"],
        "estimates": None,
        "family_effects": None,
        "samples": None,
        "sensitivity": {
            "estimand": point["estimand"],
            "estimates": point["estimates"],
            "family_effects": point["family_effects"],
            "qualification": False,
        },
    }
    if reason is not None:
        payload["reason"] = reason
    return payload


def bootstrap_effects(design: CrossedDesign, *, endpoint: str = "z", estimand: str = "method_new_source",
                      draws: int = 1999, seed: int = 0) -> dict:
    """Cross sources/global fits; nest independent fits/sample/learner seeds.

    Query clusters are drawn once per task and shared across all its fits and
    methods. Fixed-artifact inference conditions on one fit/sample/learner per
    task and only resamples query clusters; it does not draw new sources.
    Failed fits and missing crossed cells refuse the requested estimand.
    """
    if type(draws) is not int or draws < 99 or type(seed) is not int or seed < 0:
        raise ValueError("invalid bootstrap budget or stream")
    if estimand not in ("method_new_source", "fixed_artifact"):
        raise ValueError("unknown estimand")
    point = paired_family_effect(design, endpoint=endpoint)

    def refuse(status, reason=None):
        return _bootstrap_refusal(point, estimand, draws, status, reason)

    try:
        tables, namespace_seeds = _tables(design, endpoint)
    except ValueError as error:
        reason = str(error)
        # A hole in the cross is not a scheduled family mean. Keep its average as sensitivity.
        if reason in {"incomplete scheduled evidence", "unbalanced scheduled evidence", "incomplete crossed global seed axis"}:
            point = {**point, "estimand": "complete_case_sensitivity"}
        return refuse("incomplete", reason)
    families = sorted({r.family_id for r in design.runs})
    if estimand == "method_new_source" and len(families) < 2:
        return refuse("insufficient_families")
    rng = np.random.default_rng(seed)
    global_draws = {}
    if estimand == "method_new_source":
        for namespace, seeds in sorted(namespace_seeds.items()):
            if len(seeds) < 2:
                return refuse("insufficient_global_seeds")
            global_draws[namespace] = rng.integers(len(seeds), size=(draws, len(seeds)))
    query_weights = {}
    for task in sorted({r.task_id for r in design.runs}):
        pairs = tables[(task, design.required_methods[0])][3]
        clusters = sorted(set(pairs.cluster_ids))
        if len(clusters) < 2:
            return refuse("insufficient_query_clusters")
        picks = rng.integers(len(clusters), size=(draws, len(clusters)))
        counts = np.sum(picks[:, :, None] == np.arange(len(clusters))[None, None, :], axis=1)
        index = np.asarray([clusters.index(c) for c in pairs.cluster_ids])
        weights = counts[:, index].astype(float)
        query_weights[task] = weights / weights.sum(axis=1, keepdims=True)
    task_samples = {}
    for (task, method), (array, namespace, seeds, _) in sorted(tables.items()):
        nfit, nsample, nlearner, _ = array.shape
        values = np.einsum("fslq,bq->bfsl", array, query_weights[task])
        if estimand == "fixed_artifact":
            if (nfit, nsample, nlearner) != (1, 1, 1):
                raise ValueError("fixed-artifact inference requires one declared workflow")
            task_samples[(task, method)] = values[:, 0, 0, 0]
            continue
        if nfit < 2:
            return refuse("insufficient_fits")
        if namespace is None:
            fits = rng.integers(nfit, size=(draws, nfit))
        else:
            mapping = np.asarray([seeds.index(s) for s in namespace_seeds[namespace]])
            fits = mapping[global_draws[namespace]]
        samples = rng.integers(nsample, size=(draws, nfit, nsample))
        learners = rng.integers(nlearner, size=(draws, nfit, nsample, nlearner))
        selected = values[np.arange(draws)[:, None, None, None], fits[:, :, None, None], samples[:, :, :, None], learners]
        task_samples[(task, method)] = selected.mean(axis=(1, 2, 3))
    family_tasks = defaultdict(set)
    for run in design.runs:
        family_tasks[run.family_id].add(run.task_id)
    family_picks = rng.integers(len(families), size=(draws, len(families))) if estimand == "method_new_source" else None
    output = {}
    for method in design.required_methods:
        family_samples = np.column_stack([np.mean([task_samples[(task, method)] for task in sorted(family_tasks[f])], axis=0) for f in families])
        output[method] = family_samples[np.arange(draws)[:, None], family_picks].mean(axis=1) if family_picks is not None else family_samples.mean(axis=1)
    return {**point, "estimand": estimand, "draws": draws, "status": "complete", "samples": output}


def simultaneous_ucb(bootstrap: dict, *, alpha: float = 0.05) -> dict:
    """Bonferroni basic-bootstrap bounds for the frozen contrast family.

    Bias centering uses point minus bootstrap deviation. These are approximate
    bounds, not distribution-free guarantees or verified qualification receipts.
    """
    alpha = finite_number(alpha)
    if not 0 < alpha < 0.5:
        raise ValueError("invalid inference alpha")
    if bootstrap["status"] != "complete":
        return {"status": bootstrap["status"], "bounds": None}
    samples = bootstrap["samples"]
    per_tail = alpha / len(samples)
    if per_tail * bootstrap["draws"] < 5:
        return {"status": "insufficient_tail_draws", "bounds": None}
    bounds = {}
    for method, draws in samples.items():
        point = bootstrap["estimates"][method]
        lo, hi = np.quantile(draws - point, [per_tail, 1 - per_tail])
        bounds[method] = {"estimate": point, "lower": float(point - hi), "upper": float(point - lo)}
    return {"status": "complete", "alpha": alpha, "multiplicity": "bonferroni_one_sided",
            "simultaneous_one_sided_level": 1 - alpha, "two_sided_level_at_least": 1 - 2 * alpha,
            "bounds": bounds}


def interval_decision(lower: float, upper: float, *, endpoint: str = "gap",
                      equivalence_margin: float | None = None) -> dict:
    """Raw-gap superiority is separate from tolerance-adjusted noninferiority.

    Equivalence requires both bounds strictly inside the frozen margin. A
    zero-crossing interval is equivalence only when it is inside that margin.
    A negative tolerance-adjusted Z is within-tolerance noninferiority, never
    raw-loss superiority, including when synthetic risk is worse inside the margin.
    """
    lower, upper = finite_number(lower), finite_number(upper)
    if lower > upper:
        raise ValueError("reversed interval")
    if endpoint not in ("gap", "normalized_gap", "z"):
        raise ValueError("unknown endpoint")
    equivalent = False
    if equivalence_margin is not None:
        margin = finite_number(equivalence_margin)
        if margin <= 0:
            raise ValueError("equivalence requires a positive frozen margin")
        equivalent = -margin < lower and upper < margin
    superior = endpoint in ("gap", "normalized_gap") and upper < 0
    noninferior = upper <= 0
    harmful = lower > 0
    return {"endpoint": endpoint, "superior": superior, "noninferior": noninferior,
            "harmful": harmful, "equivalent": equivalent,
            "inconclusive": not (superior or noninferior or harmful or equivalent)}


def require_separate_cell_bounds(bounds: dict) -> dict:
    """One method-level population bound cannot be reused for every cell."""
    if not isinstance(bounds, dict) or not bounds:
        raise ValueError("missing cell bounds")
    values = list(bounds.values())
    if len(values) >= 2 and all(item is values[0] for item in values):
        raise ValueError("one population bound cannot cover every cell")
    return bounds


def leave_one_global_seed_out(design: CrossedDesign, *, endpoint="z") -> dict:
    output = {}
    namespaces = sorted({r.global_seed_namespace for r in design.runs if r.global_seed_namespace is not None})
    for namespace in namespaces:
        seeds = sorted({r.global_seed_id for r in design.runs if r.global_seed_namespace == namespace})
        if len(seeds) < 2:
            continue
        output[namespace] = {}
        for seed in seeds:
            retained = tuple(r for r in design.runs if not (r.global_seed_namespace == namespace and r.global_seed_id == seed))
            output[namespace][seed] = paired_family_effect(
                CrossedDesign(retained, design.required_methods, tuple(r.key for r in retained)), endpoint=endpoint,
            )["estimates"]
    return output
