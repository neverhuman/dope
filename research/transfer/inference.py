"""Portable crossed-axis fixture engine and paired max-T inference.

Source and task draws are independent (pigeonhole bootstrap); their product
retains source:task shocks. Namespace seed draws cross every affected cell.
Fit/sample/learner draws remain inside cells, and query entities keep all rows.
This is development inference, never a calibration or qualification receipt.
"""
import math

from .losses import finite_number
from .tdl.common import MetricError, XorShift64, count, mean, number, require


ASSUMPTIONS = {"moment_assumption": "finite_variance", "distribution_free": False}


class _InferenceRng(XorShift64):
    """Full uint64 task substreams; legacy aggregate keeps its 53-bit admission."""
    def __init__(self, seed):
        require(type(seed) is int and 0 <= seed < 2**64, "invalid_bootstrap_seed")
        self.state = seed or 1


def crossed_bootstrap(design, *, endpoint="z", estimand="method_new_source", draws=10000, seed=0):
    from .statistics import _bootstrap_refusal, paired_family_effect
    point = paired_family_effect(design, endpoint=endpoint)

    def refuse(status, reason=None):
        sensitivity = {**point, "estimand": "complete_case_sensitivity"} if status == "incomplete" else point
        return _bootstrap_refusal(sensitivity, estimand, draws, status, reason)

    require(99 <= count(draws) <= 100000, "invalid_bootstrap_budget")
    rng = _InferenceRng(seed)
    sources = sorted({r.family_id for r in design.runs})
    tasks = sorted({r.task_axis_id for r in design.runs if r.task_axis_id})
    methods = sorted(design.required_methods)
    if any(r.pairs is None for r in design.runs):
        return refuse("incomplete", "incomplete scheduled evidence")
    if any(not r.task_axis_id for r in design.runs):
        return refuse("incomplete", "incomplete crossed task axis")
    cells, namespaces, query_ids = {}, {}, {}
    for source in sources:
        for task in tasks:
            resampling_axes = None
            for method in methods:
                runs = [r for r in design.runs if (r.family_id, r.task_axis_id, r.method_id) == (source, task, method)]
                if not runs:
                    return refuse("incomplete", "incomplete crossed task axis")
                fits = sorted({r.fit_id for r in runs})
                samples = sorted({r.sample_id for r in runs})
                learners = sorted({r.learner_seed_id for r in runs})
                axes = (samples, learners)
                if resampling_axes is not None and resampling_axes != axes:
                    return refuse("incomplete", "unbalanced paired seed axes")
                resampling_axes = axes
                lookup = {(r.fit_id, r.sample_id, r.learner_seed_id): r for r in runs}
                if len(lookup) != len(runs) or len(lookup) != len(fits) * len(samples) * len(learners):
                    return refuse("incomplete", "unbalanced scheduled evidence")
                first = runs[0]
                signature = (first.pairs.query_ids, first.pairs.cluster_ids)
                if query_ids.setdefault((source, task), signature) != signature or any(
                        (r.pairs.query_ids, r.pairs.cluster_ids) != signature for r in runs):
                    raise MetricError("unpaired_queries")
                namespace = first.global_seed_namespace
                seeds = [lookup[(f, samples[0], learners[0])].global_seed_id for f in fits]
                if namespace is not None:
                    if len(set(seeds)) != len(seeds):
                        raise MetricError("duplicate_fit_seed")
                    if namespaces.setdefault(namespace, sorted(seeds)) != sorted(seeds):
                        return refuse("incomplete", "incomplete crossed global seed axis")
                if estimand == "fixed_artifact" and (len(fits), len(samples), len(learners)) != (1, 1, 1):
                    raise MetricError("invalid_fixed_artifact_design")
                if estimand == "method_new_source" and len(fits) < 2:
                    return refuse("insufficient_fits")
                clusters = sorted(set(first.pairs.cluster_ids))
                if len(clusters) < 2:
                    return refuse("insufficient_query_clusters")
                values = {k: tuple(float(v) for v in r.values(endpoint)) for k, r in lookup.items()}
                cells[(source, task, method)] = (fits, samples, learners, namespace, seeds, clusters, first.pairs.cluster_ids, values)
    if estimand == "method_new_source" and len(sources) < 2:
        return refuse("insufficient_families")
    if estimand == "method_new_source" and len(tasks) < 2:
        return refuse("insufficient_tasks")
    family_effects = {m: {s: mean([mean([mean(v) for _, v in sorted(cells[(s, t, m)][7].items())]) for t in tasks])
                          for s in sources} for m in methods}
    point = {**point, "family_effects": family_effects,
             "estimates": {m: mean([family_effects[m][s] for s in sources]) for m in methods}}
    output = {m: [] for m in methods}
    for _ in range(draws):
        global_picks = {ns: [ids[rng.index(len(ids))] for _ in ids] for ns, ids in sorted(namespaces.items())} if estimand == "method_new_source" else {}
        task_values = {}
        for source in sources:
            for task in tasks:
                clusters = cells[(source, task, methods[0])][5]
                picks = [clusters[rng.index(len(clusters))] for _ in clusters]
                weights = [picks.count(c) for c in cells[(source, task, methods[0])][6]]
                sample_draws, learner_draws = {}, {}
                for method in methods:
                    fits, samples, learners, ns, seeds, _, _, values = cells[(source, task, method)]
                    selected_fits = fits if estimand == "fixed_artifact" else (
                        [fits[seeds.index(s)] for s in global_picks[ns]] if ns is not None else [fits[rng.index(len(fits))] for _ in fits])
                    fit_means = []
                    for fi, fit in enumerate(selected_fits):
                        if fi not in sample_draws:
                            sample_draws[fi] = samples if estimand == "fixed_artifact" else [samples[rng.index(len(samples))] for _ in samples]
                        selected_samples = sample_draws[fi]
                        sample_means = []
                        for si, sample in enumerate(selected_samples):
                            if (fi, si) not in learner_draws:
                                learner_draws[(fi, si)] = learners if estimand == "fixed_artifact" else [learners[rng.index(len(learners))] for _ in learners]
                            selected_learners = learner_draws[(fi, si)]
                            sample_means.append(mean([sum(v * w for v, w in zip(values[(fit, sample, l)], weights)) / sum(weights) for l in selected_learners]))
                        fit_means.append(mean(sample_means))
                    task_values[(source, task, method)] = mean(fit_means)
        source_picks = sources if estimand == "fixed_artifact" else [sources[rng.index(len(sources))] for _ in sources]
        task_picks = tasks if estimand == "fixed_artifact" else [tasks[rng.index(len(tasks))] for _ in tasks]
        for method in methods:
            output[method].append(mean([mean([task_values[(s, t, method)] for t in task_picks]) for s in source_picks]))
    return {**point, "estimand": estimand, "draws": draws, "status": "complete", "samples": output, **ASSUMPTIONS}


def max_t_bounds(bootstrap, *, alpha=0.05):
    """Full-family simultaneous bounds plus Romano-Wolf step-down zero tests.

    Bounds keep the full family even after a zero-null is rejected. Subset
    critical values are used only for adjusted tests, never misrepresented as
    unconditional simultaneous confidence bounds. Each tail has level 1-alpha.
    """
    alpha = number(alpha)
    require(0 < alpha < 0.5, "invalid_inference_alpha")
    if bootstrap["status"] != "complete":
        return {"status": bootstrap["status"], "bounds": None, **ASSUMPTIONS}
    draws = bootstrap["draws"]
    if alpha * draws < 5:
        return {"status": "insufficient_tail_draws", "bounds": None, **ASSUMPTIONS}
    names = sorted(bootstrap["samples"])
    require(bool(names), "invalid_bootstrap_samples")
    points = [finite_number(bootstrap["estimates"][m]) for m in names]
    columns = [[finite_number(v) for v in bootstrap["samples"][m]] for m in names]
    require(all(len(c) == draws for c in columns), "invalid_bootstrap_samples")
    centers = [mean(c) for c in columns]
    scales = [math.sqrt(sum((v - center) ** 2 for v in c) / (draws - 1)) for c, center in zip(columns, centers)]
    if any(s <= 0 or not math.isfinite(s) for s in scales):
        return {"status": "insufficient_variation", "bounds": None, **ASSUMPTIONS}
    deviations = [[(v - p) / s for v in c] for c, p, s in zip(columns, points, scales)]
    bounds = {m: {"estimate": p} for m, p in zip(names, points)}
    for sign, bound_name in ((1, "lower"), (-1, "upper")):
        maxima = sorted(max(sign * c[b] for c in deviations) for b in range(draws))
        position = (1 - alpha) * (draws - 1)
        lo = math.floor(position)
        critical = max(0., maxima[lo] + (maxima[math.ceil(position)] - maxima[lo]) * (position - lo))
        order = sorted(range(len(names)), key=lambda i: (-sign * points[i] / scales[i], names[i]))
        previous = 0.
        for rank, i in enumerate(order):
            observed = sign * points[i] / scales[i]
            exceed = sum(max(sign * deviations[j][b] for j in order[rank:]) >= observed for b in range(draws))
            previous = max(previous, (1 + exceed) / (draws + 1))
            bounds[names[i]][bound_name] = number(points[i] - sign * critical * scales[i])
            bounds[names[i]][bound_name + "_stepdown_p"] = previous
    return {"status": "complete", "alpha": alpha, "multiplicity": "step_down_max_t",
            "bound_method": "full_family_max_t", "simultaneous_one_sided_level": 1 - alpha,
            "two_sided_level_at_least": 1 - 2 * alpha, "bounds": bounds, **ASSUMPTIONS}


def design_from_payload(data):
    """Private evidence adapter; callers sanitize any deserialization failure."""
    from .losses import PairedLosses, TrainingScale
    from .metrics import Margin
    from .statistics import CrossedDesign, EvaluationRun
    require(type(data) is dict and type(data.get("inference_version")) is int and data["inference_version"] == 2, "invalid_inference_version")
    require(data.get("cluster_unit") == "source_family", "invalid_cluster_unit")
    require(all(type(data.get(k)) is list for k in ("runs", "required_methods", "scheduled_keys")), "invalid_design_roster")
    runs = []
    for item in data["runs"]:
        require(type(item) is dict, "invalid_run")
        pairs = item.get("pairs")
        if pairs is not None:
            pairs = PairedLosses(**pairs)
        runs.append(EvaluationRun(**{**item, "pairs": pairs, "scale": TrainingScale(item["scale"]), "margin": Margin(**item["margin"])}))
    require(all(r.task_axis_id for r in runs), "missing_task_axis")
    return CrossedDesign(tuple(runs), tuple(data["required_methods"]), tuple(tuple(k) for k in data["scheduled_keys"]), data["cluster_unit"])


def evaluate(data):
    """Explicit candidate interface; legacy aggregate receipts keep v1 semantics."""
    from .statistics import bootstrap_effects, simultaneous_ucb
    try:
        design = design_from_payload(data)
        boot = bootstrap_effects(design, endpoint=data.get("endpoint", "z"), estimand=data.get("estimand", "method_new_source"),
                                 draws=data.get("bootstrap_draws", 10000), seed=data.get("bootstrap_seed", 0))
        result = simultaneous_ucb(boot, alpha=data.get("alpha", 0.05))
        return {**result, "qualification": False, "calibration": "unmeasured", "campaign": "unmeasured"}
    except (ValueError, TypeError, KeyError, OverflowError):
        # Deserialization exceptions may contain private identifiers or values.
        return {"status": "failed", "reason": "invalid_crossed_design"}
