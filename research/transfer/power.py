"""Development-only Monte Carlo validation; no paid fits or protected data.

Coverage, false support and power are measured separately. Monte Carlo error is
reported using a Wilson interval, not inferred from deterministic output hashes.
An absent validation receipt keeps the statistics protocol unfrozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .losses import PairedLosses, TrainingScale, finite_number
from .metrics import Margin
from .statistics import CrossedDesign, EvaluationRun, bootstrap_effects, interval_decision, simultaneous_ucb


@dataclass(frozen=True)
class PowerScenario:
    effect: float
    families: int = 24
    fits: int = 12
    query_clusters: int = 12
    rows_per_cluster: int = 2
    family_sd: float = 0.3
    seed_sd: float = 0.3
    query_sd: float = 0.2
    heavy_tail_df: float | None = None
    shared_global_seeds: bool = True
    unequal_family_sizes: bool = False
    missing_fit_probability: float = 0.0
    equivalence_margin: float = 0.2

    def __post_init__(self):
        finite_number(self.effect)
        for value in (self.families, self.fits, self.query_clusters, self.rows_per_cluster):
            if type(value) is not int or value < 1:
                raise ValueError("invalid simulation size")
        if any(finite_number(v) < 0 for v in (self.family_sd, self.seed_sd, self.query_sd)):
            raise ValueError("invalid variance components")
        if self.heavy_tail_df is not None and finite_number(self.heavy_tail_df) <= 2:
            raise ValueError("finite-variance heavy-tail validation requires df > 2")
        if not 0 <= finite_number(self.missing_fit_probability) < 1 or finite_number(self.equivalence_margin) <= 0:
            raise ValueError("invalid missingness or equivalence policy")


def _generate(scenario: PowerScenario, rng) -> CrossedDesign:
    global_shocks = rng.normal(0, scenario.seed_sd, scenario.fits)
    runs = []
    for family in range(scenario.families):
        family_shock = rng.normal(0, scenario.family_sd)
        ntasks = 1 + family % 3 if scenario.unequal_family_sizes else 1
        for task in range(ntasks):
            nclusters = scenario.query_clusters + (family % 3 if scenario.unequal_family_sizes else 0)
            if scenario.heavy_tail_df is None:
                query = rng.normal(0, scenario.query_sd, nclusters)
            else:
                df = scenario.heavy_tail_df
                query = rng.standard_t(df, nclusters) * scenario.query_sd * math.sqrt((df - 2) / df)
            repeats = np.asarray([1 + (family + i) % 3 for i in range(nclusters)]) if scenario.unequal_family_sizes else np.full(nclusters, scenario.rows_per_cluster)
            query = np.repeat(query, repeats)
            queries = tuple(f"q-{i}" for i in range(len(query)))
            clusters = tuple(f"e-{i}" for i, count in enumerate(repeats) for _ in range(count))
            fit_shocks = global_shocks if scenario.shared_global_seeds else rng.normal(0, scenario.seed_sd, scenario.fits)
            for fit, shock in enumerate(fit_shocks):
                diff = scenario.effect + family_shock + shock + query
                # Positive raw losses with exactly the desired, unclipped gap.
                real = np.abs(diff) + 1
                pairs = PairedLosses(queries, clusters, tuple(real), tuple(real + diff), tuple(real + 2))
                missing = rng.random() < scenario.missing_fit_probability
                runs.append(EvaluationRun(f"f-{family}", f"t-{family}-{task}", "synthetic-minus-real", str(fit), "sample", "learner",
                                          None if missing else pairs, TrainingScale(1.), Margin(0., 0.),
                                          "technical_failure" if missing else "measured",
                                          "historical-panel" if scenario.shared_global_seeds else None,
                                          str(fit) if scenario.shared_global_seeds else None))
    return CrossedDesign(tuple(runs), ("synthetic-minus-real",), tuple(r.key for r in runs))


def _rate(count: int, total: int) -> dict:
    if not total:
        return {"count": count, "denominator": total, "rate": None, "mc_se": None, "mc95": None}
    p = count / total
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"count": count, "denominator": total, "rate": p,
            "mc_se": math.sqrt(p * (1 - p) / total), "mc95": [max(0., center - half), min(1., center + half)]}


def power_simulation(scenario: PowerScenario, *, repetitions: int = 120, bootstrap_draws: int = 399,
                     seed: int = 0, alpha: float = 0.05, coverage_tolerance: float = 0.05,
                     false_positive_tolerance: float = 0.05) -> dict:
    """Validate one frozen synthetic scenario; every attempted repetition remains.

    Coverage is one-sided UCB coverage. False noninferiority is only defined
    for truly positive effects. Equivalence uses the simultaneous two-sided
    bounds inside the declared margin, never a nonsignificant difference test.
    Missing-fit simulations must fail closed instead of earning qualification.
    """
    if type(repetitions) is not int or repetitions < 20 or type(seed) is not int or seed < 0:
        raise ValueError("invalid Monte Carlo budget")
    alpha = finite_number(alpha)
    if not 0 < alpha < 0.5 or not 0 <= finite_number(coverage_tolerance) < 1 or not 0 <= finite_number(false_positive_tolerance) < 1:
        raise ValueError("invalid Monte Carlo tolerances")
    rng = np.random.default_rng(seed)
    counts = {k: 0 for k in ("valid", "upper_covered", "interval_covered", "noninferior", "superior", "harmful", "equivalent")}
    widths = []
    failures = {}
    for _ in range(repetitions):
        design = _generate(scenario, rng)
        boot = bootstrap_effects(design, draws=bootstrap_draws, seed=int(rng.integers(2**63)))
        result = simultaneous_ucb(boot, alpha=alpha)
        if result["status"] != "complete":
            failures[result["status"]] = failures.get(result["status"], 0) + 1
            continue
        bounds = result["bounds"]["synthetic-minus-real"]
        lower, upper = bounds["lower"], bounds["upper"]
        decision = interval_decision(lower, upper, endpoint=boot["endpoint"],
                                     equivalence_margin=scenario.equivalence_margin)
        counts["valid"] += 1
        counts["upper_covered"] += scenario.effect <= upper
        counts["interval_covered"] += lower <= scenario.effect <= upper
        for key in ("noninferior", "superior", "harmful", "equivalent"):
            counts[key] += decision[key]
        widths.append(upper - lower)
    valid = counts["valid"]
    metrics = {key: _rate(value, valid) for key, value in counts.items() if key != "valid"}
    # No-significance is never interpreted as equivalence or as support.
    false_noninferiority = metrics["noninferior"] if scenario.effect > 0 else None
    false_equivalence = metrics["equivalent"] if abs(scenario.effect) >= scenario.equivalence_margin else None
    coverage_ok = valid == repetitions and metrics["upper_covered"]["rate"] >= 1 - alpha - coverage_tolerance
    false_positive_ok = all(metric is None or metric["rate"] <= alpha + false_positive_tolerance for metric in (false_noninferiority, false_equivalence))
    return {"repetitions": repetitions, "valid_repetitions": valid, "failed_repetitions": failures,
            "alpha": alpha, "coverage_tolerance": coverage_tolerance,
            "false_positive_tolerance": false_positive_tolerance, "metrics": metrics,
            "false_noninferiority": false_noninferiority, "false_equivalence": false_equivalence,
            "mean_interval_width": float(np.mean(widths)) if widths else None,
            "validation_passed": coverage_ok and false_positive_ok,
            "qualification_authority": False}
