"""predeclare_v2 statistics: nested bootstrap, Hodges-Lehmann, robust equivalence.

The procedures are fixed by research/benchmark/review_fixes/predeclare_v2.json.
A fit value is one fit's sample-seed median. A lineage value is the median of
its fit values. Uncertainty resamples clusters, then lineages inside a drawn
cluster, then fit seeds inside a drawn lineage, separately for each method.
Nothing here chooses a margin, a seed, or a configuration from a result.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np
from scipy import stats

from research.benchmark.review_fixes.stats import holm, wilcoxon_p, win_tie_loss  # noqa: F401

DRAWS = 10_000
MARGIN = 0.02
LABEL_PREFIX = "review-fixes-v2|"
MIN_FITS_FIVE_SEED = 3


def rng_for(label: str) -> np.random.Generator:
    digest = hashlib.sha256((LABEL_PREFIX + label).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def hodges_lehmann(diffs) -> float | None:
    """Median of the Walsh averages (d_i + d_j) / 2 for i <= j."""
    array = np.asarray([float(item) for item in diffs if _finite(item)], dtype=float)
    if array.size == 0:
        return None
    upper = np.triu_indices(array.size)
    return float(np.median((array[upper[0]] + array[upper[1]]) / 2.0))


def lineage_table(items, min_fits: int = 1) -> dict[str, dict]:
    """Group fit values by lineage. A lineage needs at least min_fits fits."""
    grouped: dict[str, dict] = {}
    seen = set()
    for item in items:
        key = (item["dataset"], item["fit_seed"])
        if key in seen:
            raise ValueError("duplicate fit value for one lineage and fit seed")
        seen.add(key)
        if not _finite(item["value"]):
            raise ValueError("fit values must be finite")
        slot = grouped.setdefault(item["dataset"], {"cluster": item["cluster"], "values": []})
        if slot["cluster"] != item["cluster"]:
            raise ValueError("one lineage has two clusters")
        slot["values"].append(float(item["value"]))
    return {key: slot for key, slot in grouped.items() if len(slot["values"]) >= min_fits}


def _clusters(lineages) -> list[list[str]]:
    members: dict[str, list[str]] = defaultdict(list)
    for dataset in sorted(lineages):
        members[lineages[dataset]["cluster"]].append(dataset)
    return [members[key] for key in sorted(members)]


class _Design:
    """Padded arrays for a vectorized cluster -> lineage -> fit bootstrap."""

    def __init__(self, order: list[str], clusters: list[list[str]]):
        position = {dataset: index for index, dataset in enumerate(order)}
        self.sizes = np.asarray([len(group) for group in clusters])
        width = int(self.sizes.max())
        self.members = np.zeros((len(clusters), width), dtype=int)
        for row, group in enumerate(clusters):
            self.members[row, :len(group)] = [position[dataset] for dataset in group]
        self.columns = np.arange(width)[None, :]

    def lineages(self, rng) -> np.ndarray:
        chosen = rng.integers(0, len(self.sizes), size=len(self.sizes))
        sizes = self.sizes[chosen][:, None]
        picks = np.floor(rng.random((len(chosen), self.members.shape[1])) * sizes).astype(int)
        return self.members[chosen[:, None], picks][self.columns < sizes]


def _padded(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([item.size for item in values])
    table = np.full((len(values), int(counts.max())), np.nan)
    for row, item in enumerate(values):
        table[row, :item.size] = item
    return table, counts


def _fit_medians(rng, table: np.ndarray, counts: np.ndarray, drawn: np.ndarray) -> np.ndarray:
    """Median of a with-replacement fit resample inside each drawn lineage."""
    width = table.shape[1]
    sizes = counts[drawn]
    picks = np.floor(rng.random((drawn.size, width)) * sizes[:, None]).astype(int)
    values = table[drawn[:, None], picks]
    values[np.arange(width)[None, :] >= sizes[:, None]] = np.inf
    values.sort(axis=1)
    rows = np.arange(drawn.size)
    return (values[rows, (sizes - 1) // 2] + values[rows, sizes // 2]) / 2.0


def _walsh_median(values: np.ndarray) -> float:
    upper = np.triu_indices(values.size)
    return float(np.median((values[upper[0]] + values[upper[1]]) / 2.0))


def nested_level(items, label: str, min_fits: int = 1, draws: int = DRAWS) -> dict:
    """Median of lineage values with a nested percentile interval."""
    lineages = lineage_table(items, min_fits)
    n = len(lineages)
    if n == 0:
        return {"n": 0, "n_fits": 0, "median": None, "lo": None, "hi": None}
    order = sorted(lineages)
    arrays = [np.asarray(lineages[key]["values"]) for key in order]
    point = float(np.median([np.median(values) for values in arrays]))
    n_fits = sum(values.size for values in arrays)
    if n == 1:
        return {"n": 1, "n_fits": n_fits, "median": point, "lo": None, "hi": None,
                "ci_status": "not_identified_single_lineage"}
    rng = rng_for(label + "|level")
    design = _Design(order, _clusters(lineages))
    table, counts = _padded(arrays)
    samples = np.empty(draws)
    for draw in range(draws):
        samples[draw] = np.median(_fit_medians(rng, table, counts, design.lineages(rng)))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"n": n, "n_fits": n_fits, "median": point, "lo": float(lo), "hi": float(hi)}


def nested_paired(left_items, right_items, label: str, min_fits_left: int = 1,
                  min_fits_right: int = 1, draws: int = DRAWS) -> dict:
    """Left minus right on shared lineages: median difference and Hodges-Lehmann.

    Fit seeds are resampled separately for each side inside each drawn lineage.
    """
    left = lineage_table(left_items, min_fits_left)
    right = lineage_table(right_items, min_fits_right)
    shared = sorted(set(left) & set(right))
    for dataset in shared:
        if left[dataset]["cluster"] != right[dataset]["cluster"]:
            raise ValueError("paired lineages disagree on the cluster")
    left_arrays = [np.asarray(left[key]["values"]) for key in shared]
    right_arrays = [np.asarray(right[key]["values"]) for key in shared]
    diffs = [float(np.median(a) - np.median(b)) for a, b in zip(left_arrays, right_arrays)]
    result = {"n": len(shared), "median": None, "hl": None, "median_lo": None, "median_hi": None,
              "hl_lo": None, "hl_hi": None, "hl_lo90": None, "hl_hi90": None, "hl_p05": None,
              **win_tie_loss(diffs), "wilcoxon_p": wilcoxon_p(diffs), "differences": diffs,
              "lineages": shared}
    if not shared:
        return result
    result["median"] = float(np.median(diffs))
    result["hl"] = hodges_lehmann(diffs)
    if len(shared) == 1:
        result["ci_status"] = "not_identified_single_lineage"
        return result
    rng = rng_for(label + "|paired")
    design = _Design(shared, _clusters({key: left[key] for key in shared}))
    left_table, left_counts = _padded(left_arrays)
    right_table, right_counts = _padded(right_arrays)
    medians = np.empty(draws)
    walsh = np.empty(draws)
    for draw in range(draws):
        drawn = design.lineages(rng)
        sample = (_fit_medians(rng, left_table, left_counts, drawn)
                  - _fit_medians(rng, right_table, right_counts, drawn))
        medians[draw] = np.median(sample)
        walsh[draw] = _walsh_median(sample)
    result["median_lo"], result["median_hi"] = (float(x) for x in np.quantile(medians, [0.025, 0.975]))
    result["hl_lo"], result["hl_hi"] = (float(x) for x in np.quantile(walsh, [0.025, 0.975]))
    result["hl_lo90"], result["hl_hi90"] = (float(x) for x in np.quantile(walsh, [0.05, 0.95]))
    result["hl_p05"] = result["hl_lo90"]
    return result


def equivalence_hl(paired: dict, margin: float = MARGIN) -> dict:
    """Equivalent iff the 90% nested HL interval lies inside [-margin, margin]."""
    lo, hi = paired.get("hl_lo90"), paired.get("hl_hi90")
    if lo is None or hi is None:
        return {"margin": margin, "equivalent": None, "non_inferior": None}
    return {"margin": margin, "equivalent": bool(-margin < lo and hi < margin),
            "non_inferior": bool(lo > -margin)}


def yuen_tost(diffs, margin: float = MARGIN, trim: float = 0.2, alpha: float = 0.05) -> dict:
    """Two one-sided Yuen tests of the trimmed mean against +/- margin."""
    array = np.sort(np.asarray([float(item) for item in diffs if _finite(item)], dtype=float))
    n = array.size
    cut = int(math.floor(trim * n))
    kept = n - 2 * cut
    result = {"n": int(n), "trim": trim, "margin": margin, "trimmed_mean": None, "se": None,
              "p": None, "equivalent": None}
    if kept < 2:
        return result
    trimmed_mean = float(np.mean(array[cut:n - cut]))
    winsorized = array.copy()
    winsorized[:cut] = array[cut]
    winsorized[n - cut:] = array[n - cut - 1]
    variance = float(np.var(winsorized, ddof=1))
    se = math.sqrt(variance * (n - 1) / (kept * (kept - 1)))
    result["trimmed_mean"] = trimmed_mean
    result["se"] = se
    if se == 0:
        result["equivalent"] = bool(abs(trimmed_mean) < margin)
        result["p"] = 0.0 if result["equivalent"] else 1.0
        return result
    df = kept - 1
    p_lower = float(1 - stats.t.cdf((trimmed_mean + margin) / se, df))
    p_upper = float(stats.t.cdf((trimmed_mean - margin) / se, df))
    result["p"] = max(p_lower, p_upper)
    result["equivalent"] = bool(p_lower < alpha and p_upper < alpha)
    return result


def seed_rank_test(rows) -> dict:
    """Count lineages where fit seed 11 has the highest of its five fit values."""
    by_lineage: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_lineage[row["dataset"]][row["fit_seed"]] = float(row["value"])
    complete = [values for values in by_lineage.values() if set(values) == {11, 23, 37, 53, 71}]
    top = sum(1 for values in complete if values[11] == max(values.values())
              and sum(1 for value in values.values() if value == values[11]) == 1)
    n = len(complete)
    p = float(stats.binomtest(top, n, 0.2, alternative="greater").pvalue) if n else None
    return {"n": n, "seed11_top": top, "expected": 0.2 * n, "p_one_sided": p}
