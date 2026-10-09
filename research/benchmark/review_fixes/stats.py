"""Pre-declared review-fix statistics.

The procedures are fixed by research/benchmark/review_fixes/predeclare.json.
Nothing here chooses a margin, a seed, or a configuration from a result.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy import stats

DRAWS = 10_000
MARGIN = 0.02
ALPHA = 0.05
SIMULATED = ("feynman", "strogatz", "fri", "bng")


def family_of(name: str) -> str:
    text = str(name).lower()
    if text.startswith("feynman"):
        return "feynman"
    if text.startswith("strogatz"):
        return "strogatz"
    if "_fri_" in text or text.startswith("fri"):
        return "fri"
    if "bng" in text:
        return "bng"
    return "other"


def cluster_of(name: str, dataset: str) -> str:
    """Generator prefixes share a cluster. Every other lineage is its own cluster."""
    family = family_of(name)
    if family in SIMULATED:
        return family
    return "lineage:" + dataset


def rng_for(label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"review-fixes-v1|{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def finite_array(values) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def median_ci(values, label: str, cluster_ids=None) -> dict:
    if cluster_ids is None:
        array = finite_array(values)
        clusters = None
    else:
        values = list(values)
        cluster_ids = list(cluster_ids)
        if len(values) != len(cluster_ids):
            raise ValueError("cluster ids do not match values")
        kept = [
            (float(value), cluster)
            for value, cluster in zip(values, cluster_ids)
            if np.isfinite(value)
        ]
        array = np.asarray([value for value, _cluster in kept], dtype=float)
        clusters = [cluster for _value, cluster in kept]
    n = int(array.size)
    if n == 0:
        return {"n": 0, "median": None, "lo": None, "hi": None, "iid_lo": None, "iid_hi": None}
    point = float(np.median(array))
    if n == 1:
        return {"n": 1, "median": point, "lo": point, "hi": point, "iid_lo": point, "iid_hi": point}
    iid = _iid_median_ci(array, rng_for(label + "|iid"))
    hier = iid if clusters is None else _hier_median_ci(array, clusters, rng_for(label + "|hier"))
    return {
        "n": n,
        "median": point,
        "lo": hier["lo"],
        "hi": hier["hi"],
        "iid_lo": iid["lo"],
        "iid_hi": iid["hi"],
    }


def _iid_median_ci(array: np.ndarray, rng: np.random.Generator) -> dict:
    n = int(array.size)
    point = float(np.median(array))
    index = rng.integers(0, n, size=(DRAWS, n))
    samples = np.median(array[index], axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"median": point, "lo": float(lo), "hi": float(hi)}


def _hier_median_ci(array: np.ndarray, cluster_ids, rng: np.random.Generator) -> dict:
    if len(cluster_ids) != len(array):
        raise ValueError("cluster ids do not match values")
    groups: dict[str, list[float]] = {}
    for value, cluster in zip(array, cluster_ids):
        groups.setdefault(str(cluster), []).append(float(value))
    ids = list(groups)
    members = [np.asarray(groups[key], dtype=float) for key in ids]
    if all(len(item) == 1 for item in members):
        return _iid_median_ci(np.array([item[0] for item in members]), rng)
    n_clusters = len(ids)
    width = n_clusters * max(len(group) for group in members)
    buffer = np.empty(width, dtype=float)
    samples = np.empty(DRAWS, dtype=float)
    for draw in range(DRAWS):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        cursor = 0
        for index in chosen:
            group = members[int(index)]
            # NumPy's high=1 draw is deterministic and leaves the RNG unchanged.
            # Preserve the historical stream while avoiding a call per singleton.
            if len(group) == 1:
                buffer[cursor] = group[0]
                cursor += 1
                continue
            take = rng.integers(0, len(group), size=len(group))
            nxt = cursor + len(group)
            buffer[cursor:nxt] = group[take]
            cursor = nxt
        samples[draw] = float(np.median(buffer[:cursor]))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"median": float(np.median(array)), "lo": float(lo), "hi": float(hi)}


def holm(p_values):
    count = len(p_values)
    order = sorted(range(count), key=lambda i: (p_values[i] is None, p_values[i] if p_values[i] is not None else 0))
    adjusted = [None] * count
    running = 0.0
    for rank, index in enumerate(order):
        if p_values[index] is None:
            continue
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def wilcoxon_p(diff) -> float | None:
    array = finite_array(diff)
    if array.size == 0:
        return None
    if np.all(array == 0):
        return 1.0
    try:
        return float(stats.wilcoxon(array, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
    except ValueError:
        return None


def win_tie_loss(diff) -> dict:
    array = finite_array(diff)
    return {
        "wins": int(np.sum(array > 0)),
        "ties": int(np.sum(array == 0)),
        "losses": int(np.sum(array < 0)),
    }


def tost_mean(diff, margin: float = MARGIN, alpha: float = ALPHA) -> dict:
    """Two one-sided t-tests of the mean against ±margin."""
    array = finite_array(diff)
    n = int(array.size)
    result = {
        "n": n,
        "mean": None,
        "se": None,
        "margin": margin,
        "alpha": alpha,
        "p_lower": None,
        "p_upper": None,
        "p": None,
        "equivalent": False,
        "ci90_lo": None,
        "ci90_hi": None,
    }
    if n < 2:
        return result
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(n))
    result["mean"] = mean
    result["se"] = se
    if se == 0:
        inside = abs(mean) < margin
        result["p_lower"] = 0.0 if mean > -margin else 1.0
        result["p_upper"] = 0.0 if mean < margin else 1.0
        result["p"] = max(result["p_lower"], result["p_upper"])
        result["equivalent"] = inside
        result["ci90_lo"] = mean
        result["ci90_hi"] = mean
        return result
    df = n - 1
    # H0: mean <= -margin. Reject when the mean is sufficiently above -margin.
    t_lower = (mean - (-margin)) / se
    p_lower = float(1 - stats.t.cdf(t_lower, df))
    # H0: mean >= +margin. Reject when the mean is sufficiently below +margin.
    t_upper = (mean - margin) / se
    p_upper = float(stats.t.cdf(t_upper, df))
    critical = float(stats.t.ppf(1 - alpha, df))
    result["p_lower"] = p_lower
    result["p_upper"] = p_upper
    result["p"] = max(p_lower, p_upper)
    result["equivalent"] = bool(p_lower < alpha and p_upper < alpha)
    result["ci90_lo"] = mean - critical * se
    result["ci90_hi"] = mean + critical * se
    return result


def delta_method_se(tstr_losses, null_loss, trtr_loss) -> float | None:
    losses = finite_array(tstr_losses)
    if losses.size < 2 or not _finite(null_loss) or not _finite(trtr_loss):
        return None
    gap = float(null_loss) - float(trtr_loss)
    if gap == 0:
        return None
    return float(np.std(losses, ddof=1) / abs(gap))


def cap_retention(value, lo=-1.0, hi=2.0):
    if not _finite(value):
        return None, False
    number = float(value)
    if lo <= number <= hi:
        return number, False
    return float(min(hi, max(lo, number))), True


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
