"""Pre-declared review-fix statistics.

The procedures are fixed by research/benchmark/review_fixes/predeclare.json.
Nothing here chooses a margin, a seed, or a configuration from a result.
"""

from __future__ import annotations

import hashlib
import math
import sys

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
    sizes = np.asarray([len(group) for group in members], dtype=np.int64)
    tape = _WordTape(rng)
    samples, used = _hier_samples(np.concatenate(members), sizes, tape, DRAWS)
    tape.commit(used)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"median": float(np.median(array)), "lo": float(lo), "hi": float(hi)}


# The historical cluster bootstrap made, per draw, one rng.integers(0, C, size=C) call for the
# C clusters, then one rng.integers(0, s, size=s) call per chosen cluster with s > 1 members,
# and took the median of the chosen members in that order (a range of one draws nothing, so a
# singleton contributes its only member). For int64 bounds below 2**32 NumPy draws each such
# integer with Lemire's method from the next 32-bit word of the bit generator, which buffers
# the second half of each 64-bit output; integers(0, 2**32, dtype=uint64) returns exactly those
# words. The code below applies the same rule to the same words for all draws at once, so every
# median, every interval and the generator's final state are unchanged.
_WORD = 1 << 32
_LOW32 = np.uint64(0xFFFFFFFF)
_SHIFT = np.uint64(32)


class _WordTape:
    """The generator's 32-bit words, drawn ahead in bulk.

    PCG64 (default_rng) hands out the low half of each 64-bit output, then the buffered high
    half, so its words are the halves of random_raw outputs in that order, after any half
    already buffered. commit(used) restores the generator and advances it by exactly the words
    used, so drawing ahead never shows: the generator ends where the per-value calls left it.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.saved = rng.bit_generator.state
        # On a little-endian host the uint32 view of the outputs is their halves, low first.
        self.raw = type(rng.bit_generator) is np.random.PCG64 and sys.byteorder == "little"
        self.buffered = 1 if self.raw and self.saved["has_uint32"] else 0
        self.words = np.asarray([self.saved["uinteger"]] if self.buffered else [], dtype=np.uint32)

    def get(self, stop: int) -> np.ndarray:
        if stop > self.words.size:
            more = max(stop - self.words.size, self.words.size, 1 << 16)
            if self.raw:
                fresh = self.rng.bit_generator.random_raw((more + 1) // 2).view(np.uint32)
            else:
                fresh = self.rng.integers(0, _WORD, size=more, dtype=np.uint64).astype(np.uint32)
            self.words = np.concatenate([self.words, fresh]) if self.words.size else fresh
        return self.words

    def commit(self, used: int) -> None:
        self.rng.bit_generator.state = self.saved
        if not self.raw:
            if used:
                self.rng.integers(0, _WORD, size=used, dtype=np.uint64)
            return
        head = min(used, self.buffered)
        if head:
            self.rng.integers(0, _WORD, size=head, dtype=np.uint64)
        rest = used - head
        if rest:
            # Skip whole outputs in bulk; the last one or two words go through integers so the
            # buffered half is left exactly as the per-value calls left it.
            last = 2 if rest % 2 == 0 else 1
            if rest > last:
                self.rng.bit_generator.random_raw((rest - last) // 2)
            self.rng.integers(0, _WORD, size=last, dtype=np.uint64)


def _below(tape, position: int, bound: int) -> tuple[int, int]:
    """NumPy's Lemire draw in [0, bound) from the words at position; returns (value, next)."""
    threshold = (_WORD - bound) % bound
    while True:
        value = int(tape.get(position + 1)[position]) * bound
        position += 1
        if value & 0xFFFFFFFF >= threshold:
            return value >> 32, position


def _exact_draw(flat, sizes, starts, tape, position: int) -> tuple[float, int]:
    """One bootstrap draw exactly as the historical per-value calls made it."""
    n_clusters = sizes.size
    chosen = []
    for _ in range(n_clusters if n_clusters > 1 else 0):
        index, position = _below(tape, position, n_clusters)
        chosen.append(index)
    picked = []
    for index in chosen or [0]:
        size, start = int(sizes[index]), int(starts[index])
        if size == 1:
            picked.append(flat[start])
            continue
        for _ in range(size):
            local, position = _below(tape, position, size)
            picked.append(flat[start + local])
    return float(np.median(np.asarray(picked, dtype=float))), position


def _hier_samples(flat, sizes, tape, draws: int) -> tuple[np.ndarray, int]:
    """All bootstrap medians and the number of words they used."""
    n_clusters = sizes.size
    starts = np.cumsum(sizes) - sizes
    samples = np.empty(draws, dtype=float)
    position = done = 0
    signed_zero = bool(np.any((flat == 0) & np.signbit(flat)))
    if n_clusters == 1:  # One cluster draws no cluster word.
        for draw in range(draws):
            samples[draw], position = _exact_draw(flat, sizes, starts, tape, position)
        return samples, position
    member_words = np.where(sizes > 1, sizes, 0)  # Words a chosen cluster spends on its members.
    per_draw = n_clusters + int(member_words.sum())
    block = max(16, 15_000 // per_draw)
    # An array, not a scalar: NumPy 1.x would keep uint32 words * a small uint64 scalar in uint32.
    bound = np.asarray([n_clusters], dtype=np.uint64)
    cluster_threshold = (_WORD - n_clusters) % n_clusters
    span = np.arange(n_clusters)
    tape.get(position + draws * per_draw * 21 // 20 + 4 * per_draw)
    while done < draws:
        count = min(block, draws - done)
        # Each draw spends C cluster words, then one word per member of each chosen
        # multi-member cluster; the next draw starts after them.
        offsets, multis, cursor = [], [], 0
        for _ in range(count):
            words = tape.get(position + cursor + n_clusters)
            start = position + cursor
            taken = int(member_words[(words[start:start + n_clusters] * bound) >> _SHIFT].sum())
            offsets.append(cursor)
            multis.append(taken)
            cursor += n_clusters + taken
        window = tape.get(position + cursor)[position:position + cursor]
        offsets = np.asarray(offsets, dtype=np.int64)
        multis = np.asarray(multis, dtype=np.int64)
        kept = offsets.size
        cluster_scaled = window[offsets[:, None] + span] * bound
        chosen = (cluster_scaled >> _SHIFT).astype(np.int64)
        rejected = ((cluster_scaled & _LOW32) < cluster_threshold).any(axis=1)
        lengths = sizes[chosen]
        totals = lengths.sum(axis=1)
        cluster = np.repeat(chosen.ravel(), lengths.ravel())
        draw_of = np.repeat(np.arange(kept), totals)
        multi = sizes[cluster] > 1
        before = np.cumsum(multis) - multis
        word_at = np.repeat(offsets + n_clusters - before, multis) + np.arange(int(multis.sum()))
        member_bounds = sizes[cluster[multi]].astype(np.uint64)
        member_scaled = window[word_at] * member_bounds
        member_rejected = (member_scaled & _LOW32) < (_WORD - member_bounds) % member_bounds
        rejected[draw_of[multi][member_rejected]] = True
        good = int(np.argmax(rejected)) if rejected.any() else kept
        if good:
            local = np.zeros(cluster.size, dtype=np.int64)
            local[multi] = (member_scaled >> _SHIFT).astype(np.int64)
            used = int(totals[:good].sum())
            values = flat[starts[cluster[:used]] + local[:used]]
            widths = totals[:good]
            grid = np.full((good, int(widths.max())), np.inf)
            grid[draw_of[:used], np.arange(used) - np.repeat(np.cumsum(widths) - widths, widths)] = values
            grid.sort(axis=1)
            rows, half = np.arange(good), widths // 2
            upper = grid[rows, half]
            lower = grid[rows, np.maximum(half - 1, 0)]
            medians = np.where(widths % 2 == 1, upper, (lower + upper) / 2)
            if signed_zero:
                # A sort and np.median's partition can return -0.0 and 0.0 in either order, so
                # a zero median is taken again by np.median from the values in drawn order.
                ends = np.cumsum(widths)
                for row in np.flatnonzero(medians == 0):
                    medians[row] = float(np.median(values[ends[row] - widths[row]:ends[row]]))
            samples[done:done + good] = medians
            done += good
        if good < kept:
            # Lemire rejected a word in this draw, shifting every later word: replay it exactly.
            samples[done], position = _exact_draw(flat, sizes, starts, tape, position + int(offsets[good]))
            done += 1
        else:
            position += int(offsets[-1]) + n_clusters + int(multis[-1])
    return samples, position


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
