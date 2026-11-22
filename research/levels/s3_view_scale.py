"""FIT-only statistics, lattice-aware noise and scaling of an S3 view (plan M3, stage C2).

All functions work on one stacked float64 matrix in column-major form: shape
``(columns, rows)``, one table column per contiguous row of memory, the first ``n_fit``
entries of every column being the FIT half and the rest the QUERY half. Every statistic is
read from the FIT entries only and the same transform is then applied to all entries, so
nothing can flow from QUERY values into the FIT half. There is no Python loop over columns:
one ``np.sort`` along the row axis gives medians, quartiles, extremes and distinct counts for
every column, and when only the order statistics are needed (no NaN, no lattice test) one
``np.partition`` gives the same numbers for a fraction of the cost.

Scaler modes:

* ``nc2_compat`` reproduces ``s3_dynamic_augment._fit_only_nc2_scale`` (robust centre,
  ``max(IQR / 1.349, std, 1e-6)`` scale, asinh tail beyond 3, clip at 6).
* ``robust_v2`` differs in two places: a FIT-constant column is zero in both halves and is
  flagged (the 1e-6 floor turned any differing QUERY value into a +-6 spike), and the scale is
  ``IQR / 1.349`` whenever the IQR is positive, the standard deviation only when it is zero.

Raw augmentation (before the scaler, as in ``s3_fast_pipeline``): Gaussian jitter of 0.2-2.5 %
of the IQR on continuous columns only, power and heaping warps on continuous columns only, and
MCAR / MAR missingness. A column is a lattice, and is left bit-identical, when it has at most
64 distinct values or its values sit on a regular grid; the test reads the first
``lattice_probe_rows`` FIT rows (all of them for the usual sizes), never a QUERY row.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

__all__ = ["FitStats", "fit_stats", "lattice_columns", "raw_augment", "scale_fit_only", "scale_stacked"]

_EPS32 = float(np.finfo(np.float32).eps)
_EUCLID_ROUNDS = 5
_ROUND_STEPS = np.array([0.05, 0.1, 0.25])
_QS = np.array([0.25, 0.5, 0.75])


class FitStats(NamedTuple):
    count: np.ndarray  # finite FIT values per column
    q25: np.ndarray
    med: np.ndarray
    q75: np.ndarray
    vmin: np.ndarray  # exact extremes after a sort; otherwise exact only where they are equal (constant column)
    vmax: np.ndarray
    ordered: np.ndarray | None  # the full sort (NaN last) when it was computed


def _lerp(a: np.ndarray, b: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """numpy's linear-method interpolation between neighbouring order statistics."""
    diff = b - a
    return np.where(gamma >= 0.5, b - diff * (1.0 - gamma), a + diff * gamma)


def fit_stats(fit: np.ndarray, has_nan: bool, want_sorted: bool = False) -> FitStats:
    """Order statistics of every FIT column at once; quantiles follow numpy's linear method.

    ``fit`` is (p, n) float64, one column per row, with every non-finite value already NaN.
    ``has_nan`` must be true when a NaN is present (the sort puts them last and each column
    then has its own count).
    """
    p, n = fit.shape
    if has_nan:
        s = np.sort(fit, axis=1)
        count = n - np.isnan(s).sum(axis=1)
        top = np.maximum(count - 1, 0)
        pos = top[:, None] * _QS  # (p, 3) virtual indices
        low = np.floor(pos)
        gamma, low = pos - low, low.astype(np.intp)
        rows = np.arange(p)[:, None]
        a, b = s[rows, low], s[rows, np.minimum(low + 1, top[:, None])]
        vmin, vmax = s[:, 0], s[rows[:, 0], top]
    else:
        last = n - 1
        low = [int(last * q) for q in (0.25, 0.5, 0.75)]
        if want_sorted:
            s = np.sort(fit, axis=1)
            vmin, vmax = s[:, 0], s[:, last]
            b = s[:, [min(k + 1, last) for k in low]]
        else:  # three selections; the upper neighbour of each is the minimum of the segment to its right
            s = np.partition(fit, low, axis=1)
            vmin = vmax = None
            b = np.empty((p, 3))
            for i, k in enumerate(low):  # three quantiles, not columns
                later = [v for v in low if v > k]
                b[:, i] = s[:, k + 1:later[0] + 1 if later else n].min(axis=1) if k < last else s[:, k]
        count = np.full(p, n)
        a, gamma = s[:, low], last * _QS - low
    q = _lerp(a, b, gamma)
    if vmin is None:  # extremes only decide "constant", which needs a zero IQR first
        vmin, vmax = q[:, 0], q[:, 2]
        flat = np.flatnonzero(vmin == vmax)
        if flat.size:
            vmin, vmax = vmin.copy(), vmax.copy()
            vmin[flat], vmax[flat] = fit[flat].min(axis=1), fit[flat].max(axis=1)
    med = np.where(gamma[..., 1] == 0.0, a[:, 1], (a[:, 1] + b[:, 1]) * 0.5)  # np.median: mean of the middle pair
    return FitStats(count, q[:, 0], med, q[:, 2], vmin, vmax, s if has_nan or want_sorted else None)


def _on_grid(block: np.ndarray, gaps: np.ndarray, step: np.ndarray, count: np.ndarray, max_grid: int) -> np.ndarray:
    """Sorted columns whose values sit on a regular grid of at most ``max_grid`` steps.

    ``step`` is the smallest positive gap of each column. Euclid's algorithm on neighbouring
    gaps finds the grid step, three anchored refinements take the float32 storage noise out of
    it, and every value must then be within 0.02 steps of a grid point (plus that noise).
    """
    c, m = block.shape
    out = np.zeros(c, dtype=bool)
    lo, hi = block[:, 0], block[np.arange(c), np.maximum(count - 1, 0)]
    span = hi - lo
    noise = 2.0 * _EPS32 * np.maximum(np.abs(lo), np.abs(hi))  # one stored float32 value
    floor = np.maximum(span / max_grid * (1.0 - 1e-9), 64.0 * noise)
    tiny = step <= 4.0 * noise
    if tiny.any():  # gaps inside the storage noise are not steps
        step = np.where(tiny, np.where(gaps > 4.0 * noise[:, None], gaps, np.inf).min(axis=1, initial=np.inf), step)
    idx = np.flatnonzero((step >= floor) & (step <= span))
    if not idx.size:
        return out
    gp, g, nz, fl = gaps[idx], step[idx], noise[idx, None], floor[idx]
    ragged = bool((count[idx] < m).any())
    if ragged:
        gp = np.where(np.isnan(gp), 0.0, gp)
    live, settled = np.ones(idx.size, dtype=bool), np.zeros(idx.size, dtype=bool)
    for _ in range(_EUCLID_ROUNDS):
        ratio = gp / g[:, None]
        res = np.abs(gp - g[:, None] * np.rint(ratio))
        off = (ratio <= 64.0) & (res > 3.0 * nz * (1.0 + ratio))
        smallest = np.where(off, res, np.inf).min(axis=1, initial=np.inf)
        agreed = live & ~np.isfinite(smallest)
        settled |= agreed
        live &= ~agreed
        g = np.where(live, smallest, g)
        live &= g >= fl
        if not live.any():
            break
    if not settled.any():
        return out
    v = block[idx] - lo[idx, None]
    if ragged:
        v = np.where(np.isnan(v), span[idx, None], v)
    reach = 64.0
    for _ in range(3):
        far = np.where(v <= reach * g[:, None], v, 0.0).max(axis=1)
        k = np.rint(far / g)
        g = np.where(k >= 1.0, far / np.maximum(k, 1.0), g)
        reach *= 64.0
    t = v / g[:, None]
    near = (np.abs(t - np.rint(t)) <= 0.02 + 4.0 * nz / g[:, None]).all(axis=1)
    out[idx] = settled & near & (np.rint(span[idx] / g) <= max_grid)
    return out


def lattice_columns(ordered: np.ndarray, count: np.ndarray, max_distinct: int = 64,
                    max_grid: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """(is_lattice, distinct) of column-sorted data (p, m), NaN last.

    A lattice has at most ``max_distinct`` distinct values or values on a regular grid. A column
    whose smallest gap is below 1 / ``max_grid`` of its range is continuous without further work.
    """
    gaps = np.diff(ordered, axis=1)  # NaN wherever a NaN is involved, so it never counts as a step
    positive = gaps > 0
    distinct = positive.sum(axis=1) + (count > 0)
    lattice = distinct <= max_distinct
    if not lattice.all():
        cand = np.flatnonzero(~lattice)
        gc = gaps[cand]
        step = np.where(positive[cand], gc, np.inf).min(axis=1, initial=np.inf)
        span = ordered[cand, np.maximum(count[cand] - 1, 0)] - ordered[cand, 0]
        fine = step >= span / max_grid * (1.0 - 1e-9)
        if fine.any():
            hit = cand[fine]
            lattice[hit] = _on_grid(ordered[hit], gc[fine], step[fine], count[hit], max_grid)
    return lattice, distinct


def _spread(fit: np.ndarray, count: np.ndarray | None = None) -> np.ndarray:
    """Population standard deviation of every column; ``count`` switches on NaN handling."""
    if count is None:
        dev = fit - (fit.sum(axis=1) / fit.shape[1])[:, None]
        return np.sqrt(np.einsum("ij,ij->i", dev, dev) / fit.shape[1])
    fin = ~np.isnan(fit)
    safe = np.maximum(count, 1)
    dev = np.where(fin, fit - (np.where(fin, fit, 0.0).sum(axis=1) / safe)[:, None], 0.0)
    return np.sqrt(np.einsum("ij,ij->i", dev, dev) / safe)


def raw_augment(x: np.ndarray, n_fit: int, rng: np.random.Generator, cfg: Any) -> dict[str, Any]:
    """Jitter, warps and missingness on the raw stacked view (p, rows), in place. Returns what was done.

    The noise law (sigma, warp exponent, heaping step, missingness rate and MAR threshold) is
    fixed from FIT rows and applied to both halves; the draws are independent per row.
    """
    p, n = x.shape
    jitter_on, warp_on, miss_on = (rng.random(3) < (cfg.jitter_prob, cfg.warp_prob, cfg.miss_prob)).tolist()
    done = {"jitter": False, "jitter_cols": 0, "lattice_cols": -1, "warp_cols": 0, "miss_cols": 0}
    if jitter_on or warp_on:
        probe = x[:, :min(n_fit, cfg.lattice_probe_rows)]
        finite = np.isfinite(probe)
        has_nan = not bool(finite.all())
        if has_nan:
            probe = np.where(finite, probe, np.nan)
        st = fit_stats(probe, has_nan, want_sorted=True)
        lattice, _ = lattice_columns(st.ordered, st.count, cfg.lattice_max_distinct, cfg.lattice_max_grid)
        done["lattice_cols"] = int(lattice.sum())
        cont = np.flatnonzero(~lattice)
        if cont.size:
            width = (st.q75 - st.q25)[cont]
            if not (width > 0).all():  # zero IQR with more than 64 distinct values: fall back to the spread
                width = np.where(width > 0, width, _spread(probe[cont], st.count[cont] if has_nan else None))
            if jitter_on:
                sigma = rng.uniform(cfg.jitter_lo, cfg.jitter_hi, cont.size) * width
                x[cont] += rng.standard_normal((cont.size, n)) * sigma[:, None]
                done["jitter"], done["jitter_cols"] = True, int(cont.size)
            if warp_on:
                u = rng.random((3, cont.size))
                kind = (u[1] * 3).astype(np.intp)  # 0 power, 1 heaping, 2 the dropped affine rescale (no-op)
                pick = u[0] < cfg.warp_col_prob
                power, heap = pick & (kind == 0), pick & (kind == 1)
                if power.any():  # sign(x) |x / m|^a m with m = max |x| over FIT: the legacy warp, made scale-free
                    v = x[cont[power]]
                    ref = np.nanmax(np.abs(v[:, :n_fit]), axis=1)[:, None]
                    ref = np.where(ref > 0, ref, 1.0)
                    x[cont[power]] = np.sign(v) * np.abs(v / ref) ** (1.3 + 1.5 * u[2, power])[:, None] * ref
                if heap.any():  # heaping on a step of 5-25 % of the FIT IQR (the legacy step was absolute)
                    step = (_ROUND_STEPS[(u[2, heap] * 3).astype(np.intp)] * width[heap])[:, None]
                    x[cont[heap]] = np.rint(x[cont[heap]] / step) * step
                done["warp_cols"] = int(power.sum() + heap.sum())
    if miss_on and p >= 3:
        k = 1 + int(rng.random() * (max(2, min(6, p // 2 + 1)) - 1))
        mcols = rng.permutation(p)[:k]
        u = rng.random((2, k))
        rate = 0.04 + 0.16 * u[0]
        mar = u[1] >= 0.5
        draw = rng.random((k, n))
        hole = draw < rate[:, None]  # MCAR
        if mar.any():  # MAR: driven by the upper tail of the next column, threshold from FIT rows
            driver = x[(mcols[mar] + 1) % p]
            ds = np.sort(driver[:, :n_fit], axis=1)
            top = np.maximum(n_fit - np.isnan(ds).sum(axis=1) - 1, 0)
            pos = top * (1.0 - np.minimum(rate[mar] * 1.8, 0.45))
            low = np.floor(pos).astype(np.intp)
            ar = np.arange(low.size)
            a, b = ds[ar, low], ds[ar, np.minimum(low + 1, top)]
            hole[mar] = (driver >= (a + (b - a) * (pos - low))[:, None]) & (draw[mar] < 0.65)
        v = x[mcols]
        v[hole] = np.nan
        x[mcols] = v
        done["miss_cols"] = k
    return done


def scale_stacked(x: np.ndarray, n_fit: int, mode: str = "nc2_compat") -> tuple[np.ndarray | None, np.ndarray]:
    """Scale the stacked view (p, rows) in place from FIT statistics.

    Returns (missing mask (p, rows) or None when nothing is missing, is_constant (p,)).
    """
    clean = bool(np.isfinite(x.sum()))  # float32-sized values cannot overflow a float64 sum
    has_nan = False
    if not clean:
        finite = np.isfinite(x)
        np.copyto(x, np.nan, where=~finite)
        has_nan = not bool(finite[:, :n_fit].all())
    fit = x[:, :n_fit]
    st = fit_stats(fit, has_nan)
    spread = _spread(fit, st.count if has_nan else None)
    width = (st.q75 - st.q25) / 1.349
    few = st.count < 2
    constant = few | (st.vmin == st.vmax)
    if mode == "nc2_compat":
        scale, zero = np.maximum(np.maximum(width, spread), 1e-6), few
    elif mode == "robust_v2":
        scale, zero = np.where(width > 0, width, spread), constant
    else:
        raise ValueError("scaler_mode")
    med, flat = st.med, bool(zero.any())
    if flat:
        med, scale = np.where(zero, 0.0, med), np.where(zero, 1.0, scale)
    med = med[:, None]
    if not clean:
        np.copyto(x, np.broadcast_to(med, x.shape), where=~finite)
    x -= med
    x /= scale[:, None]
    cells = x.reshape(-1)
    tail = np.flatnonzero(np.abs(cells) > 3.0)
    if tail.size:  # smooth asinh compression beyond |z| > 3, then the hard clip at 6
        v = cells[tail]
        cells[tail] = np.copysign(np.minimum(3.0 + np.arcsinh(np.abs(v) - 3.0), 6.0), v)
    if flat:
        x[zero] = 0.0
    return (None if clean else ~finite), constant


def scale_fit_only(x_fit: np.ndarray, x_qry: np.ndarray, mode: str = "nc2_compat",
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-array form of ``scale_stacked``: (X_fit, X_qry, miss_fit, miss_qry, is_constant), float32.

    Arrays are (rows, columns) as everywhere outside this module.
    """
    n_fit = x_fit.shape[0]
    x = np.concatenate([x_fit, x_qry], axis=0).T.astype(np.float64, order="C")
    miss, constant = scale_stacked(x, n_fit, mode)
    out = np.ascontiguousarray(x.T, dtype=np.float32)
    m = np.zeros(out.shape, dtype=np.float32) if miss is None else np.ascontiguousarray(miss.T, dtype=np.float32)
    return out[:n_fit], out[n_fit:], m[:n_fit], m[n_fit:], constant
