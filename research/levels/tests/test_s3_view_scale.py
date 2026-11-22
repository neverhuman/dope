"""FIT-only scaler, lattice-aware noise and row planning of the S3 view builder (plan M3, stage C2)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from research.levels.s3_dynamic_augment import _fit_only_nc2_scale
from research.levels.s3_sampler_index import hash_perm
from research.levels.s3_view_builder import ViewConfig
from research.levels.s3_view_rows import class_allocation, group_split, plan_rows, row_hashes, row_keys
from research.levels.s3_view_scale import fit_stats, lattice_columns, raw_augment, scale_fit_only, scale_stacked


def _messy_table(rng: np.random.Generator, trial: int) -> tuple[np.ndarray, int]:
    """Random float32 table with NaN, inf, constant, FIT-constant, heavy-tailed and offset columns."""
    nf, nq, p = int(rng.integers(2, 200)), int(rng.integers(1, 100)), int(rng.integers(1, 12))
    x = rng.standard_normal((nf + nq, p)).astype(np.float32)
    x[:, rng.integers(0, p)] = np.round(x[:, rng.integers(0, p)])  # heavy ties
    if trial % 3 == 0:
        x[rng.random(x.shape) < 0.1] = np.nan
    if trial % 4 == 0:
        x[:, 0] = 2.5  # constant everywhere
    if trial % 5 == 0:
        x[:nf, -1] = 1.0  # constant in FIT only
    if trial % 7 == 0:
        x[:, p // 2] = rng.standard_cauchy(nf + nq) * 1e3  # heavy tails
    if trial % 11 == 0:
        x[rng.random(x.shape) < 0.02] = np.inf
    if trial % 13 == 0:
        x[:nf, 0] = np.nan  # no finite FIT value at all
    if trial % 17 == 0:
        x[:, -1] = x[:, -1] * 1e6 + 3e7  # large offset, float32 resolution matters
    return x, nf


def test_nc2_compat_reproduces_the_legacy_scaler():
    rng = np.random.default_rng(1)
    worst = 0.0
    for trial in range(400):
        x, nf = _messy_table(rng, trial)
        legacy = _fit_only_nc2_scale(x[:nf], x[nf:])
        ours = scale_fit_only(x[:nf], x[nf:], "nc2_compat")
        for a, b in zip(legacy, ours):
            assert a.shape == b.shape and a.dtype == b.dtype == np.float32
            assert np.isfinite(b).all()
            worst = max(worst, float(np.abs(a - b).max(initial=0.0)))
        assert np.array_equal(legacy[2], ours[2]) and np.array_equal(legacy[3], ours[3])  # missing masks: exact
    assert worst <= 1e-6


def test_order_statistics_equal_numpy_on_every_path():
    rng = np.random.default_rng(2)
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 33, 288, 449):
        for rep in range(8):
            f = np.round(rng.standard_normal((6, n)) * 3) / 2 if rep % 2 else rng.standard_normal((6, n))
            want = np.percentile(f, [25, 50, 75], axis=1)
            for st in (fit_stats(f, False), fit_stats(f, False, want_sorted=True)):  # selection and sort
                assert np.array_equal(st.q25, want[0]) and np.array_equal(st.q75, want[2])
                assert np.array_equal(st.med, np.median(f, axis=1))
            g = f.copy()
            g[0, ::3], g[1] = np.nan, np.nan
            st = fit_stats(g, True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # numpy warns on the all-NaN column
                want = np.nanpercentile(g, [25, 50, 75], axis=1)
                med = np.nanmedian(g, axis=1)
            ok = ~np.isnan(want[0])
            assert np.array_equal(st.count, (~np.isnan(g)).sum(axis=1))
            assert np.array_equal(st.q25[ok], want[0][ok]) and np.array_equal(st.q75[ok], want[2][ok])
            assert np.array_equal(st.med[ok], med[ok])


def test_robust_v2_zeroes_fit_constant_columns_instead_of_spiking():
    rng = np.random.default_rng(3)
    nf, nq = 64, 200
    x = rng.standard_normal((nf + nq, 4)).astype(np.float32)
    x[:nf, 0] = 2.0  # constant in FIT, varies in QUERY
    x[:, 1] = np.where(rng.random(nf + nq) < 0.1, x[:, 1], 0.0)  # zero IQR, not constant
    x[:nf, 1][:3] = (5.0, -4.0, 7.0)
    x[:, 3] = rng.standard_cauchy(nf + nq)  # heavy tails: std far above the IQR scale
    xf, xq, mf, mq, const = scale_fit_only(x[:nf], x[nf:], "robust_v2")
    assert const.tolist() == [True, False, False, False]
    assert not xf[:, 0].any() and not xq[:, 0].any()  # zero in both halves
    legacy_f, legacy_q, _, _ = _fit_only_nc2_scale(x[:nf], x[nf:])
    assert (np.abs(legacy_q[:, 0]) == 6.0).mean() > 0.9  # the defect: +-6 spikes from a 1e-6 scale floor
    for out in (xf, xq, mf, mq):
        assert np.isfinite(out).all() and out.dtype == np.float32
    q25, med, q75 = np.percentile(x[:nf, 3].astype(np.float64), [25, 50, 75])
    z = (x[:nf, 3].astype(np.float64) - med) / ((q75 - q25) / 1.349)  # IQR scale, not the inflated std
    body = np.abs(z) <= 3.0
    assert body.sum() > nf // 2 and np.allclose(xf[body, 3], z[body], atol=1e-6)
    assert np.abs(xf[:, 3]).max() > np.abs(legacy_f[:, 3]).max()  # std scaling had squashed the column
    spread = x[:nf, 1].astype(np.float64).std()
    assert np.allclose(xf[:, 1][np.abs(xf[:, 1]) <= 3.0], (x[:nf, 1] / spread)[np.abs(x[:nf, 1] / spread) <= 3.0], atol=1e-6)
    with pytest.raises(ValueError, match="scaler_mode"):
        scale_fit_only(x[:nf], x[nf:], "unknown")


def _typed_columns(rng: np.random.Generator, n: int) -> tuple[np.ndarray, list[bool]]:
    """Stacked (columns, rows) float64 view from float32 storage, with the expected lattice flag per column."""
    cols = [
        (rng.integers(0, 1000, n).astype(np.float32), True),  # integers
        ((rng.integers(0, 400, n) * 0.25).astype(np.float32), True),  # 0.25 grid
        ((rng.integers(0, 1000, n) / np.float32(999.0)).astype(np.float32), True),  # min-max scaled integers
        (rng.choice(rng.standard_normal(60), n).astype(np.float32), True),  # 60 arbitrary levels
        ((rng.integers(0, 2000, n) * 2 + 7).astype(np.float32), True),  # sparse grid: observed gaps are multiples
        (np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 500, n)).astype(np.float32), True),  # grid with NaN
        (rng.standard_normal(n).astype(np.float32), False),
        (rng.random(n).astype(np.float32), False),
        (np.exp(2.0 * rng.standard_normal(n)).astype(np.float32), False),
        ((rng.integers(0, 1000, n) + 0.3 * rng.random(n)).astype(np.float32), False),  # integers plus a smear
    ]
    return np.stack([c for c, _ in cols]).astype(np.float64), [flag for _, flag in cols]


def test_lattice_columns_are_recognised():
    rng = np.random.default_rng(4)
    for n in (128, 256, 448):
        x, want = _typed_columns(rng, n)
        st = fit_stats(x, True, want_sorted=True)
        lattice, distinct = lattice_columns(st.ordered, st.count)
        assert lattice.tolist() == want, (n, lattice.tolist())
        assert distinct[3] <= 60 and distinct[6] == n
    few = np.stack([rng.standard_normal(40), rng.random(40)])  # 64 or fewer rows: everything counts as a lattice
    assert lattice_columns(np.sort(few, axis=1), np.array([40, 40]))[0].all()


def test_jitter_leaves_lattice_columns_bit_identical_and_moves_continuous_ones():
    rng = np.random.default_rng(5)
    n_fit, n = 448, 448 + 320
    x, lattice = _typed_columns(rng, n)
    x[5] = np.where(np.isnan(x[5]), 3.0, x[5])  # keep this test free of NaN so every cell is comparable
    lattice, before = np.array(lattice), x.copy()
    cfg = ViewConfig(jitter_prob=1.0, warp_prob=0.0, miss_prob=0.0)
    done = raw_augment(x, n_fit, np.random.Generator(np.random.PCG64(9)), cfg)
    assert done["jitter"] and done["jitter_cols"] == int((~lattice).sum()) and done["lattice_cols"] == int(lattice.sum())
    assert np.array_equal(x[lattice].view(np.uint64), before[lattice].view(np.uint64))  # bit-identical, both halves
    assert (x[~lattice] != before[~lattice]).all()
    for j in np.flatnonzero(~lattice):
        q25, q75 = np.percentile(before[j, :256], [25, 75])  # the law is set from the probe rows of FIT
        fit_sd, qry_sd = (x[j] - before[j])[:n_fit].std(), (x[j] - before[j])[n_fit:].std()
        assert 0.0015 * (q75 - q25) < fit_sd < 0.03 * (q75 - q25)
        assert 0.75 < fit_sd / qry_sd < 1.33  # one law on both halves, independent draws
    same = before.copy()
    off = ViewConfig(jitter_prob=0.0, warp_prob=0.0, miss_prob=0.0)
    done = raw_augment(same, n_fit, np.random.Generator(np.random.PCG64(9)), off)
    assert np.array_equal(same, before) and not done["jitter"]


def test_warps_and_missingness_touch_only_continuous_columns_and_report_masks():
    rng = np.random.default_rng(6)
    n_fit, n = 320, 640
    cfg = ViewConfig(jitter_prob=0.0, warp_prob=1.0, warp_col_prob=1.0, miss_prob=1.0)
    warped = holes = 0
    for seed in range(40):
        x, lattice = _typed_columns(rng, n)
        x[5] = np.where(np.isnan(x[5]), 3.0, x[5])
        lattice, before = np.array(lattice), x.copy()
        done = raw_augment(x, n_fit, np.random.Generator(np.random.PCG64(seed)), cfg)
        gone = np.isnan(x)
        assert np.array_equal(x[lattice][~gone[lattice]], before[lattice][~gone[lattice]])  # lattices: holes only
        assert int(gone.any(axis=1).sum()) <= done["miss_cols"] <= 5
        assert gone.mean(axis=1).max() < 0.45
        warped += done["warp_cols"]
        holes += int(gone.sum())
        miss, _ = scale_stacked(x, n_fit, "nc2_compat")
        assert np.isfinite(x).all() and np.array_equal(miss, gone)
    assert warped > 40 and holes > 1000


def test_class_allocation_keeps_priors_within_one_row():
    rng = np.random.default_rng(7)
    assert class_allocation(np.array([916, 84]), 100).tolist() == [92, 8]  # 8.4 % stays 8 %, not 19 %
    assert class_allocation(np.array([916, 84]), 32).tolist() == [29, 3]
    assert class_allocation(np.array([999, 1]), 16).tolist() == [15, 1]  # the rare class keeps its row
    for _ in range(3000):
        k = int(rng.integers(1, 33))
        counts = (rng.integers(0, 50, k) * (rng.random(k) < 0.8) * rng.integers(1, 40)).astype(np.int64)
        total = int(counts.sum())
        if not total:
            continue
        n = int(rng.integers(0, total + 1))
        alloc = class_allocation(counts, n)
        quota = n * counts / total
        assert alloc.sum() == n and (alloc >= 0).all() and (alloc <= counts).all()
        assert (np.abs(alloc - quota) < 1.0).all()  # each prior moves by less than 1 / n
        floor = (n * counts) // total
        dropped = (alloc == 0) & (counts > 0)
        if dropped.any():  # a present class is dropped only when every spare seat already saved another one
            assert ((alloc > floor) <= (floor == 0)).all()
    with pytest.raises(ValueError, match="allocation_size"):
        class_allocation(np.array([3, 2]), 6)


def test_row_keys_and_hashes():
    for n in (1, 5, 1000, 20000):
        assert np.array_equal(np.argsort(row_keys(77, n), kind="stable"), hash_perm(77, n))
    rng = np.random.default_rng(8)
    for p in (1, 2, 5, 8):
        x = rng.standard_normal((50, p)).astype(np.float32)
        y = rng.standard_normal(50).astype(np.float32)
        twin = x.copy()
        twin[0, 0] = x[0, 0] + np.float32(2e-7)  # same value at 1e-5
        x[1, 0], twin[1, 0] = 0.0, -0.0
        x[2, p - 1], twin[2, p - 1] = np.nan, -np.float32(np.nan)
        assert np.array_equal(row_hashes(x), row_hashes(twin))
        assert np.array_equal(row_hashes(x, y), row_hashes(twin, y.copy()))
        other = x.copy()
        other[3, p - 1] += np.float32(1e-3)
        differs = row_hashes(x) != row_hashes(other)
        assert differs[3] and differs.sum() == 1
        assert (row_hashes(x, y) != row_hashes(x, y + np.float32(1.0))).all()
        assert np.unique(row_hashes(x, y)).size == 50


def _grouped_hashes(rng: np.random.Generator) -> np.ndarray:
    """300 rows: one group of 90, ten groups of 8, the rest singletons."""
    ids = np.concatenate([np.zeros(90, np.int64), np.repeat(np.arange(1, 11), 8), np.arange(11, 141)])
    return rng.permutation(ids).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)


def test_group_split_keeps_groups_whole_without_favouring_large_groups():
    rng = np.random.default_rng(9)
    h = _grouped_hashes(rng)
    big = np.flatnonzero(h == 0)
    in_fit = 0
    for key in range(400):
        fit, qry, dup = group_split(h, key, 64, 48, 0.6)
        assert dup == 300 - 141 and fit.size >= 64 and qry.size >= 48
        assert np.array_equal(np.sort(np.concatenate([fit, qry])), np.arange(300))
        assert not np.intersect1d(h[fit], h[qry]).size  # no group on both sides
        in_fit += bool(np.isin(big, fit).all())
    assert 0.5 < in_fit / 400 < 0.7  # the 90-row group lands in FIT about as often as the ratio says
    giant = np.concatenate([np.zeros(280, np.uint64), np.arange(1, 21, dtype=np.uint64)])
    assert all(group_split(giant, key, 100, 100, 0.6) is None for key in range(20))  # no cut avoids the group
    assert group_split(h, 1, 200, 101, 0.6) is None  # more rows asked than there are
    singles = np.arange(500, dtype=np.uint64)
    fit, qry, dup = group_split(singles, 5, 100, 100, 0.5)
    assert dup == 0 and np.array_equal(np.concatenate([fit, qry]), hash_perm(5, 500)) and fit.size == 250
    valid = np.ones(500, dtype=bool)
    valid[::2] = False
    fit, qry, _ = group_split(singles, 5, 50, 50, 0.5, valid)
    assert valid[fit].all() and valid[qry].all() and fit.size + qry.size == 250


def test_plan_rows_subsamples_with_pool_priors_and_never_scores_unseen_classes():
    rng = np.random.default_rng(10)
    for trial in range(200):
        n = int(rng.integers(120, 600))
        h = rng.integers(0, n // (1 + trial % 3), n).astype(np.uint64) if trial % 2 else np.arange(n, dtype=np.uint64)
        k = int(rng.integers(2, 12))
        labels = np.minimum(rng.geometric(0.45, n) - 1, k - 1).astype(np.int64)
        n_fit, n_qry = int(rng.integers(8, n // 3)), int(rng.integers(8, n // 4))
        plan = plan_rows(h, trial, n_fit, n_qry, 0.6, labels)
        if isinstance(plan, str):
            assert plan in ("split_groups", "one_class_fit", "query_rows")
            continue
        assert plan.fit.size == n_fit and plan.qry.size == n_qry
        assert np.unique(np.concatenate([plan.fit, plan.qry])).size == n_fit + n_qry
        assert not np.intersect1d(h[plan.fit], h[plan.qry]).size
        fit_counts = np.bincount(labels[plan.fit], minlength=plan.pool_counts.size)
        assert np.array_equal(fit_counts, plan.fit_counts) and plan.pool_counts.sum() == plan.fit_pool
        assert (np.abs(fit_counts / n_fit - plan.pool_counts / plan.fit_pool) < 1.0 / n_fit).all()
        assert (fit_counts[labels[plan.qry]] > 0).all()  # every QUERY class was seen in FIT
        assert (fit_counts > 0).sum() >= 2
    one_class = plan_rows(np.arange(200, dtype=np.uint64), 3, 32, 32, 0.6, np.zeros(200, np.int64))
    assert one_class == "one_class_fit"
    plain = plan_rows(np.arange(200, dtype=np.uint64), 3, 32, 32, 0.6)
    assert plain.pool_counts is None and plain.fit.size == 32 and plain.qry_dropped == 0
