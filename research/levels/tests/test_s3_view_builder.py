"""Vectorised S3 view builder (plan M3, stage C2): Rule A-1 on the view, priors, leak safety, replay."""
from __future__ import annotations

from collections import Counter
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from research.levels.s3_counter_sampler import SamplerConfig, sampler_for
from research.levels.s3_dynamic_augment import _fit_only_nc2_scale
from research.levels.s3_view_bench import synthetic_corpus
from research.levels.s3_view_builder import RetryEpisode, ViewConfig, build_episode, build_slot
from research.levels.s3_view_corpus import ArenaCorpus, MemoryCorpus, write_flat_arena
from research.levels.tests.s3_view_fixtures import (
    DISTINCT_COLUMN, NARROW, TASKS, arena_digest, digest, make_corpus, make_sampler, make_table,
)

LANE = Path(__file__).resolve().parents[3]
REAL_ARENA = Path("/home/ubuntu/dope-data/flat/s3-v250")
SEED = 20261122
PLAIN = ViewConfig(jitter_prob=0.0, warp_prob=0.0, miss_prob=0.0)  # no raw augmentation: values are checkable
PLAIN_DEFAULT = ViewConfig()


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


def _build_steps(corpus, sampler, steps, cfg=PLAIN_DEFAULT):
    out, reasons = [], Counter()
    for step in range(steps):
        for spec in sampler.plan_step(SEED, step):
            episode, used, why = build_slot(corpus, sampler, SEED, spec, cfg)
            reasons.update(why)
            out.append((used, episode))
    return out, reasons


@pytest.fixture(scope="module")
def run(corpus):
    """38 steps x 24 slots = 912 episodes with 6-column views, so duplicate rows appear after the column drop."""
    return _build_steps(corpus, make_sampler(corpus, **NARROW), 38)


def _source(corpus, episode):
    """(feature rows of the view, target per row) of the source table, exactly as stored."""
    table = corpus.table(episode["table_id"])
    target = table["y"] if episode["target_col"] < 0 else table["X"][:, episode["target_col"]]
    return table["X"][:, episode["cols"]], target


def _row_keys(view, target, rows, with_target):
    block = view[rows] if not with_target else np.column_stack([view[rows], target[rows]])
    block = np.where(np.isnan(block), np.float32(-12345.0), block) + np.float32(0.0)
    return {row.tobytes() for row in np.ascontiguousarray(block, dtype=np.float32)}


def test_no_identical_row_sits_on_both_sides_of_the_split(corpus, run):
    episodes, reasons = run
    assert len(episodes) == 38 * 24
    with_dups = after_drop = checked = 0
    for spec, ep in episodes:
        assert ep["X_fit"].shape == (spec.n_fit, spec.p_total) and ep["X_qry"].shape == (spec.n_qry, spec.p_total)
        assert ep["miss_fit"].shape == ep["X_fit"].shape and ep["distractor_mask"].shape == (spec.p_total,)
        for key in ("X_fit", "X_qry", "y_fit", "y_qry", "miss_fit", "miss_qry"):
            assert ep[key].dtype == np.float32 and np.isfinite(ep[key]).all()
        assert np.abs(ep["X_fit"]).max() <= 6.0 and np.abs(ep["X_qry"]).max() <= 6.0
        assert not np.intersect1d(ep["fit_rows"], ep["qry_rows"]).size
        if ep["kind"] == "scm_prior":
            continue
        view, target = _source(corpus, ep)
        for with_target in (False, True):  # identical feature rows, hence also identical (row, target) pairs
            both = _row_keys(view, target, ep["fit_rows"], with_target) & _row_keys(view, target, ep["qry_rows"], with_target)
            assert not both, (spec.table_id, spec.step, spec.slot)
        checked += 1
        with_dups += ep["aug_stats"]["dup_rows"] > 0
        hidden = DISTINCT_COLUMN.get(ep["table_id"])
        if hidden is not None and hidden not in ep["cols"] and hidden != ep["target_col"]:
            after_drop += 1
            assert ep["aug_stats"]["dup_rows"] > 0  # the view lost its only distinct column
    assert checked > 700 and with_dups >= 200 and after_drop >= 25  # the guarantee was exercised, not vacuous
    assert set(reasons) <= {"split_groups", "one_class_fit", "query_rows", "constant_target", "no_swap_column"}


def test_whole_row_key_separates_identical_rows_but_not_feature_twins(corpus):
    """``dup_key="row"``: no identical (row, target) pair crosses the split; feature twins may, so it is opt-in."""
    sampler = make_sampler(corpus, **NARROW)
    episodes, _ = _build_steps(corpus, sampler, 13, ViewConfig(dup_key="row"))
    grouped = twins = 0
    for _, ep in episodes:
        if ep["kind"] == "scm_prior":
            continue
        view, target = _source(corpus, ep)
        with_target = ep["task"] != "multilabel"  # a multilabel view has no stored target
        assert not _row_keys(view, target, ep["fit_rows"], with_target) & _row_keys(view, target, ep["qry_rows"], with_target)
        grouped += ep["aug_stats"]["dup_rows"] > 0
        twins += bool(_row_keys(view, target, ep["fit_rows"], False) & _row_keys(view, target, ep["qry_rows"], False))
    assert grouped >= 60 and twins >= 1
    with pytest.raises(ValueError, match="dup_key"):
        build_episode(corpus, sampler.plan_step(SEED, 0)[0], ViewConfig(dup_key="stored"))


def test_fit_half_keeps_the_class_priors_of_its_pool(corpus, run):
    seen = minority = 0
    for spec, ep in run[0]:
        stats = ep["aug_stats"]
        if "fit_counts" not in stats or ep["kind"] == "scm_prior":
            assert ep["task"] in ("regression", "multilabel") or ep["kind"] == "scm_prior"
            continue
        _, target = _source(corpus, ep)
        levels = np.unique(target[np.isfinite(target)])
        labels = np.searchsorted(levels, target) if ep["kind"] == "target_swap" else np.rint(target).astype(np.int64)
        pool, fit = np.array(stats["pool_counts"]), np.array(stats["fit_counts"])
        assert np.array_equal(np.bincount(labels[ep["fit_rows"]], minlength=fit.size), fit)  # counted from the source
        assert pool.sum() == stats["fit_pool"] and fit.sum() == spec.n_fit and (fit <= pool).all()
        assert (np.abs(fit / spec.n_fit - pool / pool.sum()) < 1.0 / spec.n_fit).all()  # every class, every episode
        assert (fit > 0).sum() >= 2 and ep["n_classes"] == ((fit > 0).sum() if ep["task"] == "multiclass" else 2)
        assert (fit[labels[ep["qry_rows"]]] > 0).all()  # QUERY rows only of classes present in FIT
        assert set(np.unique(ep["y_fit"]).tolist()) == set(range(ep["n_classes"]))  # contiguous class ids
        assert set(np.unique(ep["y_qry"]).tolist()) <= set(range(ep["n_classes"]))
        seen += 1
        if ep["task"] == "binary" and ep["kind"] == "real" and pool.min() / pool.sum() < 0.15:
            minority += 1
            assert abs(fit.min() / fit.sum() - pool.min() / pool.sum()) < 1.0 / spec.n_fit  # 8 % stays 8 %
    assert seen >= 200 and minority >= 20


def _real_specs(corpus, sampler, kinds, steps, tables=None):
    for step in range(steps):
        for spec in sampler.plan_step(SEED, step):
            if spec.kind in kinds and (tables is None or spec.table_id in tables):
                yield spec


def test_real_columns_equal_the_legacy_scaler_and_only_noise_is_flagged(corpus):
    sampler, checked = make_sampler(corpus, batch_size=24), 0
    clean = {i for i in range(len(TASKS)) if i % 4 == 2}  # continuous tables: no duplicate rows, no NaN
    fresh = np.random.default_rng(99)
    other = MemoryCorpus([dict(t, X=fresh.standard_normal(t["X"].shape)) if i in clean else t
                          for i, t in enumerate(corpus.tables)])  # same targets and shapes, different feature values
    for spec in _real_specs(corpus, sampler, ("real",), 12, clean):
        ep, twin = build_episode(corpus, spec, PLAIN), build_episode(other, spec, PLAIN)
        if isinstance(ep, RetryEpisode) or isinstance(twin, RetryEpisode):
            continue
        src = ep["col_source"]
        assert np.array_equal(ep["distractor_mask"], src == -1)
        assert (src == -1).sum() == spec.n_noise and (src == -2).sum() == spec.n_derived
        assert np.array_equal(np.sort(src[src >= 0]), ep["cols"])
        table = corpus.table(spec.table_id)
        ref_f, ref_q, _, _ = _fit_only_nc2_scale(table["X"][ep["fit_rows"]][:, ep["cols"]],
                                                 table["X"][ep["qry_rows"]][:, ep["cols"]])
        at = np.searchsorted(ep["cols"], src[src >= 0])
        assert np.abs(ep["X_fit"][:, src >= 0] - ref_f[:, at]).max() <= 1e-6
        assert np.abs(ep["X_qry"][:, src >= 0] - ref_q[:, at]).max() <= 1e-6
        assert np.array_equal(ep["fit_rows"], twin["fit_rows"]) and np.array_equal(src, twin["col_source"])
        same = (ep["X_fit"] == twin["X_fit"]).all(axis=0) & (ep["X_qry"] == twin["X_qry"]).all(axis=0)
        assert np.array_equal(same, ep["distractor_mask"])  # only noise ignores the table's values
        checked += 1
    assert checked >= 40


def test_target_swap_removes_the_target_from_the_features(corpus, run):
    swaps = [(spec, ep) for spec, ep in run[0] if ep["kind"] == "target_swap"]
    tasks = Counter(ep["task"] for _, ep in swaps)
    assert len(swaps) >= 60 and len(tasks) >= 2
    for spec, ep in swaps:
        t = ep["target_col"]
        assert t >= 0 and t not in ep["cols"] and t not in ep["col_source"] and ep["dataset_hash"].endswith(f"_sw{t}")
        assert ep["cols"].size == spec.p_view and (ep["col_source"] >= 0).sum() == spec.p_view
        width = 1 if ep["task"] in ("regression", "binary") else 8
        assert width <= spec.c_out  # the actual target never needs a wider head than the plan
        raw = corpus.table(spec.table_id)["X"][:, t]
        rows = np.concatenate([ep["fit_rows"], ep["qry_rows"]])
        got = np.concatenate([ep["y_fit"], ep["y_qry"]])
        if ep["task"] == "regression":
            fit = raw[ep["fit_rows"]].astype(np.float64)
            want = np.clip((raw[rows] - fit.mean()) / max(fit.std(), 1e-5), -6.0, 6.0)
            assert np.allclose(got, want, atol=1e-5)
        else:  # the class id is a one-to-one relabelling of the column's values
            pairs = {(float(a), float(b)) for a, b in zip(raw[rows], got)}
            assert len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs}) == ep["n_classes"]


def _poisoned(corpus, table_id, rows, rng, shuffle):
    """Copy of the corpus whose ``rows`` of one table are overwritten: features and target."""
    tables = [dict(t) for t in corpus.tables]
    x, y = tables[table_id]["X"].copy(), tables[table_id]["y"].copy()
    if shuffle:  # a target-swap task is defined by the table's value levels: keep them, move them
        x[rows] = x[rng.permutation(rows)]
    else:
        x[rows] = rng.standard_normal((rows.size, x.shape[1])).astype(np.float32) * 50.0
        y[rows] = y[rng.permutation(rows)] if tables[table_id]["task"] != "regression" else rng.standard_normal(rows.size)
    tables[table_id].update(X=x, y=y)
    return MemoryCorpus(tables)


def test_query_rows_cannot_reach_the_fit_half(corpus):
    """Overwrite the QUERY rows and labels in the source table: the FIT half must not move a bit."""
    sampler, rng = make_sampler(corpus, batch_size=24), np.random.default_rng(5)
    clean = {i for i in range(len(TASKS)) if i % 4 == 2}
    fit_side = ("X_fit", "y_fit", "miss_fit", "distractor_mask", "constant_mask", "col_source", "cols", "fit_rows",
                "task", "n_classes", "target_col")
    fit_stats = ("lattice_cols", "jitter", "jitter_cols", "warp_cols", "miss_cols", "const_cols", "fit_pool",
                 "pool_counts", "fit_counts", "flipped")
    seen, jittered, holed, moved = Counter(), 0, 0, 0
    for spec in _real_specs(corpus, sampler, ("real", "target_swap"), 14, clean):
        ep = build_episode(corpus, spec, ViewConfig())  # default config: jitter, warps and missingness active
        if isinstance(ep, RetryEpisode):
            continue
        bad = _poisoned(corpus, spec.table_id, ep["qry_rows"], rng, shuffle=spec.kind == "target_swap")
        again = build_episode(bad, spec, ViewConfig())
        assert not isinstance(again, RetryEpisode)
        for key in fit_side:
            assert np.array_equal(ep[key], again[key]), (key, spec.kind, spec.task)
        for key in fit_stats:
            assert ep["aug_stats"].get(key) == again["aug_stats"].get(key), key
        moved += not np.array_equal(ep["X_qry"], again["X_qry"])
        seen[(spec.kind, ep["task"])] += 1
        jittered += ep["aug_stats"]["jitter"]
        holed += ep["aug_stats"]["miss_cols"] > 0
    assert {k for k, _ in seen} == {"real", "target_swap"}
    assert {t for _, t in seen} >= {"regression", "binary", "multiclass", "multilabel"}
    assert sum(seen.values()) >= 60 and jittered >= 10 and holed >= 10 and moved >= 0.9 * sum(seen.values())


def test_same_spec_gives_the_same_bytes_in_a_fresh_interpreter(corpus):
    sampler = make_sampler(corpus, batch_size=16)
    specs = sampler.plan_step(SEED, 3)
    first = [digest(build_episode(corpus, spec)) for spec in specs]
    assert first == [digest(build_episode(make_corpus(), spec)) for spec in specs]  # no hidden state in the builder
    assert len(set(first)) == len(first)
    moved = [digest(build_episode(corpus, dataclasses.replace(spec, view_seed=spec.view_seed + 1))) for spec in specs]
    assert all(a != b for a, b in zip(first, moved))
    with tempfile.TemporaryDirectory(prefix="dope-c2-") as tmp:
        write_flat_arena(corpus, tmp)
        mapped = ArenaCorpus(tmp)
        for t in (0, 17, 71):
            assert all(np.array_equal(mapped.table(t)[k], corpus.table(t)[k], equal_nan=True) for k in ("X", "y", "dup"))
            assert not mapped.table(t)["X"].flags.writeable and mapped.table(t)["task"] == corpus.table(t)["task"]
        here = arena_digest(tmp)
        code = f"from research.levels.tests.s3_view_fixtures import arena_digest; print(arena_digest({tmp!r}))"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="random")
        out = subprocess.run([sys.executable, "-B", "-c", code], cwd=LANE, env=env, capture_output=True, text=True,
                             timeout=80)
        assert out.returncode == 0, out.stderr[-400:]
        assert out.stdout.strip() == here
        sampler_for.cache_clear()


def test_one_class_or_constant_fit_half_is_a_retry_never_invented_labels():
    tables = [make_table(i) for i in range(len(TASKS))]
    flat_binary, flat_target = TASKS.index("binary") + 2, 2  # two style-2 tables
    tables[flat_binary]["y"] = np.zeros_like(tables[flat_binary]["y"])
    tables[flat_target]["y"] = np.full_like(tables[flat_target]["y"], 3.25)
    corpus = MemoryCorpus(tables)
    sampler = make_sampler(corpus, batch_size=24)
    spec = sampler._spec(SEED, 0, 0, 0, "binary", flat_binary, None)
    marker = build_episode(corpus, spec)
    assert isinstance(marker, RetryEpisode) and marker.reason == "one_class_fit" and marker.spec is spec
    episode, used, why = build_slot(corpus, sampler, SEED, spec)
    assert why == ["one_class_fit"] * sampler.cfg.max_attempts and used.kind == "scm_prior" and used.task == "binary"
    assert used.attempt == sampler.cfg.max_attempts and episode["source_kind"] == "scm_prior"
    assert set(np.unique(episode["y_fit"]).tolist()) == {0.0, 1.0}  # a real two-class prior table, not a patch
    spec = sampler._spec(SEED, 0, 1, 0, "regression", flat_target, None)
    marker = build_episode(corpus, dataclasses.replace(spec, task="regression", n_classes=0))
    assert isinstance(marker, RetryEpisode) and marker.reason == "constant_target"


def test_arena_views_are_built_and_free_of_cross_half_duplicates():
    """On the real flat arena when this host has it, otherwise on a synthetic arena of the same layout."""
    with tempfile.TemporaryDirectory(prefix="dope-c2-") as tmp:
        real = (REAL_ARENA / "index.jsonl").is_file()
        arena = str(REAL_ARENA) if real else str(write_flat_arena(synthetic_corpus(160, seed=4), tmp))
        corpus = ArenaCorpus(arena)
        sampler = sampler_for(SamplerConfig(batch_size=64, arena=arena))
        retries, built, grouped = Counter(), 0, 0
        for step in range(3):
            for spec in sampler.plan_step(0, step):
                ep, used, why = build_slot(corpus, sampler, 0, spec)
                retries.update(why)
                built += 1
                grouped += ep["aug_stats"]["dup_rows"] > 0
                assert ep["X_fit"].shape == (used.n_fit, used.p_total) and np.isfinite(ep["X_qry"]).all()
                if ep["kind"] != "scm_prior":
                    view, target = _source(corpus, ep)
                    assert not _row_keys(view, target, ep["fit_rows"], False) & _row_keys(view, target, ep["qry_rows"], False)
        assert built == 3 * 64 and sum(retries.values()) < built // 4 and (grouped >= 10 or not real)
        sampler_for.cache_clear()
