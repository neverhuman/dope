"""Synthetic corpus shared by the S3 view-builder tests (plan M3, stage C2). Not a test module.

72 small tables, one family each: 36 regression, 20 binary, 16 multiclass, in four styles.

0. duplicates after a column drop: seven low-cardinality columns drawn from 40 row patterns plus
   one all-distinct column; the target is a function of the pattern, so (view row, target)
   repeats exactly when a view drops the distinct column;
1. fully duplicated: about 60 distinct (row, target) pairs repeated 1 to 12 times;
2. continuous with lattice columns (integers, a 0.25 grid, 20 levels) and skewed class priors;
3. like 2 with NaN cells, a constant column and a column that is 99 % zeros.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from research.levels.s3_counter_sampler import CorpusIndex, CounterSampler, SamplerConfig
from research.levels.s3_view_builder import RetryEpisode, ViewConfig, build_episode
from research.levels.s3_view_corpus import ArenaCorpus, MemoryCorpus

TASKS = ["regression"] * 36 + ["binary"] * 20 + ["multiclass"] * 16
DISTINCT_COLUMN = {}  # table index -> position of the all-distinct column (style 0)
NARROW = dict(batch_size=24, max_view_cols=6, max_total_cols=16)  # views of 6 columns: style-0 drops happen


def _target(rng: np.random.Generator, task: str, signal: np.ndarray) -> tuple[np.ndarray, int]:
    """Target from a real-valued signal: noisy value, an 8 % minority label, or five skewed classes."""
    if task == "regression":
        return (signal + 0.3 * rng.standard_normal(signal.size)).astype(np.float32), 0
    if task == "binary":
        return (signal > np.quantile(signal, 0.916)).astype(np.float32), 2
    cuts = np.quantile(signal + 0.2 * rng.standard_normal(signal.size), [0.5, 0.8, 0.93, 0.98])
    return np.searchsorted(cuts, signal).astype(np.float32), 5


def make_table(i: int, seed: int = 20261122) -> dict[str, Any]:
    rng = np.random.default_rng([seed, i])
    task, style = TASKS[i], i % 4
    if style == 0:
        n = int(rng.choice([300, 600, 1000]))
        patterns = rng.integers(0, 4, (40, 7)) * 0.25
        pick = rng.integers(0, 40, n)
        where = int(rng.integers(0, 8))
        DISTINCT_COLUMN[i] = where
        x = np.insert(patterns[pick], where, rng.permutation(n) / n, axis=1)
        per_pattern, k = _target(rng, task, rng.standard_normal(40))
        y = per_pattern[pick]
    elif style == 1:
        base = rng.standard_normal((60, 6))
        base_y, k = _target(rng, task, base[:, 0] - base[:, 1])
        rows = rng.permutation(np.repeat(np.arange(60), rng.integers(1, 13, 60)))
        x, y = base[rows], base_y[rows]
    else:
        n, p = int(rng.choice([400, 1000, 1500])), int(rng.integers(6, 11))
        x = rng.standard_normal((n, p))
        x[:, 0] = rng.integers(0, 1000, n)  # integers, far more than 64 distinct
        x[:, 1] = rng.integers(0, 400, n) * 0.25  # a 0.25 grid
        x[:, 2] = rng.integers(0, 20, n) / 7.0  # 20 levels off any grid
        y, k = _target(rng, task, x[:, 3] + 0.5 * x[:, 4] * x[:, 5])
        if style == 3:
            x[rng.random(n) < 0.03, 3] = np.nan
            x[:, 4] = 1.5
            x[:, 5] = np.where(rng.random(n) < 0.01, rng.standard_normal(n), 0.0)
    return {"X": x.astype(np.float32), "y": y, "task": task, "n_classes": k, "family_id": f"fam{i}",
            "dataset_hash": f"{i:016x}"}


def make_corpus() -> MemoryCorpus:
    return MemoryCorpus([make_table(i) for i in range(len(TASKS))])


def make_sampler(corpus: MemoryCorpus, **overrides: Any) -> CounterSampler:
    return CounterSampler(CorpusIndex(corpus.index_rows()), SamplerConfig(**overrides))


def digest(result: Any) -> str:
    """Stable digest of an episode (every array byte and scalar) or of a retry marker."""
    h = hashlib.blake2b(digest_size=16)
    if isinstance(result, RetryEpisode):
        h.update(f"retry:{result.reason}".encode())
        return h.hexdigest()
    for key in sorted(result):
        value = result[key]
        h.update(key.encode())
        if isinstance(value, np.ndarray):
            h.update(str((value.dtype, value.shape)).encode() + np.ascontiguousarray(value).tobytes())
        else:
            h.update(repr(sorted(value.items()) if isinstance(value, dict) else value).encode())
    return h.hexdigest()


def arena_digest(arena: str, steps: int = 3, batch_size: int = 16) -> str:
    """Digest of every episode of the first ``steps`` plans of an on-disk arena (subprocess replay)."""
    from research.levels.s3_counter_sampler import sampler_for

    sampler = sampler_for(SamplerConfig(batch_size=batch_size, arena=arena))
    corpus, cfg, h = ArenaCorpus(arena), ViewConfig(), hashlib.blake2b(digest_size=16)
    for step in range(steps):
        for spec in sampler.plan_step(11, step):
            h.update(digest(build_episode(corpus, spec, cfg)).encode())
    return h.hexdigest()
