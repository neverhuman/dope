"""Single-core benchmark and view audit of the S3 view builder (plan M3, stage C2).

    CUDA_VISIBLE_DEVICES="" nice -n 19 python3 -B -m research.levels.s3_view_bench [--arena DIR] [--steps 40] [--baseline-steps 10]

Measures, on one thread and in CPU time (``time.thread_time_ns``, so a busy host does not
count waiting as work):

* ``build_slot`` over planned global steps of the counter sampler: episodes per second overall,
  for the median-size episode (padded cells within 25 % of the plan's median) and per kind;
* the two existing builders on the same tables: ``sample_dynamic_episode`` at its own size law
  and at the planned sizes, and ``build_unique_episode`` at curriculum stage 3;
* what the views contain: retries by reason, rows shared by both halves (must be zero), views
  with duplicate rows, FIT-constant columns, jitter and lattice shares, class-prior drift.

The real arena is used when it exists; otherwise a synthetic arena with its shape
distribution (median 1,000 rows and 10 columns, 15 % of tables wider than 48 columns) is
written to a scratch directory that is removed on exit. numpy only: importing this module never imports torch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from .s3_counter_sampler import CounterSampler, SamplerConfig, sampler_for
from .s3_dynamic_augment import AugmentConfig, sample_dynamic_episode
from .s3_fast_pipeline import build_unique_episode
from .s3_view_builder import ViewConfig, build_slot
from .s3_view_corpus import ArenaCorpus, MemoryCorpus, write_flat_arena

REAL_ARENA = "/home/ubuntu/dope-data/flat/s3-v250"
_NOW = time.thread_time_ns


def synthetic_corpus(n_tables: int = 600, seed: int = 0) -> MemoryCorpus:
    """Tables with the s3-v250 shape distribution; about half of the columns are lattices."""
    rng = np.random.default_rng(seed)
    tables = []
    for t in range(n_tables):
        n = int(np.clip(np.exp(rng.normal(np.log(1000.0), 1.1)), 60, 8192)) if rng.random() > 0.35 else 1000
        p = int(rng.integers(49, 257)) if rng.random() < 0.15 else int(np.clip(np.exp(rng.normal(np.log(9.0), 0.7)), 2, 48))
        x = rng.standard_normal((n, p)).astype(np.float32)
        grid = rng.random(p) < 0.5
        x[:, grid] = np.rint(x[:, grid] * rng.choice([1.0, 4.0, 50.0], int(grid.sum()))).astype(np.float32)
        signal = x[:, 0] - 0.5 * x[:, -1] + 0.5 * rng.standard_normal(n)
        task = str(rng.choice(["regression", "binary", "multiclass"], p=[0.73, 0.20, 0.07]))
        if task == "regression":
            y, k = signal, 0
        elif task == "binary":
            y, k = signal > np.quantile(signal, rng.uniform(0.5, 0.95)), 2
        else:
            k = int(rng.integers(3, 9))
            y = np.searchsorted(np.quantile(signal, np.sort(rng.random(k - 1))), signal)
        tables.append({"X": x, "y": np.asarray(y, dtype=np.float32), "task": task, "n_classes": k, "family_id": f"syn{t}",
                       "dataset_hash": f"{t:016x}"})
    return MemoryCorpus(tables)


def _kernel_microseconds() -> float:
    """A fixed numpy kernel (sort of 448 x 10 doubles), best of 200: how fast this core is right now."""
    block = np.random.default_rng(0).random((10, 448))
    best = 1 << 62
    for _ in range(200):
        start = _NOW()
        np.sort(block, axis=1)
        best = min(best, _NOW() - start)
    return best / 1e3


def _rate(micro: np.ndarray) -> dict[str, float]:
    return {"episodes": int(micro.size), "mean_us": round(float(micro.mean()), 1), "median_us": round(float(np.median(micro)), 1),
            "p95_us": round(float(np.quantile(micro, 0.95)), 1), "episodes_per_s": round(1e6 / float(micro.mean()), 1)}


class _TableStore:
    def __init__(self, corpus: Any) -> None:
        self.corpus = corpus

    def get(self, idx: int) -> dict[str, Any]:
        t = self.corpus.table(idx)
        return {"X": t["X"], "y": t["y"], "dup_groups": t["dup"], "task": t["task"], "family_stem": t["family_id"],
                "n_classes": t["n_classes"] or 1, "dataset_hash": t["dataset_hash"]}


def _timed(repeats: int, fn: Any, *args: Any, **kwargs: Any) -> tuple[float, Any]:
    """(best CPU microseconds of ``repeats`` calls, the last result)."""
    best, out = 1 << 62, None
    for _ in range(repeats):
        start = _NOW()
        out = fn(*args, **kwargs)
        best = min(best, _NOW() - start)
    return best / 1e3, out


def bench_new(corpus: Any, sampler: CounterSampler, seed: int, steps: int, cfg: ViewConfig, repeats: int) -> dict[str, Any]:
    """Time every planned slot (including its retries) and audit the built views."""
    micro, cells, kinds, reasons, audit = [], [], [], Counter(), Counter()
    drift, n_fit = [], []
    for step in range(steps):
        for spec in sampler.plan_step(seed, step):
            took, (ep, used, why) = _timed(repeats, build_slot, corpus, sampler, seed, spec, cfg)
            micro.append(took)
            cells.append((used.n_fit + used.n_qry) * used.p_total)
            kinds.append(used.kind)
            n_fit.append(used.n_fit)
            reasons.update(why)
            st = ep["aug_stats"]
            audit["episodes"] += 1
            audit["dup_views"] += st["dup_rows"] > 0
            audit["const_col_views"] += st["const_cols"] > 0
            audit["jitter_views"] += st["jitter"]
            audit["miss_views"] += st["miss_cols"] > 0
            audit["qry_rows_dropped"] += st["qry_dropped"]
            if st["lattice_cols"] >= 0:
                audit["probed_cols"] += used.p_view
                audit["lattice_cols"] += st["lattice_cols"]
            if "fit_counts" in st:
                pool, fit = np.array(st["pool_counts"]), np.array(st["fit_counts"])
                drift.append(float(np.abs(fit / fit.sum() - pool / pool.sum()).max() * fit.sum()))
            if ep["kind"] != "scm_prior":
                audit["cross_half_identical_feature_rows"] += _twins(corpus, ep)
                audit["checked_views"] += 1
    micro, cells, kinds = np.array(micro), np.array(cells), np.array(kinds)
    centre = float(np.median(cells))
    typical = (cells > 0.8 * centre) & (cells < 1.25 * centre)
    slope = np.linalg.lstsq(np.vstack([np.ones(cells.size), cells]).T, micro, rcond=None)[0]
    n_fit = np.array(n_fit)
    return {
        "all": _rate(micro), "median_size": dict(_rate(micro[typical]), cells=int(centre)),
        "by_kind": {k: dict(_rate(micro[kinds == k]), share_of_time=round(float(micro[kinds == k].sum() / micro.sum()), 3))
                    for k in ("real", "target_swap", "scm_prior")},
        "fit_microseconds": {"fixed": round(float(slope[0]), 1), "per_1000_cells": round(float(slope[1] * 1e3), 1)},
        "retries": dict(reasons), "retry_rate": round(sum(reasons.values()) / micro.size, 4),
        "audit": dict(audit),
        "max_class_prior_drift_rows": round(max(drift), 3) if drift else 0.0,  # |fit share - pool share| * n_fit, must be < 1
        "n_fit": {"median": float(np.median(n_fit)), "share_ge_256": round(float((n_fit >= 256).mean()), 3)},
    }


def bench_existing(corpus: Any, sampler: CounterSampler, seed: int, steps: int, repeats: int) -> dict[str, Any]:
    """The existing builders on the tables of the same plan (real slots only; they know no other kind)."""
    store, live, matched, nxt, matched_cells = _TableStore(corpus), [], [], [], []
    for step in range(steps):
        for spec in sampler.plan_step(seed, step):
            if spec.kind != "real":
                continue
            table, key = store.get(spec.table_id), dict(global_seed=seed, step=step, rank=0)
            live.append(_timed(repeats, sample_dynamic_episode, table, cfg=AugmentConfig(), **key)[0])
            same = AugmentConfig(min_fit_rows=spec.n_fit, max_fit_rows=spec.n_fit, min_query_rows=spec.n_qry,
                                 max_query_rows=spec.n_qry)
            took, ep = _timed(repeats, sample_dynamic_episode, table, cfg=same, **key)
            matched.append(took)
            matched_cells.append((ep["X_fit"].shape[0] + ep["X_qry"].shape[0]) * ep["X_fit"].shape[1])
            nxt.append(_timed(repeats, build_unique_episode, store, spec.table_id, global_seed=seed, step=20000, rank=0,
                              slot=spec.slot, scm_prob=0.0, target_swap_prob=0.0)[0])
    return {"sample_dynamic_episode_own_sizes": _rate(np.array(live)),
            "sample_dynamic_episode_planned_sizes": dict(_rate(np.array(matched)), mean_cells=int(np.mean(matched_cells))),
            "build_unique_episode_stage3": _rate(np.array(nxt))}


def _twins(corpus: Any, ep: dict[str, Any]) -> int:
    """QUERY rows of a view whose selected feature values also occur as a FIT row (exact float32 equality)."""
    block = corpus.table(ep["table_id"])["X"][:, ep["cols"]]
    block = np.ascontiguousarray(np.where(np.isnan(block), np.float32(-12345.0), block) + np.float32(0.0), dtype=np.float32)
    fit_keys = {block[r].tobytes() for r in ep["fit_rows"].tolist()}
    return sum(block[r].tobytes() in fit_keys for r in ep["qry_rows"].tolist())


def feature_twin_rate(corpus: Any, sampler: CounterSampler, seed: int, steps: int) -> dict[str, Any]:
    """Views whose QUERY half repeats a FIT feature row, and retries, under each Rule A-1 key."""
    out: dict[str, Any] = {}
    for mode in ("features", "row"):
        cfg, hit, total, share, retries = ViewConfig(dup_key=mode), 0, 0, [], 0
        for step in range(steps):
            for spec in sampler.plan_step(seed, step):
                ep, _, why = build_slot(corpus, sampler, seed, spec, cfg)
                retries += len(why)
                if ep["kind"] != "scm_prior":
                    twins = _twins(corpus, ep)
                    total += 1
                    hit += twins > 0
                    share += [twins / ep["qry_rows"].size] if twins else []
        out[mode] = {"views": total, "views_with_twins": hit, "share": round(hit / max(total, 1), 4),
                     "mean_share_of_query_rows_in_those": round(float(np.mean(share)), 3) if share else 0.0,
                     "retries": retries}
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arena", default=REAL_ARENA if Path(REAL_ARENA, "index.jsonl").is_file() else "")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3, help="best-of-N CPU time per episode")
    ap.add_argument("--baseline-steps", type=int, default=10, help="steps timed with the existing builders")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="dope-view-bench-") as tmp:
        arena = args.arena or str(write_flat_arena(synthetic_corpus(), tmp))
        corpus = ArenaCorpus(arena)
        sampler = sampler_for(SamplerConfig(batch_size=args.batch_size, arena=arena))
        before = _kernel_microseconds()
        report = {
            "arena": args.arena or "synthetic (s3-v250 shape distribution)", "tables": len(corpus),
            "steps": args.steps, "batch_size": args.batch_size, "seed": args.seed, "numpy": np.__version__,
            "timing": "CPU time of one thread (thread_time_ns), best of %d per episode" % args.repeats,
            "new": bench_new(corpus, sampler, args.seed, args.steps, ViewConfig(), args.repeats),
            "existing_builders": bench_existing(corpus, sampler, args.seed, min(args.baseline_steps, args.steps), args.repeats),
            "feature_twins_by_dup_key": feature_twin_rate(corpus, sampler, args.seed, min(args.baseline_steps, args.steps)),
        }
        report["host_kernel_us"] = {"before": round(before, 1), "after": round(_kernel_microseconds(), 1)}
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return report


if __name__ == "__main__":
    main()
