"""Benchmark of the gradient farm on the reference problem (plan M3, stage G1). Not a test.

    CUDA_VISIBLE_DEVICES="" nice -n 19 python3 -B -m research.levels.s3_farm_bench \
        --workers 1,2,4 --out target/claude-farm/s3_farm_bench.json

Part A - throughput: episodes/s and scaling efficiency E(W) = rate(W) / (W * rate(1)) for a
configurable per-episode cost (``--rows``, ``--hidden``, ``--depth``), with the phase breakdown.
Part B - reduction: for a gradient dimension D (padding parameters), the leader-side and the
worker-side reduce time per step, the per-episode cost of isolating and accumulating a gradient
in a worker, and the slab size. Part C - the reduce kernel alone, by block size.

Every number is wall time on whatever CPUs the process inherited; the report records the load
of those CPUs while it ran, because a shared host moves all of them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import torch

from .s3_farm_reference import ref_model, reference_spec
from .s3_farm_shm import reduce_rows
from .s3_grad_farm import PHASES, Farm


def cpu_busy(seconds: float = 1.0) -> dict[str, float]:
    """Mean busy share of the inherited CPUs and of the others over a short window."""
    def snap() -> dict[int, tuple[int, int]]:
        out = {}
        with open("/proc/stat", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("cpu") and line[3].isdigit():
                    parts = line.split()
                    vals = [int(v) for v in parts[1:9]]
                    out[int(parts[0][3:])] = (sum(vals), vals[3] + vals[4])
        return out
    a = snap()
    time.sleep(seconds)
    b = snap()
    busy = {c: 1.0 - (b[c][1] - a[c][1]) / max(b[c][0] - a[c][0], 1) for c in b}
    mine = os.sched_getaffinity(0)
    rest = [v for c, v in busy.items() if c not in mine]
    return {"inherited": round(statistics.fmean(busy[c] for c in mine if c in busy), 3),
            "others": round(statistics.fmean(rest), 3) if rest else 0.0, "load1": os.getloadavg()[0]}


def run_farm(spec: Any, steps: int, warmup: int) -> dict[str, Any]:
    """Median step of a farm after warm-up, with phase times in CPU-seconds per step."""
    t0 = time.perf_counter()
    with Farm(spec) as farm:
        start_s = time.perf_counter() - t0
        for step in range(warmup):
            farm.step(step)
        walls, phases, leader, valid = [], {ph: [] for ph in PHASES}, {"wait": [], "reduce": []}, 0
        worker_total, worker_cpu = [], []
        t1 = time.perf_counter()
        for step in range(warmup, warmup + steps):
            res = farm.step(step)
            valid += res.n_valid
            walls.append(res.timings["leader"]["total"])
            for ph in PHASES:
                phases[ph].append(res.timings[ph]["total"])
            for key in leader:
                leader[key].append(res.timings["leader"][key])
            worker_total.append(sum(res.timings["worker_wall"]))
            worker_cpu.append(sum(res.timings["worker_cpu"]))
        wall = time.perf_counter() - t1
        out = {"n_workers": farm.n_workers, "dim": farm.dim, "cpus": farm.cpus, "start_s": round(start_s, 2),
               "episodes_per_s": valid / wall, "step_ms": 1e3 * statistics.median(walls),
               # what the workers were given: CPU time over wall time (1.0 on idle CPUs), and the rate per CPU-second
               "cpu_share": sum(worker_cpu) / sum(worker_total), "episodes_per_cpu_s": valid / sum(worker_cpu),
               "episodes_per_step": valid / steps,
               "phase_ms": {ph: 1e3 * statistics.median(v) for ph, v in phases.items()},
               "leader_ms": {key: 1e3 * statistics.median(v) for key, v in leader.items()},
               # share of the workers' wall time spent computing rather than waiting at the barrier
               "busy_share": statistics.median(worker_total) / (farm.n_workers * statistics.median(leader["wait"])),
               "slab_mib": farm.shm_bytes["slab"] / 2 ** 20}
    return out


def part_a(args: argparse.Namespace, workers: list[int]) -> list[dict[str, Any]]:
    rows = []
    for w in workers:
        spec = reference_spec(w, seed=1, batch=args.batch, rows=args.rows, hidden=args.hidden, depth=args.depth)
        rec = run_farm(spec, args.steps, args.warmup)
        first = rows[0] if rows else rec
        rec["efficiency"] = rec["episodes_per_s"] / (w * first["episodes_per_s"] / first["n_workers"])
        rec["cpu_efficiency"] = rec["episodes_per_cpu_s"] / first["episodes_per_cpu_s"]
        per_ep = rec["episodes_per_step"]
        rec["episode_ms"] = {ph: rec["phase_ms"][ph] / per_ep for ph in PHASES}
        rows.append(rec)
        print(f"A  W={w:<2d} {rec['episodes_per_s']:8.1f} episodes/s  E={rec['efficiency']:.2f}"
              f"  step {rec['step_ms']:7.1f} ms  cpu share {rec['cpu_share']:.2f}"
              f"  {rec['episodes_per_cpu_s']:.1f} episodes/CPU-s"
              f" (E_cpu={rec['cpu_efficiency']:.2f})  busy {rec['busy_share']:.2f}  per episode ms: " +
              " ".join(f"{ph}={rec['episode_ms'][ph]:.2f}" for ph in PHASES), flush=True)
    return rows


def part_b(args: argparse.Namespace, workers: list[int], dims: list[int]) -> list[dict[str, Any]]:
    rows = []
    small = sum(p.numel() for p in ref_model(hidden=32, depth=2).parameters() if p.requires_grad)
    for dim in dims:
        for w in workers:
            for mode in ("leader", "workers"):
                if mode == "workers" and w == 1:
                    continue
                spec = reference_spec(w, seed=1, batch=args.reduce_batch * w, rows=32, hidden=32, depth=2,
                                      pad=dim - small, reduce=mode)
                rec = run_farm(spec, args.reduce_steps, 1)
                per_ep = rec["episodes_per_step"]
                rec.update(mode=mode, accumulate_ms_per_episode=rec["phase_ms"]["reduce"] / per_ep,
                           backward_ms_per_episode=rec["phase_ms"]["backward"] / per_ep,
                           reduce_ns_per_slab_float=1e6 * rec["leader_ms"]["reduce"] / (w * rec["dim"]))
                rows.append(rec)
                print(f"B  D={rec['dim']:>9d} W={w} {mode:7s} leader reduce {rec['leader_ms']['reduce']:7.2f} ms/step"
                      f"  ({rec['reduce_ns_per_slab_float']:.2f} ns per slab float)  worker accumulate "
                      f"{rec['accumulate_ms_per_episode']:6.2f} ms/episode"
                      f" (backward {rec['backward_ms_per_episode']:.2f})  slab {rec['slab_mib']:.0f} MiB"
                      f"  step {rec['step_ms']:.0f} ms  cpu share {rec['cpu_share']:.2f}", flush=True)
    return rows


def part_c(dims: list[int], workers: list[int]) -> list[dict[str, Any]]:
    rows, rng = [], np.random.default_rng(0)
    for dim in dims:
        for w in workers:
            slab = rng.standard_normal((w, dim), dtype=np.float32)
            out = np.empty(dim, dtype=np.float32)
            rec: dict[str, Any] = {"dim": dim, "rows": w, "ms": {}}
            for chunk in (8192, 32768, 131072, dim):
                best = min(_timed(reduce_rows, slab, dim, out, 3.0, chunk) for _ in range(3))
                rec["ms"]["unblocked" if chunk == dim else str(chunk)] = 1e3 * best
            rec["numpy_sum_ms"] = 1e3 * min(_timed(lambda: np.divide(slab.sum(axis=0), np.float32(3.0), out=out))
                                            for _ in range(3))
            rows.append(rec)
            print(f"C  D={dim:>9d} rows={w} " + " ".join(f"{k}={v:.1f}ms" for k, v in rec["ms"].items())
                  + f" numpy_sum={rec['numpy_sum_ms']:.1f}ms", flush=True)
    return rows


def _timed(fn: Any, *args: Any) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", default="1,2,4,8")
    ap.add_argument("--dims", default="1000000,5600000,16000000")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rows", type=int, default=384, help="rows per episode (cost knob)")
    ap.add_argument("--hidden", type=int, default=128, help="hidden width (cost knob)")
    ap.add_argument("--depth", type=int, default=3, help="layers (cost knob)")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reduce-batch", type=int, default=3, help="episodes per worker and step in part B")
    ap.add_argument("--reduce-steps", type=int, default=4)
    ap.add_argument("--parts", default="abc")
    ap.add_argument("--out", default="target/claude-farm/s3_farm_bench.json")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("set CUDA_VISIBLE_DEVICES='' (CPU only)")
    torch.set_num_threads(1)
    workers, dims = [int(w) for w in args.workers.split(",")], [int(d) for d in args.dims.split(",")]
    report: dict[str, Any] = {"host": os.uname().nodename, "torch": torch.__version__, "numpy": np.__version__,
                              "inherited_cpus": sorted(os.sched_getaffinity(0)), "args": vars(args),
                              "cpu_busy_before": cpu_busy()}
    print(f"host {report['host']}  torch {report['torch']}  inherited CPUs {len(report['inherited_cpus'])}"
          f"  busy before: {report['cpu_busy_before']}", flush=True)
    if "a" in args.parts:
        report["throughput"] = part_a(args, workers)
    if "b" in args.parts:
        report["reduce_in_farm"] = part_b(args, workers, dims)
    if "c" in args.parts:
        report["reduce_kernel"] = part_c(dims, workers)
    report["cpu_busy_after"] = cpu_busy()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"busy after: {report['cpu_busy_after']}  ->  {out}", flush=True)


if __name__ == "__main__":
    main()
