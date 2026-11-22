# Gradient farm — interface contract and measurements (M3, stage G1)

Code: `research/levels/s3_grad_farm.py` (leader: `FarmSpec`, `Farm`, `StepResult`),
`research/levels/s3_farm_worker.py` (worker loop), `research/levels/s3_farm_shm.py`
(segments, model ↔ flat vector, reduction kernel, CPU choice),
`research/levels/s3_farm_reference.py` (reference problem and single-process oracle),
`research/levels/s3_farm_bench.py` (benchmark).
Test: `research/levels/tests/test_s3_grad_farm.py` (17 tests, 49–98 s on the shared host;
at most 4 worker processes alive at a time).
CPU only. No GPU path exists; workers run with `CUDA_VISIBLE_DEVICES=""`.

## 1. What it is

Exact large-batch SGD on CPU cores. One optimizer step is one global plan, a
pure function of `(seed, step)`. `W` worker processes, one logical CPU and one
torch thread each, plan the step themselves, take their share, and write the
SUM of their valid per-episode gradients into their row of a shared slab. The
leader performs one reduction, divides by the global number of valid episodes
and reports `apply` or `skip:<reason>`. The farm never clips and never steps
an optimizer. It holds no training state, so resume needs nothing from it.

```python
from research.levels.s3_grad_farm import Farm, FarmSpec, RETRY, spike_rule

spec = FarmSpec(model="pkg.mod:build", model_kwargs={...},
                episode_fn="pkg.mod:episode", episode_init="pkg.mod:open", episode_kwargs={...},
                loss_fn="pkg.mod:loss", loss_kwargs={...},
                plan_kwargs={"arena": "/path/to/arena", "batch_size": 64},   # default plan: counter sampler
                seed=7, n_workers=16, diag_keys=("tstr", "distill"))
with Farm(spec) as farm:
    model = farm.parameters()                    # leader nn.Module; tensors alias the shared vector
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(start, stop):
        res = farm.step(step, skip_rule=spike_rule(history))
        if res.decision == "apply":              # model.<param>.grad already is res.grad
            optim.step()                         # in place: every worker reads it at step + 1
            history.append(res.grad_norm)
        if step % 200 == 0:
            farm.param_checksum()
```

## 2. `FarmSpec`

Plain data. It reaches the workers as JSON; nothing is pickled, so every kwarg
must be JSON (`spec_not_json` otherwise) and every callable is an import path
`"package.module:attr"`.

| Field | Called in the worker as | Notes |
|---|---|---|
| `model`, `model_kwargs` | `model(**model_kwargs) -> nn.Module` | Same structure in every process. Float32 CPU parameters; no BatchNorm running statistics. Initial values come from the leader's instance |
| `plan_fn`, `plan_kwargs` | `plan_fn(seed=seed, **plan_kwargs) -> (step -> list of specs)` | Default `s3_grad_farm:counter_plan`; `plan_kwargs` are `SamplerConfig` fields. A spec needs `slot` and `cells`; optional `n_episodes` (default 1) when one spec is a micro-batch whose loss is a sum |
| `episode_init`, `episode_kwargs` | `episode_init(episode_kwargs) -> state`, once | Omitted: `state` is the kwargs dict |
| `episode_fn` | `episode_fn(spec, state) -> episode \| RETRY` | |
| `loss_fn`, `loss_kwargs` | `loss_fn(model, episode, generator, **loss_kwargs) -> scalar \| (scalar, {name: float})` | `generator` and the global torch RNG are seeded from `(seed, step, slot)` |
| `seed`, `n_workers` | | |
| `cpus` | | Explicit CPU per worker; must lie inside the inherited mask |
| `diag_keys` | | Diagnostics that get a slab slot. Others are counted in `diagnostics["dropped"]` |
| `train_mode` | `model.train(train_mode)` once | Default `True` |
| `reduce` | | `auto` \| `leader` \| `workers` (section 6) |
| `worker_env` | | Extra environment of the workers (for example a BLAS preload) |
| `step_timeout_s`, `start_timeout_s` | | 600 s and 300 s |

## 3. `Farm`

| Call | Returns | Notes |
|---|---|---|
| `parameters()` | the leader's `nn.Module` | Every float32 tensor is a view into the shared vector. Update in place (`optimizer.step()`, `load_state_dict`). Re-binding a tensor (`p.data = ...`, `.to()`) detaches the leader; `param_checksum` reports it |
| `flat`, `dim` | trainable parameters as one shared float32 vector, its length D | `named_parameters()` order of the trainable parameters |
| `step(s, skip_rule=None)` | `StepResult` | Skip iff `n_valid == 0`, or the reduced vector holds a non-finite entry, or `skip_rule(result)` returns a reason. On `apply` the leader module's `.grad` fields are views of `result.grad`; on a skip they are `None` |
| `param_checksum()` | hex digest | Leader and every worker hash the shared tensors as their own model holds them. Any difference raises |
| `collect(name)` | one picklable object per worker | `state.farm_collect(name)` if the state has it, else `state[name]`. Built in: `farm.info`, `farm.fail_reasons`. Unknown name raises, farm stays open |
| `close()` | | Idempotent. The leader module stays usable |
| `cpus`, `pids`, `prefix`, `shm_bytes`, `workers`, `closed` | | `workers` is the start-up info of each worker (pid, CPU, thread counts, versions) |

`StepResult`: `step`, `grad` (flat float32, SUM / `n_valid`, farm-owned and
valid until the next `step`), `grad_norm`, `n_planned = n_valid + n_fail +
n_retry + n_nonfinite`, `loss_mean`, `sq_norm_sum` (Σ over workers of the
squared norm of that worker's gradient SUM), `ep_sq_norm_sum` (Σ over valid
episodes of the squared norm of the episode gradient), `diagnostics` (sums),
`timings` (phase → `{"total", "workers"}` for plan, build, forward, backward,
reduce; `worker_wall`, `worker_cpu`; `leader` → wait, reduce, total),
`workers` (every scalar-block column per worker), `fail_reasons`, `decision`.

`spike_rule(history, factor=10, floor=20, min_history=16)` is the caller-side
rule of the plan (`‖g‖ > max(10 × median, 20)`); `history` belongs to the
caller and to its checkpoint.

## 4. What a step guarantees

- **Pure.** `grad(s)` is a function of (parameters, `seed`, `s`). The plan and
  the shard are pure, each episode seeds its RNG streams with
  `key64("farm-episode", seed, s, slot)`, sums run in slot order inside a worker
  and in worker order across workers.
- **Worker-count independent** up to float32 summation order. Measured on the
  reference problem: W = 2 against W = 1 1.0e-7, W = 3 against W = 1 9.0e-8,
  farm against the float64 oracle 4.8e-8 to 7.7e-8 (relative L2). With
  `DopeSingleEncV6` and `compute_single_enc_loss` (scratch adapter, not
  committed): W = 2 against the oracle 3.9e-8, W = 1 against W = 2 9.3e-8.
- **Bit-identical** for a fixed worker count across fresh farms and across the
  two reduction modes (tested). Same runtime only: torch versions differ
  between hosts.
- **Counted, never crashed.** `RETRY` → `n_retry`. An exception in
  `episode_fn`, `loss_fn` or the backward pass → `n_fail` with a reason code
  `stage:ExceptionType@file:line` (no message text). A non-finite loss or
  gradient → `n_nonfinite`. None of them touches the sum, and the divisor is
  the global valid count, so a worker whose episodes all fail still yields the
  right mean.
- **Per-episode isolation.** The worker takes `torch.autograd.grad` into fresh
  tensors, computes the squared norm, and only then adds into its slab row.
  Finite squared norm ⇔ every entry finite; a gradient with an entry beyond
  1.8e19 overflows it and is dropped as non-finite too. Cost over accumulating
  into `.grad`: one read pass over D floats per episode (0.06–0.19 ns per
  float with numpy's BLAS). On the real model the whole worker-side
  accumulation was 8.1 ms of 367 ms per episode (5.68M parameters) and 29.6 ms
  of 1,236 ms (15.96M).

## 5. Processes, CPUs, shared memory

- Workers are fresh interpreters (`fork` + `exec` through `subprocess`, not a
  fork of a process that holds torch, and not `multiprocessing`'s spawn, which
  re-imports `__main__`). `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` = 1 and
  `CUDA_VISIBLE_DEVICES=""` are in the environment before torch is imported; a
  worker refuses to start otherwise. `PYTHONHASHSEED=0`.
- A worker pins itself to one logical CPU of the inherited mask
  (`sched_setaffinity` to a subset, never wider). Default order: spread over L3
  cache domains, then over physical cores, SMT siblings last. The leader is not
  pinned and its thread count is the caller's.
- Segments `/dev/shm/dope-farm.<leader pid>.<token>.{params,slab,grad}`.
  `params` = trainable parameters, then frozen parameters and float32 buffers.
  `slab` = W rows of `D + S` floats rounded up to 4 KiB; pages are first touched
  by the owning worker. Size: 4 × W × D bytes, i.e. 61 MiB per worker at 16M
  parameters (0.95 GiB at W = 16, 3.3 GiB at W = 56) and 21 MiB at 5.6M.
- Channel: one socket pair per worker. JSON commands down, small pickles up.
  A dead worker closes its socket, so the leader sees it at once and raises
  `FarmError("worker 1 (pid …) exited with code 3 during step 3")`; a silent
  one hits `step_timeout_s`. Either closes the farm.
- Leaks. `close()`, an exception in the `with` block and SIGTERM (the farm
  installs a handler while open, if SIGTERM was at its default) stop the
  workers and unlink the segments. After SIGKILL of the leader the orphaned
  workers unlink the segments themselves and exit. `sweep_stale()` removes
  segments whose leader pid is gone; a farm calls it on start.
- glibc malloc in workers: `MALLOC_MMAP_THRESHOLD_=32 MiB`,
  `MALLOC_TRIM_THRESHOLD_=16 GiB`, `MALLOC_TOP_PAD_=64 MiB` unless the
  environment already sets them. Without them every large tensor is mapped and
  unmapped again. Measured in one process on the reference problem padded to a
  16M-float gradient: 242 → 83 ms per episode, 28,290 → 0 page faults and
  124 → 0.1 ms system time per episode.

## 6. The reduction

`reduce_rows` adds the rows in worker order, blocked by 32,768 columns so the
accumulator block stays in cache, divides by `n_valid` and returns the squared
norm in the same pass. The scalar block (counts, loss sum, norms, phase times,
diagnostic sums) is S floats at the end of each row.

- `reduce="leader"`: the leader makes the pass. Cost grows with W × D.
- `reduce="workers"`: each worker reduces one column block of all rows into the
  shared `grad` segment (one more command round). Same bits.
- `auto`: workers from 32M slab floats per step (W × D), leader below. At
  D = 16M the worker reduction was faster in three of four runs (W = 2: 21–41
  against 39–70 ms; W = 4: 38–59 against 73–84 ms) and slower in the run with
  the busiest CPUs (83 against 34 ms, 94 against 53 ms), because it then waits
  for workers that are not being scheduled.

Multi-node seam (not built): call `reduce_rows(..., divisor=None)` over the
first `D + S` floats on each node, all-reduce (SUM) that one vector across
nodes, then divide by the valid count that travelled in it.

## 7. Measurements

Host xbabe2, 32 inherited logical CPUs (48–63, 112–127) that other jobs kept
52–98% busy, `nice 19`, torch 2.7.0, numpy 1.26.4, 2026-11-22. All wall-clock
numbers moved by up to 4× between runs; `cpu share` is worker CPU time over
worker wall time.

Throughput, reference problem (hidden 128, depth 3, 384–768 rows, 32 episodes
per step, about 80 ms per episode), `s3_farm_bench --parts a`:

| Run | cpu share W = 1 / 2 / 4 | episodes/s W = 1 | W = 2 | W = 4 | E(2) | E(4) |
|---|---|---|---|---|---|---|
| quiet (console only; workers on CPUs 48–51) | 0.99 / 0.99 / 1.00 | 12.6 | 23.4 | 46.2 | 0.93 | 0.92 |
| `s3_farm_bench.json` | 0.52 / 0.49 / 0.57 | 8.5 | 10.5 | 17.5 | 0.62 | 0.51 |
| `s3_farm_bench_a1.json` | 0.59 / 0.52 / 0.43 | 9.0 | 11.1 | 14.4 | 0.62 | 0.40 |
| `s3_farm_bench_a2.json` | 0.82 / 0.56 / 0.91 | 16.2 | 13.6 | 35.7 | 0.42 | 0.55 |

E(W) = rate(W) / (W × rate(1)). The files are under `target/claude-farm/`. The
quiet run's file was overwritten by a later run; its numbers are from the
console. Efficiency follows the CPU share the workers were given: a worker on a
busy CPU holds the barrier for everyone (share of worker time spent computing
rather than waiting: 0.90–0.98 in the quiet run, 0.58–0.85 otherwise).

Reduction per step, leader wall time, in a farm (`s3_farm_bench.json`):

| D | W = 1 | W = 2 leader / workers | W = 4 leader / workers | slab at W = 4 |
|---|---|---|---|---|
| 1.0M | 2.5 ms | 4.4 / 2.4 ms | 8.1 / 45.5 ms | 15 MiB |
| 5.6M | 23.2 ms | 21.3 / 15.5 ms | 72.1 / 41.5 ms | 85 MiB |
| 16M | 32.8 ms | 38.6 / 20.8 ms | 73.6 / 58.7 ms | 244 MiB |

Reduction kernel alone at D = 16M (private memory, best of 3): blocked 26 / 32
/ 59 ms for 1 / 2 / 4 rows; unblocked 261 / 56 / 147 ms and `numpy.sum` 291 /
55 / 84 ms in that run, 330–625 ms and 340–500 ms in two earlier runs. Blocked
accumulation straight into the output is therefore required at 16M. The leader
pass costs roughly 26 ms plus 11 ms per further row at 16M, i.e. about 0.2 s
per step at W = 16 by extrapolation (not measured).

The existing model through the farm (scratch adapter around `DopeSingleEncV6`,
`build_unique_episode` with stage-3 prior tables and `compute_single_enc_loss`;
2 workers, 16 timed episodes per model):

| Model | D | episodes per CPU-second | forward | backward | build | accumulate |
|---|---|---|---|---|---|---|
| d_model 256, 6 layers | 5,681,707 | 2.74 | 145 ms | 205 ms | 8.9 ms | 8.1 ms |
| d_model 384, 8 layers | 15,963,307 | 0.98 | 436 ms | 757 ms | 13.0 ms | 29.6 ms |

The plan's working estimate was 14 and 6 episodes/s per core. On this host
torch is the distribution package (`python3-torch-cuda 2.7.0+ds`,
`BLAS_INFO=generic`) linked against the reference `libblas.so.3`: single-thread
matrix multiply ran at 2.6 / 2.9 / 4.3 GFLOP/s (n = 128 / 384 / 1024) against
37 / 52 / 104 GFLOP/s for numpy's OpenBLAS in the same process. The estimate
assumed about 20. Whether a fast BLAS closes the gap is not measured.

## 8. Limits

- Static sharding: the step waits for the slowest worker. On dedicated cores
  the shard balances planned cells; on shared cores a descheduled worker stalls
  the step.
- Float32 buffers are shared and must not be written by a forward pass. A model
  with BatchNorm running statistics is refused.
- Non-float32 buffers stay private; the farm refuses to start when they differ
  between processes and does not propagate later changes.
- `skip:nonfinite` cannot be reached through the per-episode filter alone
  (finite entries below 1.8e19 cannot overflow a float32 sum of fewer than 2²⁴
  terms); it guards the reduced vector against anything else.
- The SIGTERM guard needs the farm to be created in the main thread. A caller
  with its own SIGTERM handler must unwind the stack so `with` can close.
