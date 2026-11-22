"""Gradient farm (plan M3, stage G1): worker-count independence, replay, shared parameters, drop rules, leaks.

CPU only; never more than 4 worker processes alive at a time. Every farm of this file runs the
reference problem of ``s3_farm_reference`` and is compared with its single-process oracle.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from research.levels.s3_counter_sampler import shard
from research.levels.s3_farm_reference import ref_model, ref_plan, reference_spec, single_process_step
from research.levels.s3_farm_shm import (
    PREFIX, SHM_DIR, FarmError, cpu_topology, model_layout, pick_cpus, reduce_rows, segments, sweep_stale,
)
from research.levels.s3_grad_farm import Farm, FarmSpec, StepResult, spike_rule

torch.set_num_threads(1)
LANE = Path(__file__).resolve().parents[3]
SEED, BATCH, TOL = 20261122, 24, 1e-6
PLAN = ref_plan(SEED, batch=BATCH)
# Faults by step. Step 11 fails every episode that worker 1 of a two-worker farm owns.
STARVED = [s.slot for s in shard(PLAN(11), 1, 2)]
FAULTS = {
    "10": {"retry": [1, 5], "raise": [2], "raise_loss": [7]},
    "11": {"raise": STARVED},
    "12": {"nan_loss": [3]},
    "13": {"nan_grad": [4]},
    "14": {"nan_loss": list(range(0, BATCH, 2)), "nan_grad": list(range(1, BATCH, 2))},
    "15": {"retry": list(range(BATCH))},
}
STEPS = (0, 1, 3, 10, 11, 12, 13, 14, 15)


def _spec(n_workers: int, **overrides) -> FarmSpec:
    return reference_spec(n_workers, seed=SEED, batch=BATCH, faults=FAULTS, **overrides)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _gone(prefix: str, pids: list[int], patience: float = 0.0) -> bool:
    """No segment of that farm in /dev/shm and none of its workers alive."""
    deadline = time.monotonic() + patience
    while segments(prefix) or any(map(_alive, pids)):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _leader_grad(farm: Farm) -> torch.Tensor | None:
    grads = [p.grad for p in farm.parameters().parameters() if p.requires_grad]
    return None if any(g is None for g in grads) else torch.cat([g.reshape(-1) for g in grads])


@pytest.fixture(scope="module")
def runs():
    """STEPS on fresh farms of W = 1, 2, 3, one farm at a time. W = 3 reduces in the workers."""
    out = {}
    for w in (1, 2, 3):
        with Farm(_spec(w, reduce="workers" if w == 3 else "leader")) as farm:
            rec = {"cpus": farm.cpus, "pids": farm.pids, "dim": farm.dim, "flat": farm.flat.clone(),
                   "info": farm.collect("farm.info"), "built": [farm.collect("built")], "steps": {}, "bound": {}}
            for step in STEPS:
                res = farm.step(step)
                bound = _leader_grad(farm)
                rec["bound"][step] = None if bound is None else torch.equal(bound, res.grad)
                rec["steps"][step] = replace(res, grad=res.grad.clone())
            rec["built"].append(farm.collect("built"))
            rec["fail_reasons"] = farm.collect("farm.fail_reasons")
            rec["unchanged"] = torch.equal(farm.flat, rec["flat"])
            prefix = farm.prefix
        rec["gone"] = _gone(prefix, rec["pids"])
        out[w] = rec
    return out


@pytest.fixture(scope="module")
def live(runs):
    """One two-worker farm for the tests that talk to a running farm (created after ``runs`` is done)."""
    with Farm(_spec(2)) as farm:
        yield farm
        prefix, pids = farm.prefix, farm.pids
    assert _gone(prefix, pids)
    assert segments(f"{PREFIX}.{os.getpid()}.") == []  # nothing of this test process is left behind


def test_worker_counts_give_the_same_gradient(runs):
    assert torch.equal(runs[1]["flat"], runs[2]["flat"]) and torch.equal(runs[1]["flat"], runs[3]["flat"])
    assert all(rec["unchanged"] and rec["gone"] for rec in runs.values())
    for step in (0, 1, 3):
        oracle = single_process_step(_spec(1), step)
        for w, rec in runs.items():
            res = rec["steps"][step]
            assert res.decision == "apply" and res.grad.dtype == torch.float32 and res.grad.shape == (rec["dim"],)
            assert (res.n_planned, res.n_valid, res.n_fail, res.n_retry, res.n_nonfinite) == (BATCH, BATCH, 0, 0, 0)
            assert _rel(res.grad, runs[1]["steps"][step].grad) < TOL and _rel(res.grad, oracle["grad"]) < TOL
            assert res.grad_norm == pytest.approx(float(res.grad.double().norm()), rel=1e-6)
            assert res.loss_mean == pytest.approx(oracle["loss_mean"], rel=1e-5)
            assert res.ep_sq_norm_sum == pytest.approx(oracle["ep_sq_norm_sum"], rel=1e-4)
            assert sum(res.workers["n_planned"]) == BATCH and len(res.workers["n_valid"]) == w
            assert max(res.workers["n_planned"]) - min(res.workers["n_planned"]) <= 1
            assert set(res.diagnostics) == {"mse", "ridge", "dropped"} and res.diagnostics["dropped"] == BATCH
            assert res.diagnostics["mse"] > 0.0 and res.timings["backward"]["total"] > 0.0
            assert len(res.timings["forward"]["workers"]) == w and res.timings["leader"]["wait"] > 0.0
            assert rec["bound"][step] is True  # the leader module's .grad fields are the reduced vector
        one, three = runs[1]["steps"][step], runs[3]["steps"][step]
        assert one.sq_norm_sum == pytest.approx((BATCH * one.grad_norm) ** 2, rel=1e-4)  # W = 1: ||SUM||^2
        assert three.sq_norm_sum >= one.sq_norm_sum / 3.0 * (1.0 - 1e-4)  # sum_w ||S_w||^2 >= ||sum_w S_w||^2 / W


def test_fixed_worker_count_replays_bit_identically(runs):
    first, other_step = runs[2]["steps"][0].grad, runs[2]["steps"][3].grad
    assert not torch.equal(first, other_step) and _rel(first, other_step) > 1e-3  # the step matters
    with Farm(_spec(2, reduce="workers")) as again:
        # new processes, and the reduction done by the workers instead of the leader: the same bits
        assert torch.equal(again.step(3).grad, other_step) and torch.equal(again.step(0).grad, first)
        assert torch.equal(again.step(3).grad, other_step)  # a step does not depend on what ran before it
    # same data seed and parameters as ``runs``, another farm seed: another plan and other noise
    assert _spec(2).episode_kwargs["data_seed"] == SEED + 1
    with Farm(reference_spec(2, seed=SEED + 1, batch=BATCH, episode_kwargs={"data_seed": SEED + 1})) as farm:
        assert torch.equal(farm.flat, runs[2]["flat"]) and _rel(farm.step(0).grad, first) > 1e-3


def _checkpoint_state(spec: FarmSpec, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """A state dict as a checkpoint would hold it: the factory's model with ``flat`` as its trainable part."""
    model = ref_model(**spec.to_json()["model_kwargs"])
    torch.nn.utils.vector_to_parameters(flat.clone(), [p for p in model.parameters() if p.requires_grad])
    return model.state_dict()


def test_leader_updates_reach_every_worker_and_the_checksum_sees_a_stale_one():
    spec = _spec(2)
    with Farm(spec) as farm:
        model = farm.parameters()
        optim = torch.optim.SGD(model.parameters(), lr=0.5)
        before, digest = farm.flat.clone(), farm.param_checksum()
        old = farm.step(1).grad.clone()
        optim.step()  # in place, on the leader only
        assert not torch.equal(farm.flat, before) and farm.param_checksum() != digest
        new = farm.step(1)
        assert _rel(new.grad, single_process_step(spec, 1, farm.flat)["grad"]) < TOL
        assert _rel(new.grad, old) > 1e-3
        model.load_state_dict(_checkpoint_state(spec, before))  # the resume path: in place as well
        assert torch.equal(farm.flat, before) and farm.param_checksum() == digest
        assert torch.equal(farm.step(1).grad, old)
        with torch.no_grad():  # a float32 buffer is shared too: the workers see the shift
            model.shift.add_(0.25)
        assert _rel(farm.step(1).grad, old) > 1e-3 and farm.param_checksum() != digest
        farm._debug_detach(1)  # worker 1 now reads a private copy
        farm.param_checksum()  # still equal: nothing has changed since
        with torch.no_grad():
            farm.flat.add_(1e-3)
        with pytest.raises(FarmError, match=r"param_checksum mismatch: workers \[1\]"):
            farm.param_checksum()
        first = next(model.parameters())
        first.data = first.data.clone()  # the leader itself re-bound away from the shared vector
        with pytest.raises(FarmError, match="leader tensor no longer aliases"):
            farm.param_checksum()


def test_retry_and_exceptions_are_counted_and_excluded(runs):
    oracle = single_process_step(_spec(1), 10)
    assert (oracle["n_valid"], oracle["n_retry"], oracle["n_fail"]) == (BATCH - 4, 2, 2)
    for rec in runs.values():
        res = rec["steps"][10]
        assert (res.n_planned, res.n_valid, res.n_retry, res.n_fail, res.n_nonfinite) == (BATCH, BATCH - 4, 2, 2, 0)
        assert res.decision == "apply" and _rel(res.grad, oracle["grad"]) < TOL
        assert sorted(code.split("@")[0] for code in res.fail_reasons) == ["build:ValueError", "loss:KeyError"]
        assert all(code.split("@")[1].startswith("s3_farm_reference.py:") for code in res.fail_reasons)
        assert sum(res.fail_reasons.values()) == 2
        assert res.diagnostics["dropped"] == BATCH - 4  # one undeclared diagnostic per valid episode
        totals: dict[str, int] = {}
        for reasons in rec["fail_reasons"]:  # cumulative, per worker
            for code, k in reasons.items():
                totals[code.split("@")[0]] = totals.get(code.split("@")[0], 0) + k
        assert totals == {"build:ValueError": 1 + len(STARVED), "loss:KeyError": 1}


def test_a_worker_whose_episodes_all_fail_still_gives_the_global_mean(runs):
    oracle = single_process_step(_spec(1), 11)
    assert oracle["n_valid"] == BATCH - len(STARVED) > 0
    res = runs[2]["steps"][11]
    assert res.workers["n_valid"] == [BATCH - len(STARVED), 0] and res.workers["n_fail"] == [0, len(STARVED)]
    assert res.workers["sq_norm"][1] == 0.0 and res.n_valid == oracle["n_valid"] and res.decision == "apply"
    assert _rel(res.grad, oracle["grad"]) < TOL
    for w in (1, 3):  # the same global mean for the other worker counts
        other = runs[w]["steps"][11]
        assert other.n_valid == oracle["n_valid"] and _rel(other.grad, res.grad) < TOL
    # dividing by the planned count (the legacy denominator) would be off by n_valid / n_planned
    assert _rel(res.grad * (res.n_valid / res.n_planned), oracle["grad"]) > 0.3


def test_nonfinite_episodes_are_dropped_without_poisoning_the_sum(runs, live):
    for step in (12, 13):  # a NaN loss; a finite loss with a NaN gradient
        oracle = single_process_step(_spec(1), step)
        assert (oracle["n_valid"], oracle["n_nonfinite"]) == (BATCH - 1, 1)
        for rec in runs.values():
            res = rec["steps"][step]
            assert (res.n_valid, res.n_nonfinite, res.n_fail, res.decision) == (BATCH - 1, 1, 0, "apply")
            assert bool(torch.isfinite(res.grad).all()) and _rel(res.grad, oracle["grad"]) < TOL
    for step, counts in ((14, (0, BATCH, 0)), (15, (0, 0, BATCH))):  # every episode non-finite; every one a retry
        for rec in runs.values():
            res = rec["steps"][step]
            assert (res.n_valid, res.n_nonfinite, res.n_retry) == counts and res.decision == "skip:no_valid"
            assert not res.grad.any() and rec["bound"][step] is None  # .grad is None: a stray optimizer step is inert
    # a caller-supplied rule: the farm reports and never clips
    res = live.step(0, skip_rule=lambda r: "spike" if r.grad_norm > 0.0 else None)
    assert res.decision == "skip:spike" and _leader_grad(live) is None
    assert torch.equal(res.grad, runs[2]["steps"][0].grad)
    assert live.step(0, skip_rule=spike_rule([res.grad_norm] * 16)).decision == "apply"
    assert live.step(0, skip_rule=spike_rule([res.grad_norm / 100.0] * 16, floor=0.0)).decision == "skip:spike"
    assert live.step(0, skip_rule=spike_rule([res.grad_norm / 100.0] * 15, floor=0.0)).decision == "apply"


def test_reduce_rows_is_one_exact_pass_and_flags_nonfinite_sums():
    rng = np.random.default_rng(5)
    rows = rng.standard_normal((5, 1000), dtype=np.float32)
    want = rows[0].copy()
    for k in range(1, 5):
        want += rows[k]
    raw = np.empty(1000, dtype=np.float32)
    reduce_rows(rows, 1000, raw, None)
    assert np.array_equal(raw, want)
    want /= np.float32(7.0)
    for chunk in (1, 64, 333, 4096):  # row order for any block size: the same bits
        out = np.full(1000, np.nan, dtype=np.float32)
        sq = reduce_rows(rows, 1000, out, 7.0, chunk)
        assert np.array_equal(out, want)
        assert sq == pytest.approx(float((want.astype(np.float64) ** 2).sum()), rel=1e-6)  # float32 squares
    big = np.full((2, 8), 3.0e38, dtype=np.float32)  # finite rows whose sum overflows
    assert not np.isfinite(reduce_rows(big, 8, np.empty(8, dtype=np.float32), 1.0))
    rows[2, 17] = np.nan
    assert np.isnan(reduce_rows(rows, 1000, np.empty(1000, dtype=np.float32), 7.0))


def test_a_dead_worker_is_reported_and_nothing_is_left():
    slot = shard(PLAN(3), 1, 2)[0].slot
    spec = reference_spec(2, seed=SEED, batch=BATCH, faults={"3": {"exit": [slot]}, "2": {"retry": [0]}},
                          plan_kwargs={"batch": BATCH, "n_episodes": 2}, step_timeout_s=60.0)
    farm = Farm(spec)
    prefix, pids = farm.prefix, farm.pids
    res = farm.step(2)  # every spec stands for a micro-batch of two: counts and the divisor are in episodes
    assert (res.n_planned, res.n_valid, res.n_retry, res.decision) == (2 * BATCH, 2 * BATCH - 2, 2, "apply")
    assert _rel(res.grad * 2.0, single_process_step(spec, 2)["grad"]) < TOL  # the oracle counts one per spec
    assert len(segments(prefix)) == 3 and all(map(_alive, pids))
    began = time.monotonic()
    with pytest.raises(FarmError, match=r"worker 1 \(pid \d+\) exited with code 3 during step 3"):
        farm.step(3)
    assert time.monotonic() - began < 30.0  # the death is seen at once, not at the timeout
    assert farm.closed and _gone(prefix, pids)
    with pytest.raises(FarmError, match="farm_closed"):
        farm.step(4)
    farm.close()


def test_a_silent_worker_times_out_and_nothing_is_left():
    farm = Farm(reference_spec(2, seed=SEED, batch=BATCH, faults={"1": {"sleep": [0]}}, step_timeout_s=6.0))
    prefix, pids = farm.prefix, farm.pids
    assert farm.step(0).decision == "apply"
    began = time.monotonic()
    with pytest.raises(FarmError, match=r"timeout after 6 s: workers \[\d\] silent during step 1"):
        farm.step(1)
    assert 5.5 < time.monotonic() - began < 40.0
    assert farm.closed and _gone(prefix, pids)


def test_close_and_an_exception_leave_no_segment_and_no_child():
    before = signal.getsignal(signal.SIGTERM)
    farm = Farm(_spec(1))
    prefix, pids = farm.prefix, farm.pids
    assert sorted(name.rsplit(".", 1)[1] for name in segments(prefix)) == ["grad", "params", "slab"]
    assert all(map(_alive, pids)) and farm.step(0).decision == "apply"
    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL  # guarded while a farm is open
    model = farm.parameters()
    farm.close()
    farm.close()  # idempotent
    assert farm.closed and _gone(prefix, pids)
    assert bool(torch.isfinite(model.head_w).all())  # the leader module outlives the farm
    with pytest.raises(ZeroDivisionError):
        with Farm(_spec(2)) as farm:
            prefix, pids = farm.prefix, farm.pids
            assert len(segments(prefix)) == 3 and all(map(_alive, pids))
            farm.step(0)
            1 / 0
    assert _gone(prefix, pids)
    assert signal.getsignal(signal.SIGTERM) is before  # and the guard goes with the last farm


LEADER = """
import json
from research.levels.s3_farm_reference import reference_spec
from research.levels.s3_grad_farm import Farm
with Farm(reference_spec(2, seed=3, batch=8)) as farm:
    farm.step(0)
    print(json.dumps({"prefix": farm.prefix, "pids": farm.pids}), flush=True)
    step = 1
    while True:
        farm.step(step)
        step += 1
"""


@pytest.mark.parametrize("sig, code", [(signal.SIGTERM, 128 + signal.SIGTERM), (signal.SIGKILL, -signal.SIGKILL)])
def test_a_signalled_leader_leaves_no_segment_and_no_child(sig, code):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    proc = subprocess.Popen([sys.executable, "-B", "-c", LEADER], cwd=LANE, env=env, stdout=subprocess.PIPE)
    try:
        assert select.select([proc.stdout], [], [], 90.0)[0], "the leader did not start"
        info = json.loads(proc.stdout.readline())
        assert len(segments(info["prefix"])) == 3 and all(map(_alive, info["pids"]))
        time.sleep(0.3)  # let it be in the middle of training
        proc.send_signal(sig)
        assert proc.wait(timeout=30.0) == code
        # SIGTERM: the leader unwinds and cleans up. SIGKILL: the orphaned workers do it themselves.
        assert _gone(info["prefix"], info["pids"], patience=20.0)
    finally:
        proc.kill()
        proc.stdout.close()


def test_workers_are_pinned_inside_the_inherited_mask_with_one_thread(runs, live):
    inherited = set(os.sched_getaffinity(0))
    for w, rec in list(runs.items()) + [(2, {"info": live.collect("farm.info"), "cpus": live.cpus, "pids": live.pids})]:
        assert [info["worker"] for info in rec["info"]] == list(range(w))
        assert [info["pid"] for info in rec["info"]] == rec["pids"]
        for info, cpu in zip(rec["info"], rec["cpus"]):
            assert info["cpu"] == [cpu] and cpu in inherited  # one logical CPU, never outside the mask
            assert info["torch_threads"] == 1 and info["interop_threads"] == 1
            assert info["env"] == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                   "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
        assert len(set(rec["cpus"])) == min(w, len(inherited))
    assert set(os.sched_getaffinity(0)) == inherited  # the leader's own mask is untouched
    picked = pick_cpus(len(inherited))
    assert sorted(picked) == sorted(inherited) and pick_cpus(len(inherited) + 1)[-1] == picked[0]
    topo = [cpu_topology(cpu) for cpu in picked]
    n_cores, n_domains = len({core for _, core in topo}), len({domain for domain, _ in topo})
    assert len({core for _, core in topo[:n_cores]}) == n_cores  # every physical core before any SMT sibling
    assert len({domain for domain, _ in topo[:n_domains]}) == n_domains  # and every L3 domain before a second core
    outside = next(c for c in range(4096) if c not in inherited)
    with pytest.raises(FarmError, match="cpus_outside_inherited_mask"):
        pick_cpus(1, [outside])
    with pytest.raises(FarmError, match="cpus_outside_inherited_mask"):
        Farm(_spec(1, cpus=(outside,)))
    assert segments(f"{PREFIX}.{os.getpid()}.") == segments(live.prefix)  # the refused farm left nothing


def test_collect_returns_one_object_per_worker(runs, live):
    for w, rec in runs.items():
        before, after = rec["built"]
        assert len(after) == w and all(isinstance(x, int) for x in after)
        assert sum(after) - sum(before) == sum(BATCH - res.n_retry - res.n_fail + (step == 10)
                                               for step, res in rec["steps"].items())  # raise_loss was built
    assert live.collect("pid") == live.pids
    with pytest.raises(FarmError, match="collect_unknown_name"):
        live.collect("no such thing")
    assert not live.closed and live.step(2).decision == "apply"  # a wrong name is not fatal


def test_the_default_plan_is_the_counter_sampler(tmp_path):
    tasks = ["regression"] * 30 + ["binary"] * 14 + ["multiclass"] * 8
    with (tmp_path / "index.jsonl").open("w", encoding="utf-8") as fh:
        for t, task in enumerate(tasks):
            fh.write(json.dumps({
                "table_id": t, "dataset_hash": f"{t:016x}", "canonical_sha256": f"{t:064x}", "family_stem": f"stem{t}",
                "task": task, "n_classes": {"regression": 0, "binary": 2}.get(task, 4), "n_rows": 400 + 37 * t,
                "n_cols": 6 + t % 9, "split": "train", "x_off": 0, "row_off": 0}) + "\n")
    spec = replace(reference_spec(2, seed=SEED), plan_fn=FarmSpec.__dataclass_fields__["plan_fn"].default,
                   plan_kwargs={"arena": str(tmp_path), "batch_size": 16, "quotas": [0.5, 0.25, 0.1, 0.1, 0.05]})
    oracle = single_process_step(spec, 5)
    with Farm(spec) as farm:
        res = farm.step(5)
        assert (res.n_planned, res.n_valid) == (16, 16) and oracle["n_valid"] == 16
        assert _rel(res.grad, oracle["grad"]) < TOL


def test_refusals_and_stale_segments():
    with pytest.raises(FarmError, match="spec_not_json"):
        Farm(reference_spec(1, model_kwargs={"hidden": object()}))
    for bad in ({"n_workers": 0}, {"seed": -1}, {"reduce": "gpu"}, {"loss_fn": "no_colon"}, {"step_timeout_s": 0.0}):
        with pytest.raises(FarmError, match="spec_invalid"):
            Farm(replace(reference_spec(1), **bad))
    with pytest.raises(FarmError, match="batchnorm_running_stats"):
        model_layout(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)))
    with pytest.raises(FarmError, match="parameter_not_cpu_float32"):
        model_layout(torch.nn.Linear(2, 2).double())
    layout = model_layout(ref_model())
    kinds = [kind for _, kind, _, _, _ in layout.entries]
    assert kinds == sorted(kinds, key=("param", "frozen", "buffer").index) and kinds.count("buffer") == 1
    assert layout.dim == sum(n for _, kind, _, n, _ in layout.entries if kind == "param") < layout.n_shared
    # a segment whose leader pid is gone is swept; a live leader's segment stays
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    stale, live_name = f"{PREFIX}.{done.pid}.feedbeef.slab", f"{PREFIX}.{os.getpid()}.feedbeef.slab"
    for name in (stale, live_name):
        (SHM_DIR / name).write_bytes(b"\0" * 16)
    try:
        assert stale in sweep_stale() and segments(stale) == [] and segments(live_name) == [live_name]
    finally:
        (SHM_DIR / live_name).unlink()


def test_step_result_is_plain_data():
    res = StepResult(step=0, grad=torch.zeros(1), grad_norm=0.0, n_planned=0, n_valid=0, n_fail=0, n_retry=0,
                     n_nonfinite=0, loss_mean=float("nan"), sq_norm_sum=0.0, ep_sq_norm_sum=0.0, diagnostics={},
                     timings={}, workers={})
    assert res.decision == "" and res.fail_reasons == {}
    assert spike_rule([], min_history=1)(res) is None
