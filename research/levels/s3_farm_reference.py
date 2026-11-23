"""Reference problem and single-process oracle of the gradient farm (plan M3, stage G1).

A small, fully synthetic training problem that exercises everything the farm promises, and the
plain in-process definition of a farm step that the tests compare against.

* ``ref_model``     an MLP with dropout (global-RNG draws), a 0-d parameter, a frozen parameter and
                    a float32 buffer (both shared, outside the gradient), an int64 buffer (private)
                    and optional padding parameters that set the gradient dimension D for benchmarks.
* ``ref_plan``      ``batch`` specs per step with hashed row counts, so shards differ in load.
* ``ref_episode``   data from ``(data_seed, step, slot)`` only; fault injection per ``(step, slot)``.
* ``ref_loss``      noisy inputs (the per-episode generator), a head loss and a differentiable ridge
                    readout fitted on one half of the rows and scored on the other.
* ``single_process_step``  the oracle: plan order, per-episode seeding, drop rules, float64 mean.

Faults live in ``episode_kwargs["faults"] = {"<step>": {kind: [slots]}}`` with kinds ``retry``,
``raise`` (in episode_fn), ``raise_loss`` (in loss_fn), ``nan_loss``, ``nan_grad`` (finite loss,
NaN gradient), ``exit`` (``os._exit`` inside the worker) and ``sleep``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .s3_farm_shm import EPISODE_FAULTS, RETRY, episode_seed, resolve
from .s3_grad_farm import FarmSpec
from .s3_sampler_index import key64, unit

HERE = __name__
PAD_CHUNK = 1_000_000  # floats per padding parameter: the size of a large real layer


@dataclass(frozen=True)
class RefSpec:
    step: int
    slot: int
    rows: int
    cells: int
    n_episodes: int = 1  # > 1: the spec stands for a micro-batch and its loss for a sum over it


def ref_plan(seed: int, batch: int = 24, rows: int = 48, d_in: int = 8,
             n_episodes: int = 1) -> Callable[[int], list[RefSpec]]:
    def plan(step: int) -> list[RefSpec]:
        out = []
        for slot in range(batch):
            n = rows + 2 * int(unit(key64("ref-plan", seed, step, slot), 0) * rows / 2)
            out.append(RefSpec(step, slot, n, n * d_in, n_episodes))
        return out
    return plan


class RefNet(nn.Module):
    def __init__(self, d_in: int = 8, hidden: int = 32, depth: int = 2, dropout: float = 0.1, pad: int = 0,
                 init_seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(init_seed)  # no draw from the global RNG
        dims = [d_in] + [hidden] * depth
        self.weights = nn.ParameterList(
            nn.Parameter(torch.randn((b, a), generator=gen) / math.sqrt(a)) for a, b in zip(dims, dims[1:]))
        self.biases = nn.ParameterList(nn.Parameter(0.1 * torch.randn(b, generator=gen)) for b in dims[1:])
        self.head_w = nn.Parameter(torch.randn(hidden, generator=gen) / math.sqrt(hidden))
        self.head_b = nn.Parameter(torch.zeros(1))
        self.log_ridge = nn.Parameter(torch.zeros(()))
        self.gain = nn.Parameter(torch.ones(d_in), requires_grad=False)
        self.register_buffer("shift", torch.zeros(d_in))
        self.register_buffer("version", torch.tensor(3))
        self.drop = nn.Dropout(dropout)
        sizes = [PAD_CHUNK] * (pad // PAD_CHUNK) + ([pad % PAD_CHUNK] if pad % PAD_CHUNK else [])
        self.pad = nn.ParameterList(nn.Parameter(0.01 * torch.randn(n, generator=gen)) for n in sizes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = (x + self.shift) * self.gain
        for w, b in zip(self.weights, self.biases):
            h = self.drop(torch.tanh(h @ w.T + b))
        return h


def ref_model(**kwargs: Any) -> RefNet:
    return RefNet(**kwargs)


def _fault(cfg: dict[str, Any], spec: Any) -> str:
    for kind, slots in cfg.get("faults", {}).get(str(spec.step), {}).items():
        if spec.slot in slots:
            return kind
    return ""


def ref_episode_init(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {"cfg": kwargs, "built": 0, "pid": os.getpid()}


def ref_episode(spec: Any, state: dict[str, Any]) -> Any:
    cfg = state["cfg"]
    fault = _fault(cfg, spec)
    if fault == "exit":
        os._exit(int(cfg.get("exit_code", 3)))
    if fault == "sleep":
        time.sleep(float(cfg.get("sleep_s", 60.0)))
    if fault == "retry":
        return RETRY
    if fault == "raise":
        raise ValueError("reference fault")
    d_in, rows = int(cfg.get("d_in", 8)), int(getattr(spec, "rows", cfg.get("rows", 48)))
    rng = np.random.Generator(np.random.Philox(key64("ref-data", int(cfg.get("data_seed", 0)), spec.step, spec.slot)))
    x = rng.standard_normal((rows, d_in), dtype=np.float32)
    w = rng.standard_normal(d_in, dtype=np.float32)
    y = np.tanh(0.5 * (x @ w)) + 0.1 * rng.standard_normal(rows, dtype=np.float32)
    state["built"] += 1
    return {"X": x, "y": y.astype(np.float32), "fault": fault}


def ref_loss(model: RefNet, episode: dict[str, Any], generator: torch.Generator, noise: float = 0.05,
             pad_weight: float = 1e-3) -> tuple[torch.Tensor, dict[str, float]]:
    if episode["fault"] == "raise_loss":
        raise KeyError("reference fault")
    x, y = torch.from_numpy(episode["X"]), torch.from_numpy(episode["y"])
    x = x + noise * torch.randn(x.shape, generator=generator)
    h = model.features(x)
    mse = ((h @ model.head_w + model.head_b - y) ** 2).mean()
    half = x.shape[0] // 2
    fit, qry = h[:half], h[half:]
    gram = fit.T @ fit / half + 0.1 * torch.exp(model.log_ridge) * torch.eye(h.shape[1])
    beta = torch.linalg.solve(gram, fit.T @ y[:half] / half)
    ridge = ((qry @ beta - y[half:]) ** 2).mean()
    loss = mse + ridge
    for p in model.pad:  # one multiply and one written gradient per padding block, like a cheap layer
        loss = loss + 0.5 * pad_weight * (p * p).sum()
    if episode["fault"] == "nan_loss":
        loss = loss * float("nan")
    if episode["fault"] == "nan_grad":  # sqrt'(0) = inf, times an upstream 0: a finite loss with a NaN gradient
        loss = loss + 0.0 * torch.sqrt(model.head_b * 0.0).sum()
    return loss, {"mse": float(mse), "ridge": float(ridge), "rows": float(x.shape[0])}


def reference_spec(n_workers: int = 1, *, seed: int = 0, batch: int = 24, rows: int = 48, d_in: int = 8,
                   hidden: int = 32, depth: int = 2, pad: int = 0, faults: dict[str, Any] | None = None,
                   **overrides: Any) -> FarmSpec:
    """FarmSpec of the reference problem."""
    episode_kwargs = {"d_in": d_in, "data_seed": seed + 1, "faults": faults or {}}
    episode_kwargs.update(overrides.pop("episode_kwargs", {}))
    fields: dict[str, Any] = dict(
        model=f"{HERE}:ref_model", model_kwargs={"d_in": d_in, "hidden": hidden, "depth": depth, "pad": pad},
        episode_fn=f"{HERE}:ref_episode", episode_init=f"{HERE}:ref_episode_init", episode_kwargs=episode_kwargs,
        loss_fn=f"{HERE}:ref_loss", plan_fn=f"{HERE}:ref_plan",
        plan_kwargs={"batch": batch, "rows": rows, "d_in": d_in},
        seed=seed, n_workers=n_workers, diag_keys=("mse", "ridge"), step_timeout_s=60.0, start_timeout_s=90.0)
    fields.update(overrides)
    return FarmSpec(**fields)


def single_process_step(spec: FarmSpec, step: int, flat: torch.Tensor | None = None) -> dict[str, Any]:
    """What ``Farm(spec).step(step)`` means, computed in this process.

    Slot order; RNG streams seeded with ``episode_seed(seed, step, slot)``; RETRY, exceptions and
    non-finite losses or gradients are counted and left out; the mean over the valid episodes is
    accumulated in float64. ``flat`` sets the trainable parameters (default: the factory's values).
    Uses ``.backward()`` and ``.grad``, i.e. a different mechanism from the worker's.
    """
    cfg = spec.to_json()
    model = resolve(cfg["model"])(**cfg["model_kwargs"])
    params = [p for p in model.parameters() if p.requires_grad]
    if flat is not None:
        torch.nn.utils.vector_to_parameters(flat.detach().clone(), params)
    model.train(bool(cfg["train_mode"]))
    state = resolve(cfg["episode_init"])(cfg["episode_kwargs"]) if cfg["episode_init"] else cfg["episode_kwargs"]
    plan = resolve(cfg["plan_fn"])(seed=cfg["seed"], **cfg["plan_kwargs"])
    episode_fn, loss_fn, gen = resolve(cfg["episode_fn"]), resolve(cfg["loss_fn"]), torch.Generator()
    total = torch.zeros(sum(p.numel() for p in params), dtype=torch.float64)
    out: dict[str, Any] = dict(n_planned=0, n_valid=0, n_fail=0, n_retry=0, n_nonfinite=0)
    loss_sum = ep_sq = 0.0
    for item in sorted(plan(step), key=lambda s: s.slot):
        out["n_planned"] += 1
        try:
            episode = episode_fn(item, state)
        except EPISODE_FAULTS:
            out["n_fail"] += 1
            continue
        if episode is RETRY:
            out["n_retry"] += 1
            continue
        seed = episode_seed(cfg["seed"], step, item.slot)
        torch.manual_seed(seed)
        gen.manual_seed(seed)
        model.zero_grad(set_to_none=True)
        try:
            res = loss_fn(model, episode, gen, **cfg["loss_kwargs"])
            loss = res[0] if isinstance(res, tuple) else res
            if not math.isfinite(float(loss)):
                out["n_nonfinite"] += 1
                continue
            loss.backward()
        except EPISODE_FAULTS:
            out["n_fail"] += 1
            continue
        g = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1) for p in params]).double()
        if not bool(torch.isfinite(g).all()):
            out["n_nonfinite"] += 1
            continue
        total += g
        out["n_valid"] += 1
        loss_sum += float(loss)
        ep_sq += float(g @ g)
    n = out["n_valid"]
    out.update(grad=total / n if n else total, loss_mean=loss_sum / n if n else math.nan, ep_sq_norm_sum=ep_sq)
    return out
