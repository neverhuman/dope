"""Feed ``SingleEncV8`` training from the parallel episode factory (training-coupled batch code).

Every episode is a fresh random split of a real arena table; the DSE8 packet, the receiver
decoder's inputs and the evidence bank are computed from that episode's FIT rows by the Rust
``native_v8`` stage inside the factory's producer processes, so the trainer never builds an
episode itself and nothing row-derived is cached.

The trainer asks ``provider(step, b)`` for episode ``b`` of optimizer step ``step``; it is ring
slot ``b`` of that step, so the sampler's batch is exactly ``episodes_per_step`` and its strata
(which own contiguous slot ranges) keep their quotas. A refused slot (a counted, value-free code)
is filled with the next slot of the same step that built, cyclically; the stream stays a pure
function of (seed, step) for any producer count, and a resumed run replays it exactly.

Only the rows that enter the loss are capped (``max_loss_fit`` / ``max_loss_qry``, an evenly
spaced subset in row order); the packet and the evidence already used every FIT row.

    python -m research.levels.s3_v8_feed --arena DIR --producers 4 [--producer-cpus 0,1,2,3] \\
        -- <s3_v8_trainer arguments>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

from . import s3_episode_factory as ef

LOSS_KEYS_FIT = ("X_fit", "y_fit", "M_fit")
LOSS_KEYS_QRY = ("X_qry", "y_qry", "M_qry")


def _cap_rows(ep: dict[str, Any], keys: Sequence[str], cap: int) -> None:
    n = ep[keys[0]].shape[0]
    if cap <= 0 or n <= cap:
        return
    idx = np.linspace(0, n - 1, cap).round().astype(np.int64)
    for key in keys:
        ep[key] = ep[key][idx]


class FactoryFeed:
    """``episode_provider`` for ``s3_v8_trainer.train_v8`` backed by an ``EpisodeFactory``.

    ``spec``'s sampler must plan ``episodes_per_step`` slots per step (see ``v8_factory_spec``).
    """

    def __init__(self, spec: ef.FactorySpec, episodes_per_step: int, producers: int, *, start_step: int = 1,
                 cpus: Sequence[int] | None = None, max_loss_fit: int = 1024, max_loss_qry: int = 512,
                 timeout: float = 3600.0, log_path: str | Path | None = None, capacity: int | None = None,
                 ahead_steps: int = 2) -> None:
        # The ring's n_slots (= slots per step * ahead steps) must divide among the producers.
        ahead = max(2, ahead_steps)
        while (episodes_per_step * ahead) % producers:
            ahead += 1
        self.episodes_per_step = episodes_per_step
        self.max_loss_fit, self.max_loss_qry, self.timeout = max_loss_fit, max_loss_qry, timeout
        cap = capacity or ef.v8_capacity(4096, 1024)
        self.factory = ef.EpisodeFactory(spec, episodes_per_step, producers, cap, start_step=start_step,
                                         ahead_steps=ahead, cpus=cpus)
        self._step = -1
        self._slots: dict[int, dict[str, Any] | str] = {}
        self.failures: dict[str, int] = {}
        self.served = self.substituted = self.masked_fit = self.masked_qry = 0
        self._log = open(log_path, "a", buffering=1) if log_path else None

    def _slot(self, step: int, slot: int) -> dict[str, Any] | str:
        if slot not in self._slots:
            try:
                self._slots[slot] = self.factory.get(step, slot, timeout=self.timeout)["episode"]
            except ef.EpisodeFailed as exc:
                code = str(exc) or "failed"
                self.failures[code] = self.failures.get(code, 0) + 1
                self._slots[slot] = code
        return self._slots[slot]

    def __call__(self, step: int, b: int) -> dict[str, Any]:
        if step != self._step:
            self._finish_step()
            self._step, self._slots = step, {}
        n = self.episodes_per_step
        for k in range(n):
            got = self._slot(step, (b + k) % n)
            if isinstance(got, dict):
                break
        else:
            raise ef.FactoryError("feed_step_empty")  # every slot of this step was refused
        self.substituted += int(k > 0)
        ep = dict(got)
        _cap_rows(ep, LOSS_KEYS_FIT, self.max_loss_fit)
        _cap_rows(ep, LOSS_KEYS_QRY, self.max_loss_qry)
        self.served += 1
        self.masked_fit += int(ep.get("fit_masked", 0))
        self.masked_qry += int(ep.get("qry_masked", 0))
        return ep

    def _finish_step(self) -> None:
        if self._log is not None and self._step >= 0:
            refused = sum(1 for v in self._slots.values() if not isinstance(v, dict))
            self._log.write(json.dumps({"step": self._step, "refused": refused, "failures": self.failures,
                                        "substituted": self.substituted, "t": round(time.time(), 1)}) + "\n")

    def stats(self) -> dict[str, Any]:
        st = self.factory.stats()
        st.update(served=self.served, substituted=self.substituted, failures=dict(self.failures),
                  masked_fit=self.masked_fit, masked_qry=self.masked_qry)
        return st

    def close(self) -> None:
        self._finish_step()
        self.factory.close()
        if self._log is not None:
            self._log.close()

    def __enter__(self) -> "FactoryFeed":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def feed_start_step(done_steps: int, grad_accum: int | None = 1) -> int:
    """First feed step after ``done_steps`` optimizer steps: the trainer asks for feed step
    ``(step - 1) * grad_accum + ga + 1``, so a resumed ring skips every micro-batch already used."""
    return max(0, done_steps) * max(1, int(grad_accum or 1)) + 1


def v8_factory_spec(arena: str, seed: int, episodes_per_step: int, budget: int = 2048,
                    normalized_only: bool = False, **sampler: Any) -> ef.FactorySpec:
    """The v3 preset (``ef.V3_SAMPLER``), planning ``episodes_per_step`` slots, with the ``native_v8`` stage."""
    spec = ef.v3_spec(arena, seed, batch_size=episodes_per_step, budget=budget, **sampler)
    return dataclasses.replace(spec, stage=f"{ef.__name__}:native_v8",
                               stage_kwargs={"budget": budget, "normalized_only": normalized_only})


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arena", required=True)
    ap.add_argument("--producers", type=int, default=4)
    ap.add_argument("--producer-cpus", default="", help="comma list, one CPU per producer")
    ap.add_argument("--max-loss-fit", type=int, default=1024)
    ap.add_argument("--max-loss-qry", type=int, default=512)
    ap.add_argument("--normalized-only", action="store_true")
    a = ap.parse_args(argv[:split])
    from . import s3_v8_trainer as tr  # imports torch: only the leader does, never a producer

    targs = tr.parse_args(argv[split + 1:])
    out = Path(targs.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    if getattr(targs, "resume", False) and (out / "latest.pt").is_file():
        import torch

        done = int(torch.load(out / "latest.pt", map_location="cpu", weights_only=False)["step"])
    start = feed_start_step(done, getattr(targs, "grad_accum_steps", 1))
    cpus = [int(c) for c in a.producer_cpus.split(",") if c != ""] or None
    spec = v8_factory_spec(a.arena, targs.seed, targs.episodes_per_step, budget=targs.budget_bytes,
                           normalized_only=a.normalized_only)
    with FactoryFeed(spec, targs.episodes_per_step, a.producers, start_step=start, cpus=cpus,
                     max_loss_fit=a.max_loss_fit, max_loss_qry=a.max_loss_qry, log_path=out / "feed.jsonl") as feed:
        result = tr.train_v8(targs, episode_provider=feed)
        result["feed"] = feed.stats()
    (out / "feed-summary.json").write_text(json.dumps(result["feed"], indent=2, sort_keys=True, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "last_metrics"}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
