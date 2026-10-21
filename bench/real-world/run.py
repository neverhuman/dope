"""Run the DOPE 1.0 benchmark (resumable; one JSON line per cell).

    python3 bench/real-world/run.py --data target/bench-data --panel confirm \
        --out bench/real-world/results/confirm

A cell is (dataset, method, budget, seed). For each cell the method's
synthetic table (as many rows as the real training split) trains every
auditor family; each is scored on the held-out real test rows against the
same family trained on the real training rows, giving retention
R = (L_null − L_syn) / (L_null − L_real).

Every fit-and-sample call runs in a child process with a wall-clock limit
(GEN_TIMEOUT); a call that exceeds it is recorded as a failed cell
("Timeout") and stays in every denominator, like any other failure.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auditors  # noqa: E402
import methods  # noqa: E402

BUDGETS = ["1k", "2k", "4k", "8k", "16k", "64k"]
HEADLINE = "4k"
GEN_TIMEOUT = 1800  # seconds of wall time per fit-and-sample call


def cells(names: list[str]) -> list[tuple[str, str, str, int]]:
    out = []
    for m in names:
        if m in ("dope", "rows"):
            for b in BUDGETS + (["full"] if m == "rows" else []):
                for s in ([0, 1, 2] if b == HEADLINE else [0]):
                    out.append((m, b, s))
        elif m in ("copula", "arf", "kernel"):
            out += [(m, "-", s) for s in (0, 1, 2)]
        else:
            out.append((m, "-", 0))
    return out


def as_bytes(b: str) -> int:
    return int(float(b[:-1]) * 1024) if b.endswith("k") else int(b)


def conform(syn: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Give synthetic columns the real table's columns and dtypes."""
    out = pd.DataFrame(index=range(len(syn)))
    for c in train.columns:
        col = syn[c] if c in syn.columns else pd.Series([np.nan] * len(syn))
        if train[c].dtype == object:
            out[c] = col.astype(str).where(col.notna(), np.nan).values
        else:
            out[c] = pd.to_numeric(col, errors="coerce").values
    return out


def generate(method: str, budget: str, seed: int, d: Path, meta: dict, train: pd.DataFrame) -> dict:
    n, target = len(train), meta["target"]
    if method == "dope":
        return methods.dope(d / "train.csv", target, n, seed, budget)
    if method == "dope-independent":
        return methods.dope(d / "train.csv", target, n, seed, HEADLINE, ("--dependence", "independent", "--regimes", "1"))
    if method == "dope-linear":
        return methods.dope(d / "train.csv", target, n, seed, HEADLINE, ("--plan", "9,12,0", "--no-trees"))
    if method == "dope-notrees":
        return methods.dope(d / "train.csv", target, n, seed, HEADLINE, ("--no-trees",))
    if method == "dope-noaudit":
        return methods.dope(d / "train.csv", target, n, seed, HEADLINE, ("--no-audit",))
    if method == "dope-4x":
        return methods.dope(d / "train.csv", target, 4 * n, seed, HEADLINE)
    if method == "dope-1regime":
        return methods.dope(d / "train.csv", target, n, seed, HEADLINE, ("--regimes", "1"))
    if method == "dope-8bit":
        return methods.dope(d / "train.csv", target, n, seed, HEADLINE, ("--tolerance", "0"))
    if method == "kernel":
        return methods.kernel(d / "train.csv", target, n, seed)
    if method == "rows":
        return methods.rows(train, 1 << 40 if budget == "full" else as_bytes(budget), seed)
    if method == "bootstrap":
        return methods.bootstrap(train, n, seed)
    if method in ("copula", "ctgan", "tvae"):
        return methods.sdv(method, train, n, seed)
    if method == "arf":
        return methods.arf(train, n, seed)
    raise ValueError(method)


# Failure classes a generator may raise; each is recorded by name. Anything else
# crashes the child process and is recorded as "Crash" with no detail.
GENERATOR_ERRORS = (subprocess.CalledProcessError, ArithmeticError, AssertionError, LookupError,
                    MemoryError, OSError, RuntimeError, TypeError, ValueError)


class CellFailure(Exception):
    """A cell whose fit or sample did not produce rows: Timeout, Crash or a named error class."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


def _child(q, args) -> None:
    try:
        q.put(("ok", generate(*args)))
    except GENERATOR_ERRORS as error:  # the class name is all a record keeps
        q.put(("failed", type(error).__name__))


def generate_limited(*args) -> dict:
    """generate() in a forked child with a wall-clock limit; failures raise CellFailure."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, args), daemon=True)
    p.start()
    deadline = time.monotonic() + GEN_TIMEOUT
    while True:
        try:
            kind, value = q.get(timeout=5)
            break
        except queue.Empty:
            if not p.is_alive() and q.empty():
                p.join()
                raise CellFailure("Crash") from None
            if time.monotonic() > deadline:
                p.kill()
                p.join()
                raise CellFailure("Timeout") from None
    p.join()
    if kind == "failed":
        raise CellFailure(value)
    return value


def reference(d: Path, meta: dict, train, test, classes, seed: int, cache: Path) -> dict:
    key = f"{meta['name']}|{seed}"
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            if r["key"] == key:
                return r
    task, target = meta["task"], meta["target"]
    out = {"key": key, "null": auditors.null_loss(task, train, target, test, classes), "real": {}}
    for fam in auditors.FAMILIES:
        pred = auditors.predict(fam, task, train, target, test, classes, seed)
        out["real"][fam] = auditors.loss(task, pred, test[target], classes)
    with cache.open("a") as f:
        f.write(json.dumps(out) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--panel", default="confirm")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", default="dope,rows,bootstrap,copula,arf,dope-independent,dope-1regime,dope-notrees,dope-linear,dope-noaudit,dope-4x")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--budgets", default="", help="restrict budget sweeps, e.g. 4k,full")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    records, ref_cache = a.out / "records.jsonl", a.out / "reference.jsonl"
    done, timed_out = set(), set()
    if records.exists():
        for line in records.read_text().splitlines():
            r = json.loads(line)
            done.add((r["dataset"], r["method"], r["budget"], r["seed"]))
            if r.get("error") == "Timeout":
                timed_out.add((r["dataset"], r["method"]))
    names = sorted(p.name for p in (a.data / a.panel).iterdir() if p.is_dir())
    if a.datasets:
        names = [n for n in names if n in a.datasets.split(",")]
    for name in names:
        d = a.data / a.panel / name
        meta = json.loads((d / "meta.json").read_text())
        train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
        task, target = meta["task"], meta["target"]
        classes = sorted(pd.concat([train[target], test[target]]).astype(str).unique()) if task == "classification" else []
        for method, budget, seed in cells(a.methods.split(",")):
            if a.budgets and budget != "-" and budget not in a.budgets.split(","):
                continue
            if (name, method, budget, seed) in done:
                continue
            if (name, method) in timed_out:  # remaining seeds of a method that hit the limit
                rec = {"dataset": name, "method": method, "budget": budget, "seed": seed, "status": "failed",
                       "error": "Timeout"}
                with records.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{name:<12} {method:<17} {budget:>5} s{seed} FAILED Timeout (earlier seed)", flush=True)
                continue
            t0 = time.perf_counter()
            try:
                g = generate_limited(method, budget, seed, d, meta, train)
            except CellFailure as failure:  # a failed fit is recorded, never dropped
                rec = {"dataset": name, "method": method, "budget": budget, "seed": seed, "status": "failed",
                       "error": failure.kind}
                if rec["error"] == "Timeout":
                    timed_out.add((name, method))
                with records.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{name:<12} {method:<17} {budget:>5} s{seed} FAILED {failure.kind}", flush=True)
                continue
            syn = conform(g["synthetic"], train)
            ref = reference(d, meta, train, test, classes, seed, ref_cache)
            fams = {}
            for fam in auditors.FAMILIES:
                if len(syn.dropna(subset=[target])) < 5:
                    loss = ref["null"]
                else:
                    pred = auditors.predict(fam, task, syn, target, test, classes, seed)
                    loss = auditors.loss(task, pred, test[target], classes)
                gain = ref["null"] - ref["real"][fam]
                fams[fam] = {"null": ref["null"], "real": ref["real"][fam], "syn": loss,
                             "retention": (ref["null"] - loss) / gain if gain > 0 else None,
                             "informative": bool(gain > 0 and gain >= 0.01 * abs(ref["null"]))}
            inf = [v["retention"] for v in fams.values() if v["informative"]]
            rec = {"dataset": name, "task": task, "method": method, "budget": budget, "seed": seed, "status": "ok",
                   "bytes": int(g["bytes"]), "fit_s": g["fit_s"], "sample_s": g["sample_s"],
                   "synthetic_rows": len(syn), "rows": g.get("rows"), "auditors": fams,
                   "robust": min(inf) if inf else None, "report": g.get("report"),
                   "seconds": time.perf_counter() - t0}
            with records.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            r = rec["robust"]
            print(f"{name:<12} {method:<17} {budget:>5} s{seed} {rec['bytes']:>10} B  robust "
                  f"{'n/a' if r is None else f'{r:.3f}'}  " +
                  " ".join(f"{k}={v['retention']:.3f}" for k, v in fams.items() if v["retention"] is not None) +
                  f"  ({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
