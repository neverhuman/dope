"""Aggregate benchmark records into summary JSON, tables and figures.

    python3 bench/real-world/report.py --results bench/real-world/results/confirm

Per dataset, a method's score is the median over seeds of its *robust
retention* (minimum over informative auditor families). Across datasets we
report the median with a 95% percentile bootstrap interval (datasets
resampled, 10,000 draws, fixed seed), and paired DOPE-minus-method
differences as Hodges–Lehmann pseudo-medians with bootstrap intervals and
win/tie/loss counts. TDL_τ of a dataset is the smallest tested artifact that
reaches robust retention τ; datasets that never reach it are "not reached".
A cell whose fit failed or timed out counts as robust retention 0 (a learner
given no rows can only predict the null) and stays in every denominator.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HEADLINE = "4k"
DRAWS, SEED = 10_000, 20261021
ALPHA = 0.05  # two-sided percentile intervals
PAIRED_DRAWS = 2_000
TAU, TAU_LOW = 0.9, 0.8  # retention thresholds for TDL and coverage
BUDGET_KEYS = ["1k", "2k", "4k", "8k", "16k", "64k"]
LABELS = {"dope": "DOPE", "kernel": "DOPE research kernel", "rows": "Real rows (xz)", "copula": "Gaussian copula", "arf": "ARF", "ctgan": "CTGAN",
          "tvae": "TVAE", "bootstrap": "Real rows, resampled", "dope-independent": "DOPE, independent features",
          "dope-linear": "DOPE, linear target", "dope-notrees": "DOPE, network only (no trees)",
          "dope-noaudit": "DOPE, no self-audit", "dope-seed0": "DOPE, 4 KiB, seed 0", "dope-8bit": "DOPE, uniform 8-bit",
          "dope-1regime": "DOPE, one regime", "dope-4x": "DOPE, 4× synthetic rows"}


def load(results: Path) -> list[dict]:
    """Records, one per cell (dataset, method, budget, seed); the first record of a cell wins."""
    out, seen = [], set()
    for line in (results / "records.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["dataset"], r["method"], r["budget"], r["seed"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def per_dataset(records: list[dict]) -> dict:
    """{(method, budget): {dataset: {'robust', 'bytes', 'families': {fam: r}}}} medians over seeds."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[(r["method"], r["budget"])][r["dataset"]].append(r)
        if r["method"] == "dope" and r["budget"] == HEADLINE and r["seed"] == 0:
            # The ablations run one seed; their reference is the default program at that seed.
            groups[("dope-seed0", "-")][r["dataset"]].append(r)
    out = {}
    for key, by_ds in groups.items():
        out[key] = {}
        for ds, rs in by_ds.items():
            ok = [r for r in rs if r["status"] == "ok"]
            if not ok:
                # A cell that produced no rows leaves a learner only the null: retention 0.
                out[key][ds] = {"robust": 0.0, "bytes": None, "families": {}, "status": "failed",
                                "error": rs[0].get("error")}
                continue
            rob = [r["robust"] for r in ok if r["robust"] is not None]
            fams = defaultdict(list)
            for r in ok:
                for f, v in r["auditors"].items():
                    if v["informative"] and v["retention"] is not None:
                        fams[f].append(v["retention"])
            out[key][ds] = {"robust": float(np.median(rob)) if rob else None,
                            "bytes": float(np.median([r["bytes"] for r in ok])),
                            "families": {f: float(np.median(v)) for f, v in fams.items()},
                            "fit_s": float(np.median([r["fit_s"] for r in ok])),
                            "sample_s": float(np.median([r["sample_s"] for r in ok])),
                            "rows_per_s": float(np.median([r["synthetic_rows"] / max(r["sample_s"], 1e-9) for r in ok])),
                            "task": ok[0]["task"], "status": "ok"}
    return out


def boot_median(values: list[float]) -> tuple[float, float, float]:
    v = np.asarray(values, float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(SEED)
    meds = np.median(v[rng.integers(0, len(v), (DRAWS, len(v)))], axis=1)
    return float(np.median(v)), float(np.quantile(meds, ALPHA / 2)), float(np.quantile(meds, 1 - ALPHA / 2))


def hodges_lehmann(d: np.ndarray) -> float:
    w = (d[:, None] + d[None, :]) / 2.0
    return float(np.median(w[np.triu_indices(len(d))]))


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test (ties dropped)."""
    from math import comb
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a: dict, b: dict, fitted_only: bool = False) -> dict | None:
    """Paired DOPE-minus-method contrast over tables; `fitted_only` keeps tables where `a` produced rows."""
    common = sorted(k for k in a if k in b and a[k]["robust"] is not None and b[k]["robust"] is not None
                    and (not fitted_only or a[k]["status"] == "ok"))
    if len(common) < 3:
        return None
    d = np.array([a[k]["robust"] - b[k]["robust"] for k in common])
    rng = np.random.default_rng(SEED)
    hl = [hodges_lehmann(d[rng.integers(0, len(d), len(d))]) for _ in range(PAIRED_DRAWS)]
    wins, losses = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    return {"n": len(common), "a": float(np.median([a[k]["robust"] for k in common])),
            "b": float(np.median([b[k]["robust"] for k in common])), "sign_p": sign_test(wins, losses),
            "hl": hodges_lehmann(d), "lo": float(np.quantile(hl, ALPHA / 2)),
            "hi": float(np.quantile(hl, 1 - ALPHA / 2)), "wins": int((d > 1e-9).sum()), "ties": int((abs(d) <= 1e-9).sum()),
            "losses": int((d < -1e-9).sum())}


def budget_bytes(b: str) -> float:
    return float("inf") if b == "full" else float(b[:-1]) * 1024


def tdl(table: dict, method: str, datasets: list[str], tau: float) -> dict:
    """Per dataset: bytes of the smallest tested artifact with robust >= tau."""
    budgets = sorted({b for (m, b) in table if m == method}, key=budget_bytes)
    out = {}
    for ds in datasets:
        out[ds] = None
        for b in budgets:
            cell = table.get((method, b), {}).get(ds)
            if cell and cell["robust"] is not None and cell["robust"] >= tau:
                out[ds] = cell["bytes"]
                break
    return out


def summarize(records: list[dict], meta: dict | None = None) -> dict:
    table = per_dataset(records)
    datasets = sorted({r["dataset"] for r in records})
    methods = []
    for (m, b), cells in sorted(table.items()):
        rob = [c["robust"] for c in cells.values() if c["robust"] is not None]
        failed = sum(1 for c in cells.values() if c["status"] == "failed")
        med, lo, hi = boot_median(rob)
        fams = defaultdict(list)
        for c in cells.values():
            for f, v in c["families"].items():
                fams[f].append(v)
        byts = [c["bytes"] for c in cells.values() if c["bytes"] is not None]
        methods.append({"method": m, "budget": b, "label": LABELS.get(m, m), "datasets": len(rob), "failed": failed,
                        "robust": {"median": med, "lo": lo, "hi": hi},
                        "families": {f: boot_median(v)[0] for f, v in fams.items()},
                        "bytes": {"median": float(np.median(byts)) if byts else None,
                                  "q25": float(np.quantile(byts, 0.25)) if byts else None,
                                  "q75": float(np.quantile(byts, 0.75)) if byts else None},
                        "fit_s": float(np.median([c["fit_s"] for c in cells.values() if c["status"] == "ok"] or [np.nan])),
                        "rows_per_s": float(np.median([c["rows_per_s"] for c in cells.values() if c["status"] == "ok"] or [np.nan])),
                        "reach90": sum(1 for v in rob if v >= TAU), "reach80": sum(1 for v in rob if v >= TAU_LOW)})
    head = table.get(("dope", HEADLINE), {})
    contrasts = {}
    ablation = {}
    base = table.get(("dope-seed0", "-"), {})
    for m in ("dope-independent", "dope-1regime", "dope-notrees", "dope-linear", "dope-noaudit", "dope-4x"):
        if (m, "-") in table:
            pr = paired(table[(m, "-")], base)
            if pr:
                ablation[m] = pr
    for (m, b), cells in table.items():
        if (m, b) != ("dope", HEADLINE) and (b in ("-", HEADLINE, "full")) and m != "dope-seed0":
            p = paired(head, cells)
            if p:
                contrasts[f"{m}@{b}"] = p
    # Equal bytes: DOPE minus xz-compressed real rows at every budget both were run at.
    equal = {}
    for b in BUDGET_KEYS:
        if ("dope", b) in table and ("rows", b) in table:
            pr = paired(table[("dope", b)], table[("rows", b)], fitted_only=True)
            if pr:
                pr["dope_failed"] = sum(1 for k, c in table[("dope", b)].items()
                                        if c["status"] == "failed" and k in table[("rows", b)])
                equal[b] = pr
    # The research kernel (dope-kernel fit-table): DOPE at small budgets against it, over the tables the
    # kernel supports (regression and binary targets); a DOPE program that did not fit counts as 0.
    kernel = None
    kern = table.get(("kernel", "-"), {})
    if kern:
        multiclass = {k for k, m in (meta or {}).items() if m.get("classes", 0) > 2}
        ok = {k: v for k, v in kern.items() if k not in multiclass}
        vs = {}
        for b in ("1k", "2k", "4k"):
            if ("dope", b) in table:
                pr = paired(table[("dope", b)], ok)
                if pr:
                    vs[b] = pr
        fitted = [v["bytes"] for v in ok.values() if v["status"] == "ok"]
        kernel = {"supported": len(ok), "unsupported": len(kern) - len(ok), "vs": vs,
                  "bytes": float(np.median(fitted)) if fitted else None}
    # How well the compile-time self-audit predicts measured robust retention (every fitted DOPE cell).
    pairs = [(r["report"]["self_audit"], r["robust"]) for r in records
             if r["method"] == "dope" and r.get("status") == "ok" and r.get("robust") is not None
             and (r.get("report") or {}).get("self_audit") is not None]
    audit = None
    if len(pairs) >= 5:
        a, m = np.array(pairs).T
        rank = lambda v: np.argsort(np.argsort(v))
        audit = {"n": len(pairs), "spearman": float(np.corrcoef(rank(a), rank(m))[0, 1]),
                 "mae": float(np.median(np.abs(m - a))), "bias": float(np.median(m - a)), "pairs": pairs}
    dope_failed = {b: sum(1 for c in table.get(("dope", b), {}).values() if c["status"] == "failed")
                   for b in BUDGET_KEYS}
    coverage = {}
    for m in ("dope", "rows"):
        for tau in (TAU_LOW, TAU):
            coverage[f"{m}@{tau}"] = tdl(table, m, datasets, tau)
    return {"datasets": datasets, "methods": methods, "contrasts": contrasts, "equal_bytes": equal, "ablation": ablation, "kernel": kernel, "self_audit": audit, "dope_failed": dope_failed, "tdl": coverage,
            "per_dataset": {f"{m}@{b}": v for (m, b), v in table.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--data", type=Path, help="prepared panel directory (embeds dataset metadata)")
    a = ap.parse_args()
    records = load(a.results)
    meta = None
    if a.data:
        meta = {d: json.loads((a.data / d / "meta.json").read_text())
                for d in sorted({r["dataset"] for r in records}) if (a.data / d / "meta.json").exists()}
    s = summarize(records, meta)
    if meta is not None:
        s["meta"] = meta
    (a.results / "summary.json").write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
    for row in sorted(s["methods"], key=lambda r: (r["method"], budget_bytes(r["budget"]) if r["budget"] != "-" else 0)):
        r = row["robust"]
        print(f"{row['label']:<28} {row['budget']:>5}  n={row['datasets']:>2}  robust {r['median']:.3f} "
              f"[{r['lo']:.3f}, {r['hi']:.3f}]  bytes {row['bytes']['median'] or 0:>12,.0f}  "
              f"T90 {row['reach90']}/{row['datasets']}  " +
              " ".join(f"{f}={v:.3f}" for f, v in sorted(row["families"].items())))
    for k, c in sorted(s["contrasts"].items()):
        print(f"DOPE@4k − {k:<22} HL {c['hl']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]  W/T/L {c['wins']}/{c['ties']}/{c['losses']}  n={c['n']}")


if __name__ == "__main__":
    main()
