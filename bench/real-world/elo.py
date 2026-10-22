"""Budgeted Transfer Retention AUC and Elo from the committed benchmark ledger.

This is descriptive selection-exposed evidence, never product certification.
No test rows are opened and no model is fitted by this reducer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from elo_math import BASE, PRIOR, LABELS, leaderboard, ratings

HERE = Path(__file__).resolve().parent
METHODS = ["dope", "rows", "kernel", "copula", "arf", "tvae", "ctgan"]
# Equal log-budget grid, 1 KiB through 64 MiB. Set before the next campaign.
BUDGETS = [1024 * 2**i for i in range(17)]


def load(path: Path) -> list[dict]:
    """Identical duplicates are resumptions; conflicting cell receipts fail closed."""
    seen = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["dataset"], r["method"], r["budget"], r["seed"])
        if key in seen and seen[key] != r:
            raise ValueError("conflicting duplicate benchmark cell")
        seen[key] = r
    return list(seen.values())


def schedule_from(records: list[dict], methods: list[str]) -> dict:
    """Historical fallback. For new campaigns supply an explicit schedule JSON."""
    return {m: sorted({(r["budget"], r["seed"]) for r in records if r["method"] == m}) for m in methods}


def curve(records: list[dict], schedule: list, budgets: list[int]) -> dict:
    by_key = {(r["budget"], r["seed"]): r for r in records}
    expected = {tuple(k) for k in schedule}
    if not expected or not expected.issubset(by_key):
        return {"status": "incomplete", "missing_cells": len(expected - by_key.keys()), "btr_auc": None}
    groups = defaultdict(list)
    failures = 0
    for key in sorted(expected):
        r = by_key[key]
        if r["status"] == "failed":
            failures += 1
            groups[key[0]].append(None)
        elif r["status"] == "ok":
            b, robust = r.get("bytes"), r.get("robust")
            if (isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b)
                    or b <= 0 or robust is None or not math.isfinite(robust)):
                return {"status": "unqualified", "btr_auc": None}
            groups[key[0]].append((b, max(0., min(1., robust))))
        else:
            raise ValueError("unknown benchmark cell status")
    # A fixed configuration has its seed median at each deployment cap;
    # failed/oversized seeds contribute zero. Then take the tested envelope.
    values = [max(float(np.median([0. if r is None or r[0] > b else r[1] for r in rs]))
                  for rs in groups.values()) for b in budgets]
    x = np.log(np.array(budgets, dtype=float))
    auc = float(np.sum(np.diff(x) * (np.array(values[1:]) + values[:-1]) / 2) / (x[-1] - x[0]))
    t90 = next((b for b, v in zip(budgets, values) if v >= .9), None)
    raw = [r["robust"] for r in records if r.get("status") == "ok" and r.get("robust") is not None]
    return {"status": "complete", "btr_auc": 100 * auc, "retention_curve": values,
            "grid_t90_bytes": t90, "failed_cells": failures, "scheduled_cells": len(expected),
            "negative_retention_cells": sum(v < 0 for v in raw)}


def summarize(records: list[dict], methods: list[str], budgets: list[int], margin: float,
              draws: int, seed: int, schedule: dict | None = None,
              dataset_ids: list[str] | None = None) -> dict:
    if (len(methods) < 2 or len(set(methods)) != len(methods) or len(budgets) < 2
            or any(isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in budgets)
            or budgets != sorted(set(budgets)) or draws < 1 or not math.isfinite(margin) or margin < 0):
        raise ValueError("need unique methods, increasing positive integer budgets, draws >= 1 and a finite nonnegative tie margin")
    explicit = schedule is not None
    schedule = schedule if explicit else schedule_from(records, methods)
    if set(schedule) != set(methods):
        raise ValueError("schedule methods must match the selected methods")
    for entries in schedule.values():
        if not entries or any(len(e) != 2 or not isinstance(e[0], str) or not isinstance(e[1], int) for e in entries):
            raise ValueError("each schedule needs nonempty [budget, integer seed] cells")
        if len(set(map(tuple, entries))) != len(entries):
            raise ValueError("duplicate scheduled cell")
    if dataset_ids is not None and (not dataset_ids or any(not isinstance(d, str) or not d for d in dataset_ids)
                                    or len(set(dataset_ids)) != len(dataset_ids)):
        raise ValueError("dataset manifest must be a nonempty list of unique identifiers")
    datasets = sorted(dataset_ids if dataset_ids is not None else
                      {r["dataset"] for r in records if r["method"] in methods})
    groups = defaultdict(list)
    for r in records:
        groups[(r["dataset"], r["method"])].append(r)
    tables = {}
    for d in datasets:
        cells = {m: curve(groups[(d, m)], schedule[m], budgets) for m in methods}
        available = [m for m in methods if cells[m]["status"] == "complete"]
        if len(available) >= 2:
            local = ratings(np.array([[cells[m]["btr_auc"] for m in available]]), margin)
            for m, e in zip(available, local):
                cells[m]["elo"] = float(e)
        tables[d] = cells
    common = [d for d in datasets if all(tables[d][m]["status"] == "complete" for m in methods)]
    if not common:
        raise ValueError("no common complete datasets; inspect the schedule or select another cohort")
    scores = np.array([[tables[d][m]["btr_auc"] for m in methods] for d in common])
    return {"schema_version": 1, "evidence": "selection-exposed-descriptive-test-envelope",
            "kpi": "BTR-AUC-v1", "budgets_bytes": budgets, "tie_margin_points": margin,
            "elo": {"base": BASE, "scale": 400, "prior_log_odds_precision": PRIOR,
                    "fit": "batch-Bradley-Terry", "bootstrap_unit": "dataset", "draws": draws, "seed": seed},
            "schedule_source": "explicit" if explicit else "inferred-from-ledger-not-proof-of-completeness",
            "dataset_source": "explicit" if dataset_ids is not None else "inferred-from-ledger",
            "schedule": schedule, "common_datasets": common, "all_datasets": datasets,
            "leaderboard": leaderboard(scores, methods, margin, draws, seed), "per_dataset": tables}


def markdown(result: dict) -> str:
    rows = ["# Budgeted transfer Elo", "", "Selection-exposed development evidence; no sealed-test or D80 claim.", "",
            "BTR-AUC: area under clipped worst-family retention versus log byte budget (0–100).",
            "This is the envelope of tested configurations; it does not predict validation-selected deployment performance.",
            f"Common complete cohort: {len(result['common_datasets'])}/{len(result['all_datasets'])} datasets. "
            f"Tie margin: {result['tie_margin_points']:g} BTR-AUC points. 95% intervals resample datasets.", "",
            "| Alternative | Elo [95% interval] | Mean BTR-AUC | W / T / L |", "|---|---:|---:|---:|"]
    for r in result["leaderboard"]:
        rows.append(f"| {LABELS.get(r['method'], r['method'])} | {r['elo']:.0f} [{r['lo']:.0f}, {r['hi']:.0f}] | "
                    f"{r['mean_btr_auc']:.1f} | {r['wins']} / {r['ties']} / {r['losses']} |")
    rows += ["", "![Elo ratings and paired dataset intervals](elo.svg)", "",
             "## Individual dataset rankings", "", "Local Elo uses only the alternatives complete on that dataset. "
             "Its 1500 center is local, so compare alternatives within a row group, not scores across datasets.", "",
             "![Per-dataset local Elo with missing alternatives explicit](datasets.svg)", "",
             "| Dataset | Alternative | BTR-AUC | Local Elo | Status |", "|---|---|---:|---:|---|"]
    for d, cells in result["per_dataset"].items():
        for m, c in sorted(cells.items(), key=lambda kv: (-(kv[1].get("elo") or 0), kv[0])):
            score = "—" if c["btr_auc"] is None else f"{c['btr_auc']:.1f}"
            elo = "—" if "elo" not in c else f"{c['elo']:.0f}"
            rows.append(f"| {d} | {m} | {score} | {elo} | {c['status']} |")
    return "\n".join(rows) + "\n"


def outputs(result: dict) -> dict[str, bytes]:
    from render_elo import svg, dataset_svg
    table = io.StringIO()
    writer = csv.writer(table, lineterminator="\n")
    writer.writerow(["dataset", "method", "btr_auc", "elo", "status", "failed_cells"])
    for d, cells in result["per_dataset"].items():
        for m, c in cells.items():
            writer.writerow([d, m, c["btr_auc"], c.get("elo"), c["status"], c.get("failed_cells")])
    return {"elo.json": (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
            "elo.md": markdown(result).encode(), "datasets.csv": table.getvalue().encode(),
            "elo.svg": svg(result).encode(), "datasets.svg": dataset_svg(result).encode()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=HERE / "results/confirm/records.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "results/elo")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)), help="ascending comma-separated byte caps")
    ap.add_argument("--schedule", type=Path, help="JSON {method: [[budget_label, seed], ...]}; predeclare for new studies")
    ap.add_argument("--datasets", type=Path, help="JSON list of predeclared dataset IDs, including wholly absent jobs")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--tie-margin", type=float, default=1.)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    result = summarize(load(a.records), a.methods.split(","), [int(b) for b in a.budgets.split(",")],
                       a.tie_margin, a.draws, a.seed, json.loads(a.schedule.read_text()) if a.schedule else None,
                       json.loads(a.datasets.read_text()) if a.datasets else None)
    result["provenance"] = {"records_sha256": hashlib.sha256(a.records.read_bytes()).hexdigest(),
                            "reducer_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in [Path(__file__), HERE / "elo_math.py", HERE / "render_elo.py"]}}
    for name, data in outputs(result).items():
        path = a.out / name
        if a.check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit(f"Elo surface drift: {path}")
        else:
            a.out.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    print(f"Elo: {len(result['common_datasets'])} common datasets; {'checked' if a.check else a.out}")


if __name__ == "__main__":
    main()
