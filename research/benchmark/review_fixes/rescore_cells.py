"""Re-score stored DOPE sample CSVs with the pinned evaluator (predeclare_v2 replay).

The input root uses the fit-seed runner layout: <root>/<dataset>/fit-<seed>/
sample-<sample_seed>-<size>n.csv beside its stored .metric.json. Each CSV is
scored again with research/benchmark/pilot_metrics.py, whose digest must equal
predeclare_v2. The new record keeps the evaluator digest and library versions.
The run prints only equality counts and the largest absolute difference, so
monitoring stays outcome-blind. This process never opens a file named test.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from research.benchmark.review_fixes import fit_seeds as v1
from research.benchmark.review_fixes.fit_seeds_v2 import load_contract

CELL = re.compile(r"^sample-(\d+)-([14])n\.csv$")
AUDITORS = ("catboost", "linear", "mlp")


def cells(root: Path) -> list[dict]:
    found = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for fit_dir in sorted(dataset_dir.glob("fit-*")):
            fit_seed = int(fit_dir.name.split("-", 1)[1])
            for csv_path in sorted(fit_dir.glob("sample-*n.csv")):
                match = CELL.match(csv_path.name)
                if not match:
                    continue
                v1._refuse_test(csv_path)
                found.append({"dataset": dataset_dir.name, "fit_seed": fit_seed,
                              "sample_seed": int(match.group(1)), "size": int(match.group(2)),
                              "csv": str(csv_path), "stored": str(csv_path.with_suffix(".metric.json"))})
    return found


def _stored_utility(path: Path) -> dict | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("status") != "ok":
        return None
    utility = record.get("utility") or (record.get("metrics") or {}).get("utility") or {}
    return {name: (utility.get(name) or {}).get("retention") for name in AUDITORS}


def compare(stored: dict | None, fresh: dict) -> tuple[bool, float]:
    if stored is None:
        return False, math.inf
    worst = 0.0
    equal = True
    for name in AUDITORS:
        left, right = stored.get(name), (fresh.get(name) or {}).get("retention")
        if left is None or right is None:
            equal = equal and left is None and right is None
            continue
        if left != right:
            equal = False
        worst = max(worst, abs(float(left) - float(right)))
    return equal, worst


def score(job: dict) -> dict:
    pilot = v1._load_pilot()
    worker = Path(job["workers"]) / job["dataset"]
    auth = v1._authenticate(worker, job["dataset"])
    csv_path = Path(job["csv"])
    report = pilot.measure(worker / "train.csv", worker / "validation.csv", csv_path, auth["task"],
                           seed=job["auditor_seed"])
    if report["implementation_sha256"] != job["evaluator_sha256"]:
        raise RuntimeError("evaluator digest changed")
    equal, worst = compare(_stored_utility(Path(job["stored"])), report["utility"])
    record = {"status": "ok", "dataset": job["dataset"], "method": "DOPE", "binary_role": job["role"],
              "fit_seed": job["fit_seed"], "sample_seed": job["sample_seed"], "size": job["size"],
              "report": report, "synthetic_sha256": v1._sha256(csv_path),
              "stored_metric_sha256": v1._sha256(Path(job["stored"])) if Path(job["stored"]).is_file() else None,
              "replay_equal": equal, "replay_max_abs_difference": None if math.isinf(worst) else worst,
              "predeclare_sha256": job["predeclare_sha256"], "official_tests_opened": False}
    out = Path(job["out"]) / job["dataset"] / f"fit-{job['fit_seed']}" / Path(job["csv"]).with_suffix(".metric.json").name
    v1._atomic_json(out, record)
    return {"equal": equal, "worst": worst}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--role", default="diagnostic")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if os.uname().nodename != args.host:
        raise SystemExit(f"refusing to run on {os.uname().nodename}")
    contract = load_contract()
    if v1._sha256(v1.PILOT_PATH) != contract["evaluator_sha256"]:
        raise SystemExit("evaluator digest does not match predeclare_v2")
    out = args.out.resolve()
    if out.is_relative_to(args.root.resolve()):
        raise SystemExit("the replay output must sit outside the input root")
    plan = [{**cell, "workers": str(args.workers), "out": str(out), "role": args.role,
             "auditor_seed": contract["auditor_seed"], "evaluator_sha256": contract["evaluator_sha256"],
             "predeclare_sha256": contract["sha256"]} for cell in cells(args.root)]
    print(f"cells {len(plan)}", flush=True)
    equal = unequal = failed = 0
    worst = 0.0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for job, future in zip(plan, [pool.submit(score, job) for job in plan]):
            try:
                result = future.result()
            except (ValueError, RuntimeError, OSError) as error:
                failed += 1
                print(f"failed {job['dataset']} {job['fit_seed']} {job['sample_seed']} {job['size']} "
                      f"{type(error).__name__}", flush=True)
                continue
            equal += result["equal"]
            unequal += not result["equal"]
            if math.isfinite(result["worst"]):
                worst = max(worst, result["worst"])
    print(f"replay equal {equal} unequal {unequal} failed {failed} max_abs_difference {worst:.3g}", flush=True)


if __name__ == "__main__":
    main()
