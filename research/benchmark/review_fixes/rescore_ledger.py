"""K0: re-score every stored baseline sample CSV with the pinned evaluator.

predeclare_v2 admits a metric cell to a v2 table only when the evaluator
research/benchmark/pilot_metrics.py, at the declared digest and library
versions, scored it with BLAS and OpenMP at one thread. rescore_adapters.py
turns each ledger into normalized jobs. A CSV is refused when it is named
test.csv, sits under an evaluator directory, or is reached through a symlink.
Its sha256 must equal the sample hash in the ledger. A header row is stripped
into a headerless copy under <out>/tmp. One v2 cell JSON is written per job,
and a missing sample is recorded as sample_absent, never imputed. Printing is
outcome-blind: counts, timings and the largest absolute replay difference,
never a retention value.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import resource
import shutil
import statistics
import tempfile
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from research.benchmark.review_fixes import fit_seeds as v1
from research.benchmark.review_fixes import rescore_adapters as A
from research.benchmark.review_fixes.fit_seeds_v2 import PREDECLARE, load_contract
from research.benchmark.review_fixes.rescore_cells import compare

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
WORKERS = Path("/mnt/fast-scratch/dope-benchmark/s3-v1/prepared/worker")
CELL_FORMAT = "dope-review-fix-v2-cell"
PLAN_FORMAT = "dope-review-fix-k0-plan-v1"
_STATE: dict = {}


def evaluator_contract() -> dict:
    contract = load_contract()
    declared = json.loads(PREDECLARE.read_text())["evaluator"]
    return {"predeclare_sha256": contract["sha256"], "evaluator_sha256": contract["evaluator_sha256"],
            "auditor_seed": contract["auditor_seed"], "dependencies": declared["dependencies"]}


def cell_path(out: Path, job: dict) -> Path:
    method = job["method"].replace("/", "_")
    for part in (method, job["configuration"], job["dataset"]):
        if not part or "/" in part or part.startswith("."):
            raise ValueError("cell identity cannot form a path")
    fit = "none" if job["fit_seed"] is None else str(job["fit_seed"])
    return (out / method / job["configuration"] / job["dataset"] / f"fit-{fit}"
            / f"sample-{job['sample_seed']}-{job['size']}n.cell.json")


def replay(stored: dict | None, utility: dict) -> tuple[bool | None, float | None]:
    """Exact equality per auditor and the largest absolute gap. No stored value gives (None, None)."""
    if stored is None:
        return None, None
    equal, worst = compare(stored, utility)
    return equal, None if math.isinf(worst) else worst


def cell_record(job: dict, status: str, reason: str | None, ledger: dict, contract: dict,
                report: dict | None = None, synthetic: str | None = None,
                equal: bool | None = None, worst: float | None = None) -> dict:
    record = {"format": CELL_FORMAT, "status": status, "reason": reason, "method": job["method"],
              "configuration": job["configuration"], "dataset": job["dataset"], "fit_seed": job["fit_seed"],
              "sample_seed": job["sample_seed"], "size": job["size"], "synthetic_sha256": synthetic,
              "ledger": ledger["file"], "ledger_sha256": ledger["sha256"], "replay_equal": equal,
              "replay_max_abs_difference": worst, "predeclare_sha256": contract["predeclare_sha256"],
              "official_tests_opened": False}
    if status == "ok":
        record["report"] = report
    if job.get("split_seed") is not None:
        record["split_seed"] = job["split_seed"]
    return record


def check_csv_path(path: Path) -> None:
    A.refuse_path(path)
    A.refuse_symlink(path)


def _numeric(line: bytes) -> bool:
    text = line.strip()
    if not text:
        raise A.Refused("empty_csv")
    try:
        [float(field) for field in text.split(b",")]
    except ValueError:
        return False
    return True


def headerless(csv_path: Path, has_header: bool, tmp: Path, meta: dict) -> Path:
    """Return the path pilot_metrics should read: the CSV itself, or a copy without its header row."""
    with csv_path.open("rb") as stream:
        numeric = _numeric(stream.readline())
        if numeric == has_header:
            raise A.Refused("header_absent" if has_header else "unexpected_header")
        if not has_header:
            return csv_path
        tmp.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=tmp, suffix=".headerless.csv")
        meta["headerless_copy"] = name
        with os.fdopen(handle, "wb") as sink:
            shutil.copyfileobj(stream, sink)
    meta["header_stripped"] = True
    return Path(name)


def verify_split(job: dict, worker: Path, task: str) -> None:
    """The stored split partitions must equal a regeneration from the authenticated worker."""
    import numpy as np
    from research.benchmark.review_fixes import controls
    paths = (Path(job["train_path"]), Path(job["validation_path"]))
    for path in paths:
        check_csv_path(path)
    split = controls._resplit(controls._read_table(worker / "train.csv"),
                              controls._read_table(worker / "validation.csv"), task, job["split_seed"])
    if split is None:
        raise A.Refused("split_unavailable")
    for table, path in zip(split, paths):
        buffer = io.BytesIO()
        np.savetxt(buffer, table, delimiter=",")
        if hashlib.sha256(buffer.getvalue()).hexdigest() != v1._sha256(path):
            raise A.Refused("split_input_mismatch")


def _init_worker(tmp: str) -> None:
    for key in THREAD_VARS:
        os.environ[key] = "1"
    # The MLP auditor's non-convergence notice would add two log lines per CSV; it changes no value.
    warnings.filterwarnings("ignore", message="Stochastic Optimizer: Maximum iterations")
    cwd = Path(tmp) / f"cwd-{os.getpid()}"
    cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(cwd)  # CatBoost writes catboost_info into the working directory.


def runtime_versions() -> dict:
    import catboost
    import numpy
    import scipy
    import sklearn
    import sklearn.linear_model  # noqa: F401  loads the OpenMP and BLAS pools before the check
    import sklearn.neighbors  # noqa: F401
    from threadpoolctl import threadpool_info
    return {"dependencies": {"numpy": numpy.__version__, "scipy": scipy.__version__,
                             "scikit-learn": sklearn.__version__, "catboost": catboost.__version__},
            "threadpools": sorted({(item["internal_api"], item["num_threads"]) for item in threadpool_info()}),
            "env": {key: os.environ.get(key) for key in THREAD_VARS}}


def _evaluate(job: dict, task: dict, meta: dict) -> tuple[str, str | None, dict | None, str | None]:
    if job["blocked"]:
        return job["blocked"][0], job["blocked"][1], None, job["csv_sha256"]
    csv_path = Path(job["csv_path"])
    check_csv_path(csv_path)
    if not csv_path.is_file():
        return "sample_absent", "csv_missing", None, job["csv_sha256"]
    observed = v1._sha256(csv_path)
    if observed != job["csv_sha256"]:
        return "sample_hash_mismatch", f"observed_sha256:{observed}", None, job["csv_sha256"]
    if job["target_position"] != "last":
        raise A.Refused("unsupported_target_position")
    worker = Path(task["workers"]) / job["dataset"]
    auth = _STATE.setdefault("auth", {})
    if job["dataset"] not in auth:
        auth[job["dataset"]] = v1._authenticate(worker, job["dataset"])
    auth = auth[job["dataset"]]
    expected = job.get("expected_inputs") or {}
    if any(expected.get(key) not in (None, auth[f"{key}_sha256"]) for key in ("train", "validation")):
        raise A.Refused("worker_input_mismatch")
    train, validation = worker / "train.csv", worker / "validation.csv"
    if job.get("train_path"):
        verify_split(job, worker, auth["task"])
        train, validation = Path(job["train_path"]), Path(job["validation_path"])
    scored = headerless(csv_path, job["has_header"], Path(task["tmp"]), meta)
    if "pilot" not in _STATE:
        _STATE["pilot"] = v1._load_pilot()
    contract = task["contract"]
    from catboost import CatBoostError
    try:
        report = _STATE["pilot"].measure(train, validation, scored, auth["task"], seed=contract["auditor_seed"])
    except CatBoostError:
        raise A.Refused("CatBoostError") from None
    if report.get("implementation_sha256") != contract["evaluator_sha256"]:
        raise A.Refused("evaluator_digest_mismatch")
    if report.get("dependencies") != contract["dependencies"]:
        raise A.Refused("dependency_mismatch")
    return "ok", None, report, observed


def _cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def score_group(task: dict) -> list[dict]:
    """Score one physical CSV and write the cell of every logical job that shares it."""
    started, cpu = time.monotonic(), _cpu()
    meta = {"header_stripped": False}
    first = task["jobs"][0]
    try:
        status, reason, report, synthetic = _evaluate(first, task, meta)
    except A.Refused as error:
        status, reason, report, synthetic = "failed", error.code, None, first["csv_sha256"]
    except (OSError, ValueError, RuntimeError) as error:
        status, reason, report, synthetic = "failed", type(error).__name__, None, first["csv_sha256"]
    finally:
        if meta.get("headerless_copy"):
            Path(meta["headerless_copy"]).unlink(missing_ok=True)
    wall, cpu = time.monotonic() - started, _cpu() - cpu
    results = []
    for index, job in enumerate(task["jobs"]):
        equal, worst = replay(job["stored_retention"], report["utility"]) if status == "ok" else (None, None)
        path = cell_path(Path(task["out"]), job)
        v1._atomic_json(path, cell_record(job, status, reason, task["ledger"], task["contract"],
                                          report, synthetic, equal, worst))
        results.append({"cell": str(path.relative_to(task["out"])), "status": status, "reason": reason,
                        "route": job["route"], "csv_path": job["csv_path"], "equal": equal, "worst": worst,
                        "header_stripped": meta["header_stripped"], "physical": index == 0 and status == "ok",
                        "wall_seconds": round(wall, 3), "cpu_seconds": round(cpu, 3),
                        "metric_seconds": report["metric_seconds"] if status == "ok" else None})
    return results


def groups_of(jobs: list[dict]) -> list[list[dict]]:
    groups: dict = {}
    for job in jobs:
        key = (("blocked", len(groups)) if job["blocked"] or not job["csv_path"] else
               (job["csv_path"], job["csv_sha256"], job["train_path"], job["validation_path"], job["has_header"]))
        groups.setdefault(key, []).append(job)
    return list(groups.values())


def finished(out: Path, job: dict, ledger: dict, contract: dict) -> dict | None:
    record = v1._read_json(cell_path(out, job))
    if (record and record.get("status") == "ok" and record.get("ledger_sha256") == ledger["sha256"]
            and record.get("predeclare_sha256") == contract["predeclare_sha256"]
            and (record.get("report") or {}).get("implementation_sha256") == contract["evaluator_sha256"]):
        return record
    return None


def summarize(results: list[dict]) -> dict:
    statuses = Counter(item["status"] for item in results)
    reasons = Counter((item["reason"] or "").split(":")[0] for item in results if item["reason"])
    gaps = [item["worst"] for item in results if item["worst"] is not None]
    physical = [item for item in results if item.get("physical")]

    def spread(key: str) -> dict | None:
        values = [item[key] for item in physical if item.get(key) is not None]
        if not values:
            return None
        return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
                "max": max(values), "total": math.fsum(values)}

    return {"cells": len(results), "status_counts": dict(sorted(statuses.items())),
            "reason_counts": dict(sorted(reasons.items())),
            "replay": {"equal": sum(item["equal"] is True for item in results),
                       "unequal": sum(item["equal"] is False for item in results),
                       "not_compared": sum(item["equal"] is None for item in results),
                       "max_abs_difference": max(gaps) if gaps else None},
            "header_stripped": sum(bool(item.get("header_stripped")) for item in results),
            "routes": dict(Counter(item["route"] for item in results)),
            "skipped_finished": sum(bool(item.get("skipped")) for item in results),
            "physical_scored": len(physical), "wall_seconds": spread("wall_seconds"),
            "cpu_seconds": spread("cpu_seconds"), "metric_seconds": spread("metric_seconds")}


def print_summary(name: str, summary: dict) -> None:
    replay_counts, gap = summary["replay"], summary["replay"]["max_abs_difference"]
    print(f"ledger {name} cells {summary['cells']} " + " ".join(f"{k} {v}" for k, v in summary["status_counts"].items())
          + f" | replay equal {replay_counts['equal']} unequal {replay_counts['unequal']} "
          f"not_compared {replay_counts['not_compared']} max_abs_difference "
          f"{'none' if gap is None else format(gap, '.3g')} | header_stripped {summary['header_stripped']} "
          f"skipped_finished {summary['skipped_finished']}", flush=True)
    if summary["reason_counts"]:
        print(f"ledger {name} reasons {json.dumps(summary['reason_counts'], sort_keys=True)}", flush=True)
    for key in ("wall_seconds", "cpu_seconds"):
        if summary[key]:
            item = summary[key]
            print(f"ledger {name} {key} per physical csv n {item['n']} mean {item['mean']:.2f} "
                  f"median {item['median']:.2f} max {item['max']:.2f} total {item['total']:.1f}", flush=True)


def classify(group: list[dict], host: str) -> tuple[str, str, int]:
    """Inventory class, host location and byte count of one physical CSV. Only hashing reads bytes."""
    job = group[0]
    if job["blocked"]:
        status, reason = job["blocked"]
        kind = "no_sample_in_ledger" if reason.startswith("ledger_") else (
            "absent" if status == "sample_absent" else "hash_mismatch")
        return kind, "none", 0
    path, where = Path(job["csv_path"]), A.location(job["csv_path"])
    try:
        check_csv_path(path)
    except A.Refused:
        return "refused", where, 0
    if not path.is_file():
        return (f"remote_{where}" if A.remote_elsewhere(job["csv_path"], host) else "absent"), where, 0
    matched = v1._sha256(path) == job["csv_sha256"]
    return ("resolvable_here" if matched else "hash_mismatch"), where, path.stat().st_size


def inventory(jobs: list[dict], host: str, threads: int) -> dict:
    groups = groups_of(jobs)
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        classes = list(pool.map(lambda group: classify(group, host), groups))
    logical, physical, places, size = Counter(), Counter(), {}, 0
    for group, (kind, where, nbytes) in zip(groups, classes):
        logical[kind] += len(group)
        physical[kind] += 1
        places.setdefault(where, Counter())[kind] += len(group)
        size += nbytes
    return {"cells": len(jobs), "physical": len(groups), "classes": dict(sorted(logical.items())),
            "physical_classes": dict(sorted(physical.items())), "routes": dict(Counter(j["route"] for j in jobs)),
            "locations": {k: dict(sorted(v.items())) for k, v in sorted(places.items())},
            "resolvable_bytes": size}


def _adapters_sha256() -> str:
    return v1._sha256(Path(A.__file__).resolve())


def write_plan(path: Path, ledger: dict, jobs: list[dict], host: str) -> None:
    header = {"format": PLAN_FORMAT, "ledger": ledger["name"], "file": ledger["file"],
              "sha256": ledger["sha256"], "adapters_sha256": _adapters_sha256(), "resolved_on": host,
              "jobs": len(jobs), "official_tests_opened": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".jsonl.tmp")
    staging.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in [header, *jobs]))
    os.replace(staging, path)


def load_plan(path: Path, ledger: dict) -> list[dict]:
    A.refuse_path(path)
    lines = path.read_text().splitlines()
    header, jobs = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]
    if (header.get("format") != PLAN_FORMAT or header.get("ledger") != ledger["name"]
            or header.get("sha256") != ledger["sha256"] or header.get("jobs") != len(jobs)):
        raise SystemExit(f"plan {path.name} does not match the committed ledger")
    if header.get("adapters_sha256") != _adapters_sha256():
        raise SystemExit(f"plan {path.name} was resolved by different adapter code")
    return jobs


def run(planned: dict, args, out: Path, contract: dict) -> None:
    # Hosts share the output root, so each run keeps its scratch to itself.
    tmp = out / "tmp" / f"{args.host}-{os.getpid()}"
    context = mp.get_context("spawn")
    remote = {}
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context, initializer=_init_worker,
                             initargs=(str(tmp),)) as pool:
        runtime = pool.submit(runtime_versions).result()
        if runtime["dependencies"] != contract["dependencies"]:
            raise SystemExit("library versions differ from predeclare_v2")
        if any(threads != 1 for _api, threads in runtime["threadpools"]):
            raise SystemExit("BLAS or OpenMP is not at one thread")
        print(f"runtime ok threadpools {runtime['threadpools']}", flush=True)
        futures, results = {}, {name: [] for name in planned}
        for name, (ledger, jobs) in planned.items():
            local = [job for job in jobs if job["blocked"] or not A.remote_elsewhere(job["csv_path"], args.host)]
            remote[name] = len(jobs) - len(local)
            pending = 0
            for group in groups_of(local):
                done = [finished(out, job, ledger, contract) for job in group]
                if all(done):
                    results[name].extend({"cell": str(cell_path(out, job).relative_to(out)), "status": "ok",
                                          "reason": None, "route": job["route"], "equal": record["replay_equal"],
                                          "worst": record["replay_max_abs_difference"], "skipped": True,
                                          "physical": index == 0, "metric_seconds": record["report"]["metric_seconds"]}
                                         for index, (job, record) in enumerate(zip(group, done)))
                    continue
                task = {"jobs": group, "ledger": ledger, "contract": contract, "out": str(out),
                        "tmp": str(tmp / name), "workers": str(args.workers)}
                futures[pool.submit(score_group, task)] = name
                pending += 1
            print(f"ledger {name} cells {len(jobs)} here {len(local)} remote_skipped {remote[name]} "
                  f"physical_pending {pending}", flush=True)
        for index, future in enumerate(as_completed(futures), start=1):
            results[futures[future]].extend(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"progress {index}/{len(futures)}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    for name, (ledger, jobs) in planned.items():
        if not results[name]:
            print(f"ledger {name} no cells here; manifest left to the host that holds its samples", flush=True)
            continue
        summary = summarize(results[name])
        print_summary(name, summary)
        v1._atomic_json(out / f"manifest-{name}.json", {
            "format": "dope-review-fix-v2-k0-manifest", "ledger": ledger, "host": args.host,
            "runner_sha256": v1._sha256(Path(__file__).resolve()),
            "adapters_sha256": v1._sha256(Path(A.__file__).resolve()), **contract,
            "runtime": runtime, "limit": args.limit, "plan_dir": None if args.plan_dir is None else str(args.plan_dir),
            "stored_auditor_seed": A.STORED_AUDITOR_SEED.get(name),
            "jobs_planned": len(jobs), "remote_skipped": remote[name], "summary": summary, "cells": results[name],
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "official_tests_opened": False})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", action="append", required=True, choices=[*A.LEDGERS, "all"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--workers", type=Path, default=WORKERS)
    args = parser.parse_args(argv)
    if os.uname().nodename != args.host:
        raise SystemExit(f"refusing to run on {os.uname().nodename}")
    for key in THREAD_VARS:
        os.environ[key] = "1"
    out, workers = args.out.resolve(), args.workers.resolve()
    A.refuse_path(out)
    if out.is_relative_to(workers) or workers.is_relative_to(out) or out.is_relative_to(A.RESULTS):
        raise SystemExit("the output root must sit outside the workers and the committed results")
    contract = evaluator_contract()
    if v1._sha256(v1.PILOT_PATH) != contract["evaluator_sha256"]:
        raise SystemExit("evaluator digest does not match predeclare_v2")
    names = list(A.LEDGERS) if "all" in args.ledger else list(dict.fromkeys(args.ledger))
    planned, seen = {}, set()
    for name in names:
        ledger = A.ledger_info(name)
        jobs = (load_plan(args.plan_dir / f"{name}.jsonl", ledger) if args.plan_dir
                else A.resolve(name, args.host))
        jobs = jobs if args.limit is None else jobs[: args.limit]
        for job in jobs:
            path = cell_path(out, job)
            if path in seen:
                raise SystemExit(f"two jobs map to one cell in ledger {name}")
            seen.add(path)
        planned[name] = (ledger, jobs)
    if not args.dry_run:
        run(planned, args, out, contract)
        return
    for name, (ledger, jobs) in planned.items():
        found = inventory(jobs, args.host, args.jobs)
        print(f"inventory {name} cells {found['cells']} physical {found['physical']} "
              + " ".join(f"{k} {v}" for k, v in found["classes"].items()), flush=True)
        for where, counts in found["locations"].items():
            print(f"inventory {name} location {where} " + " ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
        v1._atomic_json(out / f"inventory-{name}.json", {"ledger": ledger, "host": args.host, **found,
                                                       "limit": args.limit, "official_tests_opened": False})
        if args.limit is None:
            write_plan(out / "plans" / f"{name}.jsonl", ledger, jobs, args.host)


if __name__ == "__main__":
    main()
