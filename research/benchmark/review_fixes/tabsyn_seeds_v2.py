"""Fit and sample TabSyn's scaled arm at fit seeds 23, 37, 53, and 71 (predeclare_v2 block B).

predeclare_v2 fixes the arm, its configuration digest, and the seeds. The
operational amendment predeclare_v2.amendments/B-tabsyn-seeds.json pins the
seed-11 lineage plan, the worker files, the runtime lock, the captured adapter,
the host, the CPU slot, and the operation caps, which equal this arm's seed-11
caps. Each fit, and each fit's six samples, runs as one bounded operation of
tabsyn_seeds_worker.py under the pinned runtime python, seed-major, behind the
host's shared GPU lock. --ledger writes the sample and fit ledgers that
rescore_ledger.py later scores with the pinned evaluator. Nothing here reads a
retention, so monitoring stays outcome-blind.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import functools
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from research.benchmark.review_fixes import fit_seeds as v1
from research.benchmark.review_fixes import tabsyn_sample
from research.benchmark.review_fixes.fit_seeds_v2 import gpu_name
from research.benchmark.review_fixes import tabsyn_seeds_worker as W

HERE = Path(__file__).resolve().parent
PREDECLARE = HERE / "predeclare_v2.json"
AMENDMENT = HERE / "predeclare_v2.amendments" / "B-tabsyn-seeds.json"
WORKER = HERE / "tabsyn_seeds_worker.py"
REPOSITORY_ADAPTER = HERE.parent / "tabsyn_adapter.py"
RESULTS = HERE.parent / "results" / "review-fixes-tabsyn-seeds-v2"
# Not fit.json: bind_v2_cells.collect_fits binds DOPE fit records only.
FIT_RECORD = "tabsyn-fit.json"
SAMPLE_RECORD = "tabsyn-samples.json"
GRACE_SECONDS = 5


def charged_bytes(inventory: dict) -> int:
    """Every saved file a sampler reads; intermediate diffusion epoch checkpoints are not charged (v2_bytes.py)."""
    return sum(meta["bytes"] for name, meta in inventory["files"].items() if not name.startswith("diffusion-epoch-"))


def load_amendment(path: Path = AMENDMENT, predeclare: Path = PREDECLARE) -> dict:
    """The block B amendment, checked against predeclare_v2 and the seed-11 sampler."""
    amendment = json.loads(path.read_text())
    contract = json.loads(predeclare.read_text())
    if amendment.get("format") != "dope-review-fix-predeclaration-amendment" or amendment.get("block") != "B":
        raise SystemExit("amendment is not the block B amendment")
    if amendment["amends"]["sha256"] != v1._sha256(predeclare):
        raise SystemExit("amendment names another predeclare_v2")
    declared = contract["methods"]["primary"]["TabSyn"]
    if (amendment["configuration"], amendment["config_sha256"]) != (declared["arm"], declared["config_sha256"]):
        raise SystemExit("amendment arm differs from predeclare_v2")
    if amendment["config_sha256"] != tabsyn_sample.CONFIG_SHA:
        raise SystemExit("amendment arm differs from the seed-11 sampler")
    if amendment["admitted_fit_seeds"] != declared["seeds"] or amendment["fit_seeds"] != declared["seeds"][1:]:
        raise SystemExit("amendment seeds differ from predeclare_v2")
    dope = contract["dope"]
    if amendment["sample_seeds"] != dope["sample_seeds"] or amendment["sizes"] != dope["sizes"]:
        raise SystemExit("amendment sample grid differs from predeclare_v2")
    if tuple(amendment["sample_seeds"]) != tabsyn_sample.SEEDS:
        raise SystemExit("amendment sample seeds differ from the seed-11 sampler")
    caps = amendment["caps"]
    if caps["fit_operation_seconds"] != 600 or caps["sample_operation_seconds"] != tabsyn_sample.TIMEOUT_SECONDS:
        raise SystemExit("amendment caps differ from this arm's seed-11 caps")
    if amendment["close_utc"] != contract["missing_and_stopping"]["close_utc"]:
        raise SystemExit("amendment close differs from predeclare_v2")
    if amendment["adapter"]["sha256"] != v1._sha256(REPOSITORY_ADAPTER):
        raise SystemExit("captured adapter differs from the repository copy")
    return {**amendment, "sha256": v1._sha256(path), "predeclare_sha256": v1._sha256(predeclare)}


def lineages(plan: dict, config_sha: str) -> list[dict]:
    """The 100 seed-11 lineages with the worker pins and joint input digests each fit must match."""
    if plan["configs"][0]["config_sha256"] != config_sha:
        raise SystemExit("seed-11 plan default configuration differs")
    rows, seen = [], set()
    for item in plan["lineages"]:
        identity, dataset = item["learning_identity"], item["dataset"]
        if dataset in seen or identity["dataset"] != dataset or identity["fit_seed"] != 11:
            raise SystemExit("seed-11 plan lineage identity differs")
        files = item["operation_template"]["worker"]["files"]
        if "test.csv" in files:
            raise SystemExit("seed-11 plan pins a test partition")
        seen.add(dataset)
        rows.append({"dataset": dataset, "train_rows": item["train_rows"], "files": files,
                     "joint_train_sha256": identity["joint_train_sha256"],
                     "joint_validation_sha256": identity["joint_validation_sha256"],
                     "projection_sha256": identity["projection_sha256"]})
    if len(rows) != 100:
        raise SystemExit("seed-11 plan does not hold 100 lineages")
    return rows


def seed_major(rows: list[dict], seeds: list[int]) -> list[tuple[int, dict]]:
    return [(seed, row) for seed in seeds for row in rows]


def preflight(row: dict, workers: Path, scoring: Path) -> dict:
    """Pinned worker files, joint digests, and the scorer's copy, read without numpy.

    The fit worker is a header view: the scoring worker's file with one header row
    added. A mismatch is recorded, never repaired.
    """
    worker = workers / row["dataset"]
    try:
        for name, pin in row["files"].items():
            W.checked(worker / name, pin)
        train, validation = W.joint_rows(worker / "train.csv"), W.joint_rows(worker / "validation.csv")
        scored = all((worker / name).read_bytes().split(b"\n", 1)[1] == (scoring / row["dataset"] / name).read_bytes()
                     for name in ("train.csv", "validation.csv"))
    except (OSError, ValueError, IndexError) as error:
        return {"dataset": row["dataset"], "status": "unavailable", "reason": type(error).__name__}
    if not (len(train) == row["train_rows"] and W.joint_sha256(train) == row["joint_train_sha256"]
            and W.joint_sha256(validation) == row["joint_validation_sha256"]):
        return {"dataset": row["dataset"], "status": "unavailable", "reason": "joint_input_differs"}
    if not scored:
        return {"dataset": row["dataset"], "status": "unavailable", "reason": "scoring_input_differs"}
    return {"dataset": row["dataset"], "status": "ok", "reason": None}


def child_env(runtime: dict) -> dict[str, str]:
    """The seed-11 queue's scientific environment."""
    return {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LD_LIBRARY_PATH": runtime["LD_LIBRARY_PATH"],
            "CUDA_VISIBLE_DEVICES": "0", "CUDA_CACHE_DISABLE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1"}


LISTED_CAUSES = ("host_reboot", "foreign_gpu_owner", "storage_refusal", "runtime_lock_mismatch")
LOW_DISK_BYTES = 1 << 30


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def failure_receipt(attempt: Path, mode: str) -> dict | None:
    for name in (f"{mode}.failure.receipt.json", "request.failure.receipt.json"):
        receipt = v1._read_json(attempt / name)
        if receipt is not None:
            return receipt
    return None


def outcome(attempt: Path, mode: str, code: int | None, confirm) -> tuple[str, str | None]:
    """Status of one operation from its exit code and receipts.

    A retryable infrastructure failure needs a predeclared cause that ``confirm``
    establishes on the controller's side; the worker's receipt alone is a claim.
    An operation that leaves no final receipt in the same boot is unknown and final.
    """
    if code in (0, None) and (attempt / f"{mode}.complete.receipt.json").is_file():
        return "ok", None
    if code == 124:
        return "timeout", "wall"
    receipt = failure_receipt(attempt, mode)
    if receipt is not None:
        if receipt.get("exception_type") == "TimeoutError":
            return "timeout", "deadline"
        cause = receipt.get("cause")
        if cause in LISTED_CAUSES and confirm(cause, attempt):
            return "infrastructure_failure", cause
        return "failed", receipt.get("exception_type")
    if confirm("host_reboot", attempt):
        return "infrastructure_failure", "host_reboot"
    return "unknown", "no_final_receipt"


def needs_attempt(record: dict | None) -> bool:
    """Run when absent; one retry after a confirmed infrastructure failure; every other status is final."""
    if record is None:
        return True
    return record["status"] == "infrastructure_failure" and record["attempts"] < 2


def sample_cells(row: dict, cell: Path, sizes: list[int], seeds: list[int]) -> list[dict]:
    return [{"seed": seed, "size": size, "rows": row["train_rows"] * size,
             "path": str(cell / f"sample-{seed}-{size}n.csv")} for size in sizes for seed in seeds]


def sample_status(attempt: Path, cells: list[dict], code: int | None) -> list[dict]:
    """Per-cell status from the worker's per-cell receipt; a cell without one shares the operation's end."""
    ended = (attempt / "sample.complete.receipt.json").is_file() or failure_receipt(attempt, "sample") is not None
    unfinished = "timeout" if code == 124 else ("failed" if ended else "unknown")
    statuses = []
    for item in cells:
        receipt = v1._read_json(attempt / W.cell_receipt_name(item))
        found = receipt or {**item, "status": unfinished, "reason": f"exit_{code}", "synthetic_sha256": None}
        statuses.append({key: value for key, value in found.items() if key != "path"})
    return statuses


def attempt_number(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[1])


class Controller:
    def __init__(self, amendment: dict, out: Path, git_sha: str, yield_units: list[str], *,
                 runtime: dict | None = None, clock=time.time, boot=boot_id):
        self.amendment, self.out, self.git_sha, self.yield_units = amendment, out, git_sha, yield_units
        self.runtime = runtime if runtime is not None else json.loads(
            W.checked(amendment["runtime"]["path"], amendment["runtime"]["sha256"]))
        self.clock, self.boot = clock, boot
        self.close = datetime.datetime.fromisoformat(amendment["close_utc"].replace("Z", "+00:00")).timestamp()
        self.ticks = W.start_ticks(os.getpid())
        self.worker_sha = v1._sha256(WORKER)

    def bounded(self, cap: int) -> int:
        """An operation's seconds: its cap, cut at the campaign close."""
        return min(cap, int(self.close - self.clock()))

    def confirm(self, cause: str, attempt: Path) -> bool:
        """Establish a listed cause independently of the worker's own report."""
        if cause == "host_reboot":
            request = v1._read_json(attempt / "operation.request.json") or {}
            return bool(request.get("boot_id")) and request["boot_id"] != self.boot()
        if cause == "foreign_gpu_owner":
            # The worker has exited, so any compute process still on the GPU is foreign.
            return bool(v1._compute_pids())
        if cause == "storage_refusal":
            return shutil.disk_usage(self.out).free < LOW_DISK_BYTES
        if cause == "runtime_lock_mismatch":
            path = Path(self.amendment["runtime"]["path"])
            return not path.is_file() or v1._sha256(path) != self.amendment["runtime"]["sha256"]
        return False

    def request(self, mode: str, seed: int, row: dict, attempt: int, cap: int) -> dict:
        amendment = self.amendment
        return {"format": "dope-review-fix-tabsyn-seed-operation", "mode": mode, "fit_seed": seed,
                "attempt": attempt, "host": amendment["host"], "cpu_slot": amendment["cpu_slot"],
                "controller_pid": os.getpid(), "controller_start_ticks": self.ticks, "boot_id": self.boot(),
                "admitted_fit_seeds": amendment["admitted_fit_seeds"], "config": amendment["config"],
                "runtime": amendment["runtime"], "adapter": amendment["adapter"],
                "worker_code_sha256": self.worker_sha,
                "worker": {"path": str(Path(amendment["workers"]) / row["dataset"]), "files": row["files"]},
                "train_rows": row["train_rows"], "projection_sha256": row["projection_sha256"],
                "joint_train_sha256": row["joint_train_sha256"],
                "joint_validation_sha256": row["joint_validation_sha256"],
                "cap_seconds": cap, "close_utc": amendment["close_utc"], "operation_seconds": self.bounded(cap),
                "official_tests_opened": False}

    def operate(self, attempt: Path, request: dict) -> tuple[int, float, dict]:
        """Run one worker operation until the earlier of its cap and the close; then kill its process group."""
        attempt.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        request = {**request, "operation_started_monotonic": started,
                   "deadline_monotonic": started + request["operation_seconds"]}
        ref = W.put(attempt, "operation.request.json", request)
        argv = [self.runtime["python"], "-I", "-S", "-B", str(WORKER), str(attempt), ref["path"], ref["sha256"]]
        with (attempt / "worker.stdout.log").open("xb") as log, (attempt / "worker.stderr.log").open("xb") as err:
            process = subprocess.Popen(argv, env=child_env(self.runtime), stdout=log, stderr=err, cwd=attempt,
                                       start_new_session=True, close_fds=True)
            try:
                code = process.wait(timeout=request["operation_seconds"] + GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        return code, time.monotonic() - started, ref

    def fit_record(self, seed: int, row: dict, cell: Path, attempt: Path, code: int | None,
                   elapsed: float | None) -> dict:
        status, reason = outcome(attempt, "fit", code, self.confirm)
        request_path = attempt / "operation.request.json"
        ref = {"path": str(request_path), "sha256": v1._sha256(request_path)} if request_path.is_file() else None
        record = {"method": "TabSyn", "configuration": self.amendment["configuration"], "dataset": row["dataset"],
                  "fit_seed": seed, "status": status, "reason": reason, "attempts": attempt_number(attempt.parent),
                  "returncode": code, "elapsed_seconds": None if elapsed is None else round(elapsed, 3),
                  "request": ref, "attempt_dir": str(attempt), "gpu": gpu_name(), "official_tests_opened": False}
        if status == "ok":
            receipt = json.loads((attempt / "fit.complete.receipt.json").read_text())["result"]
            fitted, inventory = receipt["result"], receipt["inventory"]
            record.update(artifact=str(attempt / "artifact"), inventory=inventory,
                          artifact_bytes=inventory["artifact_bytes"], projection_bytes=inventory["projection_bytes"],
                          charged_bytes=charged_bytes(inventory),
                          **{key: fitted[key] for key in (
                              "vae_selected_epoch", "diffusion_selected_epoch", "completed_diffusion_epochs",
                              "diffusion_stopping_reason", "peak_allocated_vram_bytes", "whole_fit_elapsed_seconds")})
        v1._atomic_json(cell / FIT_RECORD, record)
        print(f"fit {status} {row['dataset']} {seed}", flush=True)
        return record

    def sample_record(self, seed: int, row: dict, cell: Path, attempt: Path, code: int | None,
                      elapsed: float | None) -> dict:
        cells = sample_cells(row, cell, self.amendment["sizes"], self.amendment["sample_seeds"])
        status, reason = outcome(attempt, "sample", code, self.confirm)
        request_path = attempt / "operation.request.json"
        record = {"dataset": row["dataset"], "fit_seed": seed, "attempts": attempt_number(attempt),
                  "returncode": code, "status": "infrastructure_failure" if status == "infrastructure_failure"
                  else "done", "reason": reason,
                  "elapsed_seconds": None if elapsed is None else round(elapsed, 3),
                  "request": {"path": str(request_path), "sha256": v1._sha256(request_path)}
                  if request_path.is_file() else None,
                  "cells": sample_status(attempt, cells, code), "official_tests_opened": False}
        v1._atomic_json(cell / SAMPLE_RECORD, record)
        print(f"sample {record['status']} {row['dataset']} {seed}", flush=True)
        return record

    def fit(self, seed: int, row: dict, cell: Path, previous: dict | None) -> dict:
        number = 1 + (previous["attempts"] if previous else 0)
        attempt = cell / f"attempt-{number}" / "fit"
        code, elapsed, _ref = self.operate(
            attempt, self.request("fit", seed, row, number, self.amendment["caps"]["fit_operation_seconds"]))
        return self.fit_record(seed, row, cell, attempt, code, elapsed)

    def sample(self, seed: int, row: dict, cell: Path, fit: dict, previous: dict | None) -> dict:
        number = 1 + (previous["attempts"] if previous else 0)
        attempt = cell / f"attempt-{fit['attempts']}" / f"sample-{number}"
        cells = sample_cells(row, cell, self.amendment["sizes"], self.amendment["sample_seeds"])
        request = {**self.request("sample", seed, row, number, self.amendment["caps"]["sample_operation_seconds"]),
                   "artifact_path": fit["artifact"], "artifact_inventory": fit["inventory"],
                   "fit_ref": fit["request"], "samples": cells}
        code, elapsed, _ref = self.operate(attempt, request)
        return self.sample_record(seed, row, cell, attempt, code, elapsed)

    def adopt(self, seed: int, row: dict, cell: Path, mode: str) -> dict | None:
        """Record the last attempt a stopped controller left without a record; no attempt key is reused."""
        pattern = "attempt-*/fit" if mode == "fit" else "attempt-*/sample-*"
        found = sorted(cell.glob(pattern), key=lambda path: (
            attempt_number(path.parent if mode == "fit" else path), str(path)))
        if not found:
            return None
        writer = self.fit_record if mode == "fit" else self.sample_record
        return writer(seed, row, cell, found[-1], None, None)

    def locked(self, gpu_lock: Path, operation):
        """Wait for the host, take the GPU lock, and start the operation only while the campaign is open."""
        if self.bounded(1) < 1:
            print("close reached; no new operation starts", flush=True)
            return None
        v1._wait_to_fit(self.yield_units)
        with gpu_lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.bounded(1) < 1:
                print("close reached while waiting; no new operation starts", flush=True)
                return None
            return operation()

    def run(self, plan: list[tuple[int, dict]], gpu_lock: Path) -> None:
        for seed, row in plan:
            cell = self.out / row["dataset"] / f"fit-{seed}"
            cell.mkdir(parents=True, exist_ok=True)
            fit = v1._read_json(cell / FIT_RECORD) or self.adopt(seed, row, cell, "fit")
            if needs_attempt(fit):
                fit = self.locked(gpu_lock, functools.partial(self.fit, seed, row, cell, fit))
                if fit is None:
                    return
            if fit["status"] != "ok":
                continue
            samples = v1._read_json(cell / SAMPLE_RECORD) or self.adopt(seed, row, cell, "sample")
            if needs_attempt(samples) and self.locked(
                    gpu_lock, functools.partial(self.sample, seed, row, cell, fit, samples)) is None:
                return


def ledger(plan: list[tuple[int, dict]], out: Path, amendment: dict) -> tuple[list[dict], list[dict]]:
    """Fit and sample ledgers over every planned cell; a missing or failed cell stays in the ledger."""
    fits, cells = [], []
    for seed, row in plan:
        cell = out / row["dataset"] / f"fit-{seed}"
        fit = v1._read_json(cell / FIT_RECORD) or {"status": "not_run", "reason": None, "attempts": 0}
        fits.append({"dataset": row["dataset"], "fit_seed": seed, "official_tests_opened": False,
                     **{key: fit.get(key) for key in (
                         "status", "reason", "attempts", "elapsed_seconds", "artifact_bytes", "charged_bytes",
                         "projection_bytes",
                         "vae_selected_epoch", "diffusion_selected_epoch", "completed_diffusion_epochs",
                         "diffusion_stopping_reason", "peak_allocated_vram_bytes", "gpu")}})
        done = {(item["seed"], item["size"]): item for item in (v1._read_json(cell / SAMPLE_RECORD) or {}).get("cells", [])}
        for size in amendment["sizes"]:
            for sample_seed in amendment["sample_seeds"]:
                item = done.get((sample_seed, size))
                status = item["status"] if item else ("not_run" if fit["status"] == "ok" else f"fit_{fit['status']}")
                csv_path = cell / f"sample-{sample_seed}-{size}n.csv"
                digest = item.get("synthetic_sha256") if item and status == "ok" else None
                if status == "ok" and (not csv_path.is_file() or v1._sha256(csv_path) != digest
                                       or v1._line_count(csv_path) != row["train_rows"] * size + 1):
                    raise SystemExit(f"sample file differs from its receipt at {csv_path}")
                cells.append({"dataset": row["dataset"], "fit_seed": seed, "sample_seed": sample_seed, "size": size,
                              "status": status, "reason": item.get("reason") if item else None,
                              "rows": row["train_rows"] * size, "synthetic_sha256": digest,
                              "official_tests_opened": False})
    return fits, cells


def write_ledger(fits: list[dict], cells: list[dict], results: Path, amendment: dict, git_sha: str) -> dict:
    results.mkdir(parents=True, exist_ok=True)
    texts = {name: "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in rows)
             for name, rows in (("fits.jsonl", fits), ("sample-cells.jsonl", cells))}
    for name, text in texts.items():
        (results / name).write_text(text)

    def counts(rows):
        return {status: sum(item["status"] == status for item in rows) for status in sorted({r["status"] for r in rows})}
    binding = {"format": "dope-review-fix-tabsyn-seeds-binding", "method": "TabSyn",
               "configuration": amendment["configuration"], "config_sha256": amendment["config_sha256"],
               "fit_seeds": amendment["fit_seeds"], "amendment_sha256": amendment["sha256"],
               "predeclare_sha256": amendment["predeclare_sha256"], "runner_sha256": v1._sha256(Path(__file__)),
               "worker_sha256": v1._sha256(WORKER), "adapter_sha256": amendment["adapter"]["sha256"],
               "runtime_sha256": amendment["runtime"]["sha256"], "plan_sha256": amendment["plan"]["sha256"],
               "source_git_sha": git_sha, "host": amendment["host"], "fits": counts(fits), "cells": counts(cells),
               "fits_sha256": hashlib.sha256(texts["fits.jsonl"].encode()).hexdigest(),
               "cells_sha256": hashlib.sha256(texts["sample-cells.jsonl"].encode()).hexdigest(),
               "medians_not_computed": True, "official_tests_opened": False}
    (results / "binding.json").write_text(json.dumps(binding, indent=1, sort_keys=True) + "\n")
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("plan", "run", "status", "ledger"))
    parser.add_argument("--git-sha", default="")
    parser.add_argument("--yield-unit", action="append", default=[])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, help="a copy of the output tree, for status and ledger only")
    parser.add_argument("--plan-file", type=Path, help="a copy of the seed-11 plan; its digest is still checked")
    args = parser.parse_args()
    amendment = load_amendment()
    if args.action in ("plan", "run") and (os.uname().nodename != amendment["host"] or os.getuid() != 1000):
        raise SystemExit(f"refusing to run on {os.uname().nodename}")
    if args.action == "run" and args.out is not None:
        raise SystemExit("run writes only to the amendment's output root")
    plan_doc = json.loads(W.checked(args.plan_file or amendment["plan"]["path"], amendment["plan"]["sha256"]))
    plan = seed_major(lineages(plan_doc, amendment["config_sha256"]), amendment["fit_seeds"])
    out = args.out or Path(amendment["output"])
    v1._refuse_test(out)
    if args.action == "plan":
        checks = [preflight(row, Path(amendment["workers"]), Path(amendment["scoring"]["workers"]))
                  for row in lineages(plan_doc, amendment["config_sha256"])]
        bad = [item for item in checks if item["status"] != "ok"]
        print(f"fits {len(plan)} lineages {len(checks)} inputs_ok {len(checks) - len(bad)} "
              f"unavailable {[item['dataset'] for item in bad]} official_tests_opened false", flush=True)
        return
    if args.action in ("status", "ledger"):
        fits, cells = ledger(plan, out, amendment)
        if args.action == "ledger":
            if len(args.git_sha) != 40:
                raise SystemExit("ledger needs the 40-character source commit of the run")
            print(json.dumps(write_ledger(fits, cells, args.results, amendment, args.git_sha), sort_keys=True))
            return
        for name, rows in (("fits", fits), ("cells", cells)):
            print(name, {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})})
        return
    if len(args.git_sha) != 40:
        raise SystemExit("run needs the 40-character source commit")
    if v1._sha256(Path(amendment["adapter"]["path"])) != amendment["adapter"]["sha256"]:
        raise SystemExit("captured adapter digest changed")
    os.sched_setaffinity(0, amendment["cpu_slot"])
    controller = Controller(amendment, out, args.git_sha, args.yield_unit)
    out.mkdir(parents=True, exist_ok=True)
    W.put(out, f"campaign-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}.json", {
        "format": "dope-review-fix-tabsyn-seeds-campaign", "fits_planned": len(plan),
        "amendment_sha256": amendment["sha256"], "predeclare_sha256": amendment["predeclare_sha256"],
        "runner_sha256": v1._sha256(Path(__file__)), "worker_sha256": controller.worker_sha,
        "source_git_sha": args.git_sha, "host": amendment["host"], "gpu": gpu_name(),
        "medians_not_computed": True, "official_tests_opened": False})
    controller.run(plan, Path(amendment["gpu_lock"]))


if __name__ == "__main__":
    main()
