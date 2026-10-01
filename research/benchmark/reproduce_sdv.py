"""Repeat a prespecified size/width-stratified subset without changing selections."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from research.benchmark.manifest import digest
from research.benchmark.score import artifact_inventory, sha256
from research.benchmark.sdv_round import ROOT, SCRATCH, WORKERS, check_lock, process, write_once
from research.benchmark.fetch_jope import LIMIT, used_bytes
from research.benchmark.inventory_hosts import local


AUDIT = ROOT / "reproduction"


def freeze() -> Path:
    round_path = ROOT / "round.lock.json"
    locked = json.loads(round_path.read_text())
    cohort = json.loads((SCRATCH / "s3-v1/gpu-cohort.lock.json").read_text())
    rows = cohort["discovery"] + cohort["confirmation"]
    narrow = min(rows, key=lambda r: (r["projected_features"], r["train_rows"], r["id"]))
    wide = max(rows, key=lambda r: (r["projected_features"], r["train_rows"], r["id"]))
    jobs = []
    for row in (narrow, wide):
        for method in ("CTGAN", "TVAE"):
            job = next(j for j in locked["jobs"] if j["dataset"] == row["id"] and j["method"] == method and j["trial"] == 0)
            original = ROOT / "jobs" / digest(job) / "attempt-0001/receipt.json"
            receipt = json.loads(original.read_text())
            jobs.append({"job": job, "host": receipt["host"], "original_path": str(original),
                         "original_sha256": sha256(original)})
    lock = {"format": "dope-sdv-reproduction-audit", "version": 1, "round_sha256": sha256(round_path),
            "audit_source_sha256": sha256(Path(__file__)), "jobs": jobs,
            "selection_rule": "Author defaults on the narrowest and widest frozen cohort datasets; row count and dataset hash break ties.",
            "same_host_fit": True, "exact_sample_hash_required": True, "native_kpi_absolute_tolerance": 1e-9,
            "fit_compute_ceiling_seconds": 4 * 600, "selection_unchanged": True}
    path = AUDIT / "audit.lock.json"
    if path.exists():
        if json.loads(path.read_text()) != lock:
            raise ValueError("reproduction audit changed")
    else:
        write_once(path, lock)
    return path


def run() -> None:
    check_lock(ROOT / "round.lock.json")
    lock_path = AUDIT / "audit.lock.json"
    lock = json.loads(lock_path.read_text())
    if sha256(Path(__file__)) != lock["audit_source_sha256"]:
        raise ValueError("reproduction source changed")
    host = os.uname().nodename.split(".")[0]
    with (ROOT / f"{host}.gpu-reservation").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in lock["jobs"]:
            if item["host"] != host:
                continue
            job = item["job"]
            root = AUDIT / digest(job)
            if (root / "receipt.json").exists():
                continue
            snapshot = local()
            if (snapshot["active_gpu_processes"] or snapshot["gpus"][0]["memory_free_mib"] < 17 * 1024
                    or len(snapshot["allowed_cpus"]) < 16 or snapshot["memory"]["MemAvailable"] < 24 * 1024**3
                    or snapshot["scratch_disk"]["free_bytes"] < 5_000_000_000
                    or snapshot["load_average"][0] + 16 > 1.25 * len(snapshot["allowed_cpus"])
                    or used_bytes(SCRATCH) + 1_000_000_000 > LIMIT):
                raise ValueError("reproduction admission blocked")
            if sha256(Path(item["original_path"])) != item["original_sha256"]:
                raise ValueError("original audit target changed")
            original = json.loads(Path(item["original_path"]).read_text())
            worker = WORKERS / job["dataset"]
            for part in ("train", "validation"):
                if sha256(worker / f"{part}.csv") != job[f"{part}_sha256"]:
                    raise ValueError("reproduction partition changed")
            if sha256(worker / "projection.json") != job["projection_sha256"]:
                raise ValueError("reproduction projection changed")
            root.mkdir(parents=True)
            write_once(root / "host.json", snapshot)
            os.sched_setaffinity(0, snapshot["allowed_cpus"][:16])
            fit = process({"action": "fit", "method": job["method"], "config": job["config"],
                           "train": str(worker / "train.csv"), "seed": job["fit_seed"],
                           "artifact": str(root / "artifact")}, root / "fit", 600, gpu=True)
            (root / "artifact/projection.json").write_bytes((worker / "projection.json").read_bytes())
            sample = process({"action": "sample", "artifact": str(root / "artifact"),
                              "rows": job["train_rows"], "seed": job["sample_seed"],
                              "output": str(root / "sample.csv")}, root / "sample", 600)
            native = process({"action": "efficacy", "synthetic": str(root / "sample.csv"),
                              "validation": str(worker / "validation.csv")}, root / "native", 600)
            inventory, size = artifact_inventory(root / "artifact", ["model.pt", "model.json", "projection.json"])
            delta = abs(native["child"]["value"] - original["native_kpi"]["value"])
            exact = sample["child"]["sha256"] == original["sample"]["child"]["sha256"]
            receipt = {"format": "dope-sdv-reproduction-receipt", "version": 1,
                       "job": job, "host": host, "audit_lock_sha256": sha256(lock_path),
                       "original_sha256": item["original_sha256"], "fit_seconds": fit["wall_seconds"],
                       "artifact_bytes": size, "artifact_inventory": inventory,
                       "exact_artifact_hashes": inventory == original["artifact_inventory"],
                       "exact_sample_hash": exact, "sample_sha256": sample["child"]["sha256"],
                       "native_kpi_absolute_difference": delta,
                       "passed": exact and delta <= lock["native_kpi_absolute_tolerance"],
                       "evidence_files": {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}}
            write_once(root / "receipt.json", receipt)
            print(json.dumps({"dataset": job["dataset"], "method": job["method"], "passed": receipt["passed"]}), flush=True)
            os.sched_setaffinity(0, snapshot["allowed_cpus"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    print(freeze()) if args.freeze else run()
