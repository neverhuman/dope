"""Losslessly pack the frozen News q10 GPU artifact and replay its samples."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .manifest import digest
from .refine_projection import packed_projection, unpacked_projection
from .score import artifact_inventory, sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
SOURCE = ROOT / "dope-news-q10-v1"
OUT = ROOT / "dope-news-q10-packed-v1"
LOCK = OUT / "round.lock.json"
CAP = 10_240
SLOT = list(range(0, 16))
ARTIFACT_FILES = ["model.dpk", "projection.compact.json"]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def freeze() -> Path:
    parent_path = SOURCE / "round.lock.json"
    sample_lock_path = SOURCE / "sample.lock.json"
    parent = read(parent_path)
    sample_lock = read(sample_lock_path)
    fit_path = SOURCE / "fit.json"
    fit = read(fit_path)
    receipt_path = Path(fit["fit_receipt_path"])
    receipt = read(receipt_path)
    model = receipt_path.with_suffix(".dpk")
    worker = ROOT / "prepared-v2/worker/News"
    projection_path = worker / "projection.json"
    binary = Path("/mnt/fast-scratch/dope-benchmark/conditional-readout-v1/bin/dope-gpu-132f8467f56a79ec")
    if (parent["format"] != "dope-pilot-overcap-news-q10-confirmation-fit"
            or parent["source_sha256"] != sha256(SOURCE / "fit.py")
            or sample_lock["source_sha256"] != sha256(SOURCE / "sample.py")
            or fit["round_sha256"] != sha256(parent_path)
            or fit["status"] != "ok"
            or fit["fit_receipt_sha256"] != sha256(receipt_path)
            or fit["artifact_bytes"] != receipt["artifact_bytes"]
            or receipt["artifact_sha256"] != sha256(model)
            or receipt["artifact_bytes"] != model.stat().st_size + projection_path.stat().st_size
            or parent["projection_sha256"] != sha256(projection_path)
            or parent["gpu_binary_sha256"] != sha256(binary)
            or sample_lock["binary_sha256"] != sha256(binary)
            or sample_lock["official_tests_opened"] is not False
            or parent["official_tests_opened"] is not False
            or (worker / "test.csv").exists()
            or sha256(worker / "train.csv") != parent["train_sha256"]
            or sha256(worker / "validation.csv") != parent["validation_sha256"]):
        raise ValueError("News q10 frozen GPU fit changed")
    cells = []
    for job in sample_lock["jobs"]:
        if job["fit_seed"] != 23:
            continue
        sample_receipt = SOURCE / "samples" / digest(job) / "receipt.json"
        result = read(sample_receipt)
        sample_path = Path(result["sample_path"])
        if (result["job"] != job or result["status"] != "ok"
                or result["round_sha256"] != sha256(sample_lock_path)
                or result["sample_sha256"] != sha256(sample_path)
                or job["fit_receipt_sha256"] != sha256(receipt_path)
                or job["model_sha256"] != sha256(model)
                or job["projection_sha256"] != sha256(projection_path)
                or job["train_sha256"] != parent["train_sha256"]
                or job["validation_sha256"] != parent["validation_sha256"]
                or result["official_tests_opened"] is not False):
            raise ValueError("News q10 original sample changed")
        cells.append({"job": job, "sample_receipt_path": str(sample_receipt),
                      "sample_receipt_sha256": sha256(sample_receipt),
                      "sample_sha256": sha256(sample_path)})
    cells.sort(key=lambda cell: (cell["job"]["sample_seed"],
                                 cell["job"]["size_multiplier"]))
    if (len(cells) != 12
            or {(cell["job"]["sample_seed"], cell["job"]["size_multiplier"])
                for cell in cells}
            != {(seed, size) for seed in (101, 211, 307)
                for size in (1, 2, 4, 8)}):
        raise ValueError("News q10 twelve-sample matrix incomplete")
    lock = {"format": "dope-news-q10-lossless-projection-refinement",
            "version": 1, "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "codec_source_sha256": sha256(Path(__file__).with_name("refine_projection.py")),
            "parent_round_sha256": sha256(parent_path),
            "parent_sample_round_sha256": sha256(sample_lock_path),
            "fit_wrapper_sha256": sha256(fit_path),
            "fit_receipt_sha256": sha256(receipt_path),
            "model_sha256": sha256(model),
            "projection_sha256": sha256(projection_path),
            "gpu_binary_path": str(binary), "gpu_binary_sha256": sha256(binary),
            "candidate_grid": ["canonical_json_without_redundant_input_columns"],
            "compute_ceiling_seconds": 900, "per_sample_timeout_seconds": 60,
            "cpu_slot": SLOT, "l3_byte_cap": CAP,
            "sample_cells": cells, "official_tests_opened": False,
            "mfs_v2": None, "ptf_v1": None, "production_certified": False}
    write_once(LOCK, lock)
    return LOCK


def run() -> dict:
    lock = read(LOCK)
    if (lock["format"] != "dope-news-q10-lossless-projection-refinement"
            or lock["source_sha256"] != sha256(Path(__file__))
            or lock["codec_source_sha256"] != sha256(
                Path(__file__).with_name("refine_projection.py"))
            or lock["parent_round_sha256"] != sha256(SOURCE / "round.lock.json")
            or lock["parent_sample_round_sha256"] != sha256(SOURCE / "sample.lock.json")
            or lock["gpu_binary_sha256"] != sha256(Path(lock["gpu_binary_path"]))
            or lock["candidate_grid"]
            != ["canonical_json_without_redundant_input_columns"]
            or lock["cpu_slot"] != SLOT or lock["l3_byte_cap"] != CAP
            or lock["official_tests_opened"] is not False
            or len(lock["sample_cells"]) != 12
            or not set(SLOT).issubset(os.sched_getaffinity(0))):
        raise ValueError("News q10 projection refinement lock changed")
    worker = ROOT / "prepared-v2/worker/News"
    if (worker / "test.csv").exists():
        raise ValueError("sealed News partition in worker")
    fit = read(SOURCE / "fit.json")
    fit_receipt = Path(fit["fit_receipt_path"])
    model = fit_receipt.with_suffix(".dpk")
    projection = worker / "projection.json"
    if (sha256(fit_receipt) != lock["fit_receipt_sha256"]
            or sha256(model) != lock["model_sha256"]
            or sha256(projection) != lock["projection_sha256"]):
        raise ValueError("News q10 fit or projection changed")
    artifact = OUT / "artifact"
    result_path = OUT / "receipt.json"
    if not artifact.exists():
        artifact.mkdir(parents=True)
        shutil.copyfile(model, artifact / "model.dpk")
        packed = packed_projection(read(projection))
        (artifact / "projection.compact.json").write_bytes(packed)
    encoded = (artifact / "projection.compact.json").read_bytes()
    if (sha256(artifact / "model.dpk") != lock["model_sha256"]
            or unpacked_projection(encoded) != read(projection)):
        raise ValueError("News q10 packed artifact is not lossless")
    inventory, charged = artifact_inventory(artifact, ARTIFACT_FILES)
    started = time.monotonic()
    cells = []
    binary = Path(lock["gpu_binary_path"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    env["LD_LIBRARY_PATH"] = (str(Path(torch.__file__).parent / "lib") + ":"
                              + env.get("LD_LIBRARY_PATH", ""))
    for cell in lock["sample_cells"]:
        job = cell["job"]
        reference_path = Path(cell["sample_receipt_path"])
        if sha256(reference_path) != cell["sample_receipt_sha256"]:
            raise ValueError("News q10 sample receipt changed")
        reference = read(reference_path)
        if sha256(Path(reference["sample_path"])) != cell["sample_sha256"]:
            raise ValueError("News q10 sample changed")
        path = OUT / "sample-checks" / f"{digest(cell)}.json"
        if path.exists():
            check = read(path)
            if (check["round_sha256"] != sha256(LOCK)
                    or check["cell"] != cell
                    or check["sample_sha256"] != cell["sample_sha256"]
                    or check["status"] != "exact"
                    or check["stdout_sha256"] != sha256(path.with_suffix(".stdout"))
                    or check["stderr_sha256"] != sha256(path.with_suffix(".stderr"))):
                raise ValueError("News q10 prior sample check changed")
        else:
            remaining = lock["compute_ceiling_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("News q10 refinement compute ceiling reached")
            output = path.with_suffix(".csv")
            command = [str(binary), "sample", "--kernel", str(artifact / "model.dpk"),
                       "--rows", str(job["row_count"]), "--seed",
                       str(job["sample_seed"]), "--out", str(output)]
            cell_started = time.monotonic()
            try:
                completed = subprocess.run(command, capture_output=True,
                                           timeout=min(lock["per_sample_timeout_seconds"],
                                                       remaining), env=env, check=False)
                stdout, stderr = completed.stdout, completed.stderr
                status = ("exact" if completed.returncode == 0 and output.is_file()
                          and sha256(output) == cell["sample_sha256"] else "failed")
                exit_code = completed.returncode
            except subprocess.TimeoutExpired as error:
                stdout, stderr = error.stdout or b"", error.stderr or b""
                status, exit_code = "timeout", None
            path.with_suffix(".stdout").write_bytes(stdout)
            path.with_suffix(".stderr").write_bytes(stderr)
            actual = sha256(output) if output.is_file() else None
            if status == "exact":
                output.unlink()
            check = {"format": "dope-news-q10-packed-sample-check", "version": 1,
                     "round_sha256": sha256(LOCK), "cell": cell,
                     "sample_sha256": actual, "exit_code": exit_code,
                     "elapsed_seconds": time.monotonic() - cell_started,
                     "stdout_sha256": sha256(path.with_suffix(".stdout")),
                     "stderr_sha256": sha256(path.with_suffix(".stderr")),
                     "status": status, "official_tests_opened": False,
                     "mfs_v2": None, "ptf_v1": None}
            write_once(path, check)
            if status != "exact":
                raise ValueError("News q10 packed sample contract failed")
        cells.append({"sample_seed": job["sample_seed"],
                      "size_multiplier": job["size_multiplier"],
                      "sample_check_sha256": sha256(path),
                      "reference_sample_sha256": cell["sample_sha256"]})
        print(json.dumps({"sample_seed": job["sample_seed"],
                          "size_multiplier": job["size_multiplier"],
                          "status": "exact"}), flush=True)
    report = {"format": "dope-news-q10-packed-projection-receipt", "version": 1,
              "round_lock_sha256": sha256(LOCK),
              "codec_source_sha256": lock["codec_source_sha256"],
              "source_sha256": lock["source_sha256"],
              "artifact_inventory": inventory, "artifact_bytes": charged,
              "within_l3_bytes": charged <= CAP,
              "sample_checks": cells, "sample_checks_exact": len(cells),
              "wall_seconds": time.monotonic() - started,
              "restricted_projection_map": True,
              "official_tests_opened": False, "mfs_v2": None,
              "ptf_v1": None, "production_certified": False}
    if result_path.exists():
        prior = read(result_path)
        report["wall_seconds"] = prior["wall_seconds"]
        if prior != report:
            raise ValueError("News q10 packed receipt changed")
    else:
        write_once(result_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze())
    else:
        value = run()
        print(json.dumps({"artifact_bytes": value["artifact_bytes"],
                          "within_l3_bytes": value["within_l3_bytes"],
                          "sample_checks_exact": value["sample_checks_exact"]}))
