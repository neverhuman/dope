"""Lossless projection packing for a bounded, validation-only L3 refinement."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .manifest import digest
from .score import artifact_inventory, sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
CAP = 10_240


def packed_projection(projection: dict) -> bytes:
    features = projection.get("features")
    if not isinstance(features, list) or not features or not all(
            isinstance(feature, dict) and isinstance(feature.get("name"), str)
            for feature in features):
        raise ValueError("projection features are invalid")
    names = [feature["name"] for feature in features] + [projection.get("target")]
    if (not isinstance(names[-1], str) or len(names) != len(set(names))
            or projection.get("input_columns") != names):
        raise ValueError("input columns are not reconstructible")
    compact = {key: value for key, value in projection.items() if key != "input_columns"}
    encoded = (json.dumps(compact, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False) + "\n").encode()
    if unpacked_projection(encoded) != projection:
        raise ValueError("projection packing is not lossless")
    return encoded


def unpacked_projection(encoded: bytes) -> dict:
    compact = json.loads(encoded)
    if not isinstance(compact, dict) or "input_columns" in compact:
        raise ValueError("invalid packed projection")
    features = compact.get("features")
    if not isinstance(features, list) or not all(
            isinstance(feature, dict) and isinstance(feature.get("name"), str)
            for feature in features):
        raise ValueError("invalid packed features")
    target = compact.get("target")
    if not isinstance(target, str):
        raise ValueError("invalid packed target")
    compact["input_columns"] = [feature["name"] for feature in features] + [target]
    return compact


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def freeze(root: Path = ROOT) -> Path:
    original_round = root / "dope-gpu-round-v3.lock.json"
    round_data = json.loads(original_round.read_text())
    jobs = []
    for receipt_path in sorted((root / "gpu-v3/fits").glob("*/attempt-0001.json")):
        receipt = json.loads(receipt_path.read_text())
        identity = receipt["identity"]
        dataset = identity["dataset"]
        if (receipt["status"] != "ok" or
                identity["round_lock_sha256"] != sha256(original_round)):
            raise ValueError("GPU fit differs from original round")
        worker = root / "prepared-v2/worker" / dataset
        projection = worker / "projection.json"
        model = receipt_path.with_suffix(".dpk")
        if (sha256(projection) != identity["projection_sha256"]
                or sha256(model) != receipt["artifact_sha256"]
                or model.stat().st_size + projection.stat().st_size != receipt["artifact_bytes"]):
            raise ValueError("GPU artifact changed")
        jobs.append({"dataset": dataset, "fit_seed": identity["seed"],
                     "fit_receipt_path": str(receipt_path),
                     "fit_receipt_sha256": sha256(receipt_path),
                     "projection_sha256": sha256(projection),
                     "model_sha256": sha256(model)})
    if len(jobs) != 6 or {(job["dataset"], job["fit_seed"]) for job in jobs} != {
            (dataset, seed) for dataset in ("Adult", "California", "News")
            for seed in (11, 23)}:
        raise ValueError("refinement requires the six frozen pilot fits")
    jobs.sort(key=lambda job: (job["dataset"], job["fit_seed"]))
    lock = {"format": "dope-projection-refinement-round", "version": 1,
            "validation_only": True, "original_gpu_round_sha256": sha256(original_round),
            "source_sha256": sha256(Path(__file__)),
            "gpu_binary_path": round_data["gpu_binary_path"],
            "gpu_binary_sha256": round_data["gpu_binary_sha256"],
            "candidate_grid": [{"codec": "canonical_json_without_redundant_input_columns"}],
            "compute_ceiling_seconds": 360, "per_artifact_sample_timeout_seconds": 60,
            "jobs": jobs, "l3_byte_cap": CAP, "official_tests_opened": False,
            "ptf_v1": None, "mfs_v2": None, "production_certified": False}
    path = root / "projection-refine-v1/round.lock.json"
    write_once(path, lock)
    return path


def run(lock_path: Path) -> list[dict]:
    lock = json.loads(lock_path.read_text())
    if (lock.get("format") != "dope-projection-refinement-round"
            or lock.get("source_sha256") != sha256(Path(__file__))
            or lock.get("candidate_grid") != [
                {"codec": "canonical_json_without_redundant_input_columns"}]
            or lock.get("l3_byte_cap") != CAP):
        raise ValueError("refinement lock changed")
    root = lock_path.parents[1]
    if lock["original_gpu_round_sha256"] != sha256(root / "dope-gpu-round-v3.lock.json"):
        raise ValueError("original GPU round changed")
    binary = Path(lock["gpu_binary_path"])
    if sha256(binary) != lock["gpu_binary_sha256"]:
        raise ValueError("GPU binary changed")
    output = []
    started = time.monotonic()
    for job in lock["jobs"]:
        receipt_path = Path(job["fit_receipt_path"])
        if sha256(receipt_path) != job["fit_receipt_sha256"]:
            raise ValueError("fit receipt changed")
        fit = json.loads(receipt_path.read_text())
        model = receipt_path.with_suffix(".dpk")
        projection = root / "prepared-v2/worker" / job["dataset"] / "projection.json"
        if (fit["status"] != "ok" or fit["identity"]["seed"] != job["fit_seed"]
                or fit["identity"]["dataset"] != job["dataset"]
                or sha256(model) != job["model_sha256"]
                or sha256(projection) != job["projection_sha256"]):
            raise ValueError("fit lineage changed")
        cell = lock_path.parent / "cells" / digest(job)
        receipt = cell / "receipt.json"
        artifact = cell / "artifact"
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            inventory, charged = artifact_inventory(
                artifact, ["model.dpk", "projection.compact.json"])
            if (prior["round_sha256"] != sha256(lock_path)
                    or prior["artifact_inventory"] != inventory
                    or prior["artifact_bytes"] != charged):
                raise ValueError("prior refinement changed")
            output.append(prior)
            continue
        artifact.mkdir(parents=True, exist_ok=False)
        with model.open("rb") as source, (artifact / "model.dpk").open("xb") as target:
            shutil.copyfileobj(source, target)
        packed = packed_projection(json.loads(projection.read_text()))
        (artifact / "projection.compact.json").write_bytes(packed)
        if (sha256(artifact / "model.dpk") != job["model_sha256"]
                or unpacked_projection(packed) != json.loads(projection.read_text())):
            raise ValueError("packed artifact differs")
        inventory, charged = artifact_inventory(
            artifact, ["model.dpk", "projection.compact.json"])
        reference_path = (root / "gpu-v3/validation-samples" / job["dataset"]
                          / f"fit-{job['fit_seed']}/sample-101-size-1/sample-receipt.json")
        reference = json.loads(reference_path.read_text())
        reference_sample = reference_path.parent / "sample.csv"
        if (reference["fit_receipt_sha256"] != job["fit_receipt_sha256"]
                or reference["sample_sha256"] != sha256(reference_sample)):
            raise ValueError("original sample changed")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        env["LD_LIBRARY_PATH"] = (str(Path(torch.__file__).parent / "lib") + ":"
                                  + env.get("LD_LIBRARY_PATH", ""))
        remaining = lock["compute_ceiling_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("refinement compute ceiling reached")
        command = [str(binary), "sample", "--kernel", str(artifact / "model.dpk"),
                   "--rows", str(reference["row_count"]), "--seed", "101", "--out",
                   str(cell / "repeated-sample.csv")]
        try:
            subprocess.run(command, capture_output=True, check=True,
                           timeout=min(lock["per_artifact_sample_timeout_seconds"], remaining),
                           env=env)
            repeated_sha = sha256(cell / "repeated-sample.csv")
        finally:
            (cell / "repeated-sample.csv").unlink(missing_ok=True)
        if repeated_sha != reference["sample_sha256"]:
            raise ValueError("packed artifact sampling changed")
        result = {"format": "dope-projection-refinement-receipt", "version": 1,
                  "round_sha256": sha256(lock_path), "job": job,
                  "original_fit_receipt_sha256": job["fit_receipt_sha256"],
                  "original_projection_sha256": job["projection_sha256"],
                  "packed_projection_sha256": sha256(artifact / "projection.compact.json"),
                  "artifact_inventory": inventory, "artifact_bytes": charged,
                  "reference_sample_receipt_sha256": sha256(reference_path),
                  "repeated_sample_sha256": repeated_sha,
                  "status": "l3_byte_eligible" if charged <= CAP else "byte_cap_failure",
                  "restricted_projection_map": True,
                  "validation_only": True, "official_tests_opened": False,
                  "ptf_v1": None, "mfs_v2": None, "production_certified": False}
        write_once(receipt, result)
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    path = args.root / "projection-refine-v1/round.lock.json"
    if args.action == "freeze":
        print(json.dumps({"round_sha256": sha256(freeze(args.root))}, sort_keys=True))
    else:
        print(json.dumps([{"dataset": item["job"]["dataset"],
                           "fit_seed": item["job"]["fit_seed"],
                           "artifact_bytes": item["artifact_bytes"],
                           "status": item["status"]} for item in run(path)], sort_keys=True))


if __name__ == "__main__":
    main()
