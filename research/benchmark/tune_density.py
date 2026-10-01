"""Bounded native-likelihood selection for locked compact reference methods."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from . import adapters
from .fetch_jope import LIMIT, used_bytes
from .manifest import digest
from .score import artifact_inventory, sha256


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
METHODS = ("independent_marginals", "Chow-Liu")
BUDGET_SECONDS = 43_200


def cpu_affinity() -> tuple[int, ...]:
    """Choose one bounded slot from CPUs actually assigned to this process."""
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise ValueError("native tuning has no admitted CPU")
    return tuple(allowed[16:32] if len(allowed) >= 32 else allowed[:16])


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError("existing native tuning receipt differs")
    else:
        with path.open("x") as stream:
            stream.write(encoded)


def candidates(entry: dict) -> list[dict]:
    default = entry["default_config"]
    search = entry["tuning_search_space"]
    keys = sorted(search)
    configs = []
    for values in itertools.product(*(search[key] for key in keys)):
        config = dict(default)
        config.update(zip(keys, values))
        configs.append(config)
    if not 1 <= len(configs) <= 8:
        raise ValueError("native search exceeds eight tuning trials")
    return configs


def tune(worker: Path, method: str, methods_lock: Path, round_lock_path: Path,
         output_root: Path) -> dict:
    if method not in METHODS or not output_root.resolve().is_relative_to(SCRATCH.resolve()):
        raise ValueError("method or scratch root is outside native tuning study")
    methods = json.loads(methods_lock.read_text())
    entry = methods["methods"][method]
    if (entry["status"] != "locked" or entry["adapter_sha256"] != sha256(Path(adapters.__file__))
            or entry["native_objective"]["status"] != "locked"):
        raise ValueError("method or native objective is not source locked")
    import numpy as np
    if entry["dependency_or_container_digest"] != f"numpy=={np.__version__}":
        raise ValueError("native adapter dependency changed")
    manifest = json.loads((worker / "worker-manifest.json").read_text())
    round_lock = json.loads(round_lock_path.read_text())
    dataset = manifest["dataset_id"]
    jobs = [job for job in round_lock.get("jobs", [])
            if job.get("dataset") == dataset and job.get("method") == method]
    if (round_lock.get("format") != "dope-benchmark-compact-native-round"
            or round_lock.get("method_lock_sha256") != sha256(methods_lock)
            or round_lock.get("adapter_sha256") != sha256(Path(adapters.__file__))
            or round_lock.get("tuner_sha256") != sha256(Path(__file__))
            or len(jobs) != 1):
        raise ValueError("compact native round is not frozen for this job")
    job = jobs[0]
    if (job["configurations"] != candidates(entry) or job["fit_seed"] != 11
            or job["budget_seconds"] != BUDGET_SECONDS
            or job["train_sha256"] != manifest["projected_hashes"]["train"]
            or job["validation_sha256"] != manifest["projected_hashes"]["validation"]
            or job["projection_sha256"] != manifest["projection_sha256"]):
        raise ValueError("compact native job differs from frozen round")
    for name in ("train", "validation"):
        if sha256(worker / f"{name}.csv") != manifest["projected_hashes"][name]:
            raise ValueError("worker partition digest mismatch")
    if sha256(worker / "projection.json") != manifest["projection_sha256"]:
        raise ValueError("worker projection digest mismatch")
    if used_bytes(SCRATCH) + 1_000_000_000 > LIMIT:
        raise ValueError("benchmark scratch ceiling reached")
    objective = entry["native_objective"]
    affinity = cpu_affinity()
    affinity_arg = ",".join(map(str, affinity))
    trial_configs = candidates(entry)
    round_sha256 = sha256(round_lock_path)
    root = output_root / dataset / method
    root.mkdir(parents=True, exist_ok=True)
    trials = []
    total_wall = 0.0
    for index, config in enumerate(trial_configs):
        trial_root = root / f"trial-{index:02d}"
        artifact = trial_root / "artifact"
        attempt_path = trial_root / "attempt.json"
        metric_path = trial_root / "native-metric.json"
        if attempt_path.exists():
            trial = json.loads(attempt_path.read_text())
            if (trial["config"] != config or trial["identity"]["round_sha256"] != round_sha256
                    or not math.isfinite(trial["wall_seconds"])):
                raise ValueError("native tuning trial changed")
            if trial["status"] == "ok":
                inventory, artifact_bytes = artifact_inventory(
                    artifact, ["model.json", "projection.json"])
                if (sha256(metric_path) != trial["metric_receipt_sha256"]
                        or sha256(artifact / "model.json") != trial["artifact_sha256"]
                        or inventory != trial["artifact_inventory"]
                        or artifact_bytes != trial["artifact_bytes"]):
                    raise ValueError("resumed native tuning receipt changed")
        else:
            trial_root.mkdir(parents=True, exist_ok=True)
            artifact.mkdir(exist_ok=True)
            started = time.monotonic()
            try:
                if total_wall >= BUDGET_SECONDS:
                    raise TimeoutError("native tuning budget exhausted")
                request = {"action": "fit", "method": method,
                           "train": str(worker / "train.csv"),
                           "metadata": json.loads((worker / "projection.json").read_text()),
                           "config": config, "seed": 11, "artifact_dir": str(artifact),
                           "binary": None}
                fit = subprocess.run(["taskset", "-c", affinity_arg, sys.executable, "-m",
                                      "research.benchmark.adapter_worker"],
                                     input=json.dumps(request), capture_output=True, text=True,
                                     timeout=min(config["fit_timeout_seconds"],
                                                 BUDGET_SECONDS - total_wall), check=True)
                response = json.loads(fit.stdout.strip().splitlines()[-1])
                if response.get("status") != "ok" or response.get("files") != ["model.json"]:
                    raise ValueError("native adapter fit failed")
                (artifact / "projection.json").write_bytes((worker / "projection.json").read_bytes())
                inventory, artifact_bytes = artifact_inventory(artifact,
                                                              ["model.json", "projection.json"])
                remaining = BUDGET_SECONDS - total_wall - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("native tuning budget exhausted")
                metric = subprocess.run(["taskset", "-c", affinity_arg, sys.executable, "-m",
                                         "research.benchmark.native_objective", method,
                                         str(artifact), str(worker / "validation.csv")],
                                        capture_output=True, text=True,
                                        timeout=remaining, check=True)
                measured = json.loads(metric.stdout.strip().splitlines()[-1])
                if (measured["implementation_sha256"] != objective["implementation_sha256"]
                        or measured["method"] != method):
                    raise ValueError("native KPI implementation changed")
                write_once(metric_path, measured)
                trial = {"status": "ok", "config": config,
                         "native_kpi": measured["value"],
                         "metric_receipt_path": str(metric_path),
                         "metric_receipt_sha256": sha256(metric_path),
                         "artifact_sha256": sha256(artifact / "model.json"),
                         "artifact_bytes": artifact_bytes,
                         "artifact_inventory": inventory}
            except (OSError, ValueError, subprocess.SubprocessError, TimeoutError,
                    json.JSONDecodeError) as error:
                trial = {"status": "timeout" if isinstance(error, (TimeoutError,
                         subprocess.TimeoutExpired)) else "failed", "config": config,
                         "error_type": type(error).__name__}
            trial["wall_seconds"] = time.monotonic() - started
            trial["cpu_affinity"] = affinity
            trial["identity"] = {"dataset": dataset, "method": method,
                                 "round_sha256": round_sha256,
                                 "method_source_sha256": entry["source_sha256"],
                                 "adapter_sha256": entry["adapter_sha256"],
                                 "native_objective_sha256": objective["implementation_sha256"],
                                 "train_sha256": manifest["projected_hashes"]["train"],
                                 "validation_sha256": manifest["projected_hashes"]["validation"],
                                 "config": config, "fit_seed": 11}
            write_once(attempt_path, trial)
        total_wall += trial["wall_seconds"]
        trials.append(trial)
    successful = [(index, row) for index, row in enumerate(trials) if row["status"] == "ok"]
    if not successful or total_wall > BUDGET_SECONDS:
        errors = sorted({row.get("error_type", "budget_exhausted") for row in trials
                         if row["status"] != "ok"})
        raise ValueError("native tuning has no eligible selection: " + ",".join(errors))
    sign = -1 if objective["direction"] == "maximize" else 1
    selected_index, selected = min(successful,
        key=lambda pair: (sign * pair[1]["native_kpi"], pair[1]["artifact_bytes"],
                          digest(pair[1]["config"]), pair[0]))
    selection = {"format": "dope-benchmark-validation-selection", "version": 1,
                 "dataset": dataset, "method": method, "partition": "validation",
                 "round_sha256": round_sha256, "stage": job["stage"],
                 "objective": objective, "selected_trial_index": selected_index,
                 "selected_config": selected["config"], "trials": trials,
                 "total_wall_seconds": total_wall, "test_opened": False}
    selection_path = root / "selection.json"
    write_once(selection_path, selection)
    return {"dataset": dataset, "method": method, "selected_trial_index": selected_index,
            "selected_native_kpi": selected["native_kpi"],
            "selection_path": str(selection_path),
            "selection_sha256": sha256(selection_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--methods-lock", type=Path, default=Path(__file__).with_name("methods.lock.json"))
    parser.add_argument("--round-lock", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=SCRATCH / "compact-native-v1")
    args = parser.parse_args()
    print(json.dumps(tune(args.worker, args.method, args.methods_lock, args.round_lock,
                          args.output_root),
                     sort_keys=True))


if __name__ == "__main__":
    main()
