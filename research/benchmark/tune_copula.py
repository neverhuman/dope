"""Frozen, bounded GaussianCopula native-likelihood validation wave."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from . import adapters, tune_density
from .fetch_jope import LIMIT, used_bytes
from .manifest import digest
from .score import artifact_inventory, sha256


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
ROOT = SCRATCH / "copula-native-v1"
COHORT = SCRATCH / "s3-v1/gpu-cohort.lock.json"
WORKERS = SCRATCH / "s3-v1/prepared/worker"
METHODS = Path(__file__).with_name("methods.lock.json")
PYTHON = Path(__file__).resolve().parents[2] / "target/bench-venv/bin/python"
BUDGET_SECONDS = 43_200


def environment(python: Path) -> dict:
    script = ("import copulas,numpy,pandas,scipy,json; "
              "print(json.dumps({'copulas':copulas.__version__,'numpy':numpy.__version__,"
              "'pandas':pandas.__version__,'scipy':scipy.__version__}))")
    result = subprocess.run([str(python), "-c", script], capture_output=True,
                            text=True, check=True, timeout=30)
    return json.loads(result.stdout)


def freeze(cohort_path: Path = COHORT, workers: Path = WORKERS,
           methods_path: Path = METHODS, python: Path = PYTHON) -> dict:
    cohort = json.loads(cohort_path.read_text())
    entry = json.loads(methods_path.read_text())["methods"]["GaussianCopula"]
    versions = environment(python)
    if (cohort.get("validation_only") is not True or entry["status"] != "locked"
            or entry["adapter_sha256"] != sha256(Path(adapters.__file__))
            or entry["native_objective"]["implementation_sha256"]
            != sha256(Path(__file__).with_name("native_objective.py"))
            or versions != {"copulas": entry["dependency_version"].split("==")[1],
                            **entry["dependency_versions"]}):
        raise ValueError("copula source, objective, data, or environment changed")
    jobs = []
    for stage in ("discovery", "confirmation"):
        for dataset in cohort[stage]:
            worker = workers / dataset["id"]
            manifest = json.loads((worker / "worker-manifest.json").read_text())
            if (manifest["dataset_id"] != dataset["id"]
                    or manifest["projected_hashes"]["train"] != dataset["train_sha256"]
                    or manifest["projected_hashes"]["validation"] != dataset["validation_sha256"]):
                raise ValueError("worker differs from cohort")
            for name in ("train", "validation"):
                if sha256(worker / f"{name}.csv") != manifest["projected_hashes"][name]:
                    raise ValueError("worker partition changed")
            if sha256(worker / "projection.json") != manifest["projection_sha256"]:
                raise ValueError("worker projection changed")
            jobs.append({"stage": stage, "dataset": dataset["id"],
                         "train_sha256": dataset["train_sha256"],
                         "validation_sha256": dataset["validation_sha256"],
                         "projection_sha256": manifest["projection_sha256"],
                         "configurations": tune_density.candidates(entry),
                         "fit_seed": 11, "budget_seconds": BUDGET_SECONDS})
    if len(jobs) != 12:
        raise ValueError("copula cohort must have twelve datasets")
    return {"format": "dope-benchmark-copula-native-round", "version": 1,
            "validation_only": True, "cohort_sha256": sha256(cohort_path),
            "methods_sha256": sha256(methods_path), "tuner_sha256": sha256(Path(__file__)),
            "adapter_sha256": sha256(Path(adapters.__file__)),
            "objective_sha256": entry["native_objective"]["implementation_sha256"],
            "python": str(python), "environment": versions, "jobs": jobs}


def _process(command: list[str], *, input_text: str | None,
             timeout: float, root: Path, label: str) -> dict:
    affinity = ",".join(map(str, tune_density.cpu_affinity()))
    result = subprocess.run(["taskset", "-c", affinity, *command], input=input_text,
                            capture_output=True, text=True, timeout=timeout)
    (root / f"{label}.stdout").write_text(result.stdout)
    (root / f"{label}.stderr").write_text(result.stderr)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return json.loads(result.stdout.strip().splitlines()[-1])


def tune(dataset: str, round_path: Path = ROOT / "round.lock.json",
         workers: Path = WORKERS, methods_path: Path = METHODS) -> dict:
    locked = json.loads(round_path.read_text())
    entry = json.loads(methods_path.read_text())["methods"]["GaussianCopula"]
    jobs = [job for job in locked.get("jobs", []) if job["dataset"] == dataset]
    if (locked.get("format") != "dope-benchmark-copula-native-round"
            or locked["tuner_sha256"] != sha256(Path(__file__))
            or locked["methods_sha256"] != sha256(methods_path)
            or locked["adapter_sha256"] != sha256(Path(adapters.__file__))
            or locked["objective_sha256"] != sha256(Path(__file__).with_name("native_objective.py"))
            or locked["environment"] != environment(Path(locked["python"]))
            or len(jobs) != 1):
        raise ValueError("copula validation round changed")
    job = jobs[0]
    worker = workers / dataset
    for name in ("train", "validation"):
        if sha256(worker / f"{name}.csv") != job[f"{name}_sha256"]:
            raise ValueError("worker partition changed")
    if sha256(worker / "projection.json") != job["projection_sha256"]:
        raise ValueError("worker projection changed")
    if job["configurations"] != tune_density.candidates(entry):
        raise ValueError("copula search space changed")
    if used_bytes(SCRATCH) + 1_000_000_000 > LIMIT:
        raise ValueError("benchmark scratch ceiling reached")
    root = round_path.parent / dataset
    root.mkdir(parents=True, exist_ok=True)
    trials = []
    elapsed = 0.0
    for index, config in enumerate(job["configurations"]):
        trial_root = root / f"trial-{index:02d}"
        artifact = trial_root / "artifact"
        receipt_path = trial_root / "attempt.json"
        metric_path = trial_root / "native-metric.json"
        if receipt_path.exists():
            trial = json.loads(receipt_path.read_text())
            if (trial["config"] != config
                    or trial["identity"]["round_sha256"] != sha256(round_path)):
                raise ValueError("resumed copula trial differs from lock")
            if trial["status"] == "ok":
                inventory, charged = artifact_inventory(
                    artifact, ["model.json", "projection.json"])
                if (inventory != trial["artifact_inventory"]
                        or charged != trial["artifact_bytes"]
                        or sha256(metric_path) != trial["metric_receipt_sha256"]):
                    raise ValueError("resumed copula evidence changed")
        else:
            trial_root.mkdir(parents=True, exist_ok=True)
            artifact.mkdir(exist_ok=True)
            started = time.monotonic()
            try:
                remaining = BUDGET_SECONDS - elapsed
                if remaining <= 0:
                    raise TimeoutError("copula cell budget exhausted")
                request = {"action": "fit", "method": "GaussianCopula",
                           "train": str(worker / "train.csv"),
                           "metadata": json.loads((worker / "projection.json").read_text()),
                           "config": config, "seed": 11, "artifact_dir": str(artifact),
                           "binary": None}
                fitted = _process([locked["python"], "-m", "research.benchmark.adapter_worker"],
                                  input_text=json.dumps(request),
                                  timeout=min(config["fit_timeout_seconds"], remaining),
                                  root=trial_root, label="fit")
                if fitted.get("status") != "ok" or fitted.get("files") != ["model.json"]:
                    raise ValueError("copula fit contract failed")
                (artifact / "projection.json").write_bytes((worker / "projection.json").read_bytes())
                inventory, charged = artifact_inventory(
                    artifact, ["model.json", "projection.json"])
                remaining = BUDGET_SECONDS - elapsed - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("copula cell budget exhausted")
                metric = _process([locked["python"], "-m",
                                   "research.benchmark.native_objective", "GaussianCopula",
                                   str(artifact), str(worker / "validation.csv")],
                                  input_text=None, timeout=remaining,
                                  root=trial_root, label="metric")
                if (metric["implementation_sha256"] != locked["objective_sha256"]
                        or metric["validation_sha256"] != job["validation_sha256"]
                        or metric["artifact_sha256"] != sha256(artifact / "model.json")
                        or not math.isfinite(metric["value"])):
                    raise ValueError("copula native metric changed")
                tune_density.write_once(metric_path, metric)
                trial = {"status": "ok", "config": config, "native_kpi": metric["value"],
                         "artifact_bytes": charged, "artifact_inventory": inventory,
                         "artifact_sha256": metric["artifact_sha256"],
                         "metric_receipt_path": str(metric_path),
                         "metric_receipt_sha256": sha256(metric_path)}
            except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
                trial = {"status": "timeout" if isinstance(error, (TimeoutError,
                         subprocess.TimeoutExpired)) else "failed", "config": config,
                         "error_type": type(error).__name__}
            trial["wall_seconds"] = time.monotonic() - started
            trial["identity"] = {"round_sha256": sha256(round_path),
                                 "dataset": dataset, "method": "GaussianCopula",
                                 "train_sha256": job["train_sha256"],
                                 "validation_sha256": job["validation_sha256"],
                                 "config": config, "fit_seed": 11}
            trial["logs"] = {name: sha256(trial_root / name)
                             for name in ("fit.stdout", "fit.stderr", "metric.stdout", "metric.stderr")
                             if (trial_root / name).exists()}
            tune_density.write_once(receipt_path, trial)
        elapsed += trial["wall_seconds"]
        trials.append(trial)
    successful = [(index, trial) for index, trial in enumerate(trials)
                  if trial["status"] == "ok"]
    if not successful or elapsed > BUDGET_SECONDS:
        raise ValueError("copula cell has no eligible selection")
    winner_index, winner = min(successful,
        key=lambda pair: (-pair[1]["native_kpi"], pair[1]["artifact_bytes"],
                          digest(pair[1]["config"]), pair[0]))
    selection = {"format": "dope-benchmark-validation-selection", "version": 1,
                 "method": "GaussianCopula", "dataset": dataset,
                 "partition": "validation", "objective": entry["native_objective"],
                 "round_sha256": sha256(round_path), "stage": job["stage"],
                 "selected_trial_index": winner_index,
                 "selected_config": winner["config"], "trials": trials,
                 "total_wall_seconds": elapsed, "test_opened": False}
    selection_path = root / "selection.json"
    tune_density.write_once(selection_path, selection)
    return {"dataset": dataset, "selected_trial_index": winner_index,
            "selected_native_kpi": winner["native_kpi"],
            "selection_path": str(selection_path),
            "selection_sha256": sha256(selection_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "tune"))
    parser.add_argument("dataset", nargs="?")
    args = parser.parse_args()
    if args.action == "freeze":
        tune_density.write_once(ROOT / "round.lock.json", freeze())
        print(json.dumps({"jobs": 12, "sha256": sha256(ROOT / "round.lock.json")},
                         sort_keys=True))
    else:
        if args.dataset is None:
            parser.error("tune requires a dataset")
        print(json.dumps(tune(args.dataset), sort_keys=True))


if __name__ == "__main__":
    main()
