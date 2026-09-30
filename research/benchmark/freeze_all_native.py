"""Freeze the two study-owned density baselines on all prepared S3 lineages."""

from __future__ import annotations

import json
from pathlib import Path

from . import adapters, tune_density
from .score import sha256


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
RESULTS = Path(__file__).with_name("results")
DATA_LOCK = RESULTS / "s3-data.lock.json"
METHODS = Path(__file__).with_name("methods.lock.json")
WORKERS = SCRATCH / "s3-v1/prepared/worker"
OUTPUT = SCRATCH / "compact-native-all-v1/round.lock.json"


def freeze(data_path: Path = DATA_LOCK, methods_path: Path = METHODS,
           workers: Path = WORKERS) -> dict:
    data = json.loads(data_path.read_text())
    entries = json.loads(methods_path.read_text())["methods"]
    if (data["format"] != "dope-benchmark-s3-data-lock"
            or data["prepared"] != 100 or data["final_evaluation_authorized"] is not False):
        raise ValueError("S3 training-derived population is not frozen")
    jobs = []
    for dataset in data["entries"]:
        if dataset["status"] != "prepared":
            continue
        worker = workers / dataset["id"]
        manifest = json.loads((worker / "worker-manifest.json").read_text())
        if (manifest["dataset_id"] != dataset["id"]
                or manifest["projected_hashes"] !=
                {name: dataset["projected_files"][name] for name in ("train", "validation")}
                or manifest["projection_sha256"] != dataset["projection_sha256"]):
            raise ValueError("worker and S3 data lock differ")
        for name in ("train", "validation"):
            if sha256(worker / f"{name}.csv") != manifest["projected_hashes"][name]:
                raise ValueError("worker partition changed")
        if sha256(worker / "projection.json") != manifest["projection_sha256"]:
            raise ValueError("worker projection changed")
        for method in tune_density.METHODS:
            entry = entries[method]
            if (entry["status"] != "locked"
                    or entry["adapter_sha256"] != sha256(Path(adapters.__file__))):
                raise ValueError("compact reference method changed")
            jobs.append({"stage": "all_validation_lineages", "dataset": dataset["id"],
                         "method": method,
                         "train_sha256": manifest["projected_hashes"]["train"],
                         "validation_sha256": manifest["projected_hashes"]["validation"],
                         "projection_sha256": manifest["projection_sha256"],
                         "configurations": tune_density.candidates(entry),
                         "fit_seed": 11, "budget_seconds": tune_density.BUDGET_SECONDS})
    if len(jobs) != 200:
        raise ValueError("compact S3 validation matrix must have 200 cells")
    return {"format": "dope-benchmark-compact-native-round", "version": 1,
            "validation_only": True, "s3_data_lock_sha256": sha256(data_path),
            "method_lock_sha256": sha256(methods_path),
            "adapter_sha256": sha256(Path(adapters.__file__)),
            "tuner_sha256": sha256(Path(tune_density.__file__)), "jobs": jobs}


def main() -> None:
    lock = freeze()
    tune_density.write_once(OUTPUT, lock)
    print(json.dumps({"jobs": len(lock["jobs"]), "sha256": sha256(OUTPUT)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
