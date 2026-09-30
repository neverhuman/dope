"""Freeze GaussianCopula native tuning on the 100 prepared S3 lineages."""

from __future__ import annotations

import json
from pathlib import Path

from . import adapters, tune_copula, tune_density
from .score import sha256


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
DATA_LOCK = Path(__file__).with_name("results") / "s3-data.lock.json"
OUTPUT = SCRATCH / "copula-native-all-v1/round.lock.json"


def freeze(data_path: Path = DATA_LOCK,
           methods_path: Path = tune_copula.METHODS,
           workers: Path = tune_copula.WORKERS,
           python: Path = tune_copula.PYTHON) -> dict:
    data = json.loads(data_path.read_text())
    entry = json.loads(methods_path.read_text())["methods"]["GaussianCopula"]
    versions = tune_copula.environment(python)
    if (data["format"] != "dope-benchmark-s3-data-lock" or data["prepared"] != 100
            or data["final_evaluation_authorized"] is not False
            or entry["status"] != "locked"
            or entry["adapter_sha256"] != sha256(Path(adapters.__file__))
            or entry["native_objective"]["implementation_sha256"]
            != sha256(Path(__file__).with_name("native_objective.py"))
            or versions != {"copulas": entry["dependency_version"].split("==")[1],
                            **entry["dependency_versions"]}):
        raise ValueError("copula source, objective, data, or environment changed")
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
        jobs.append({"stage": "all_validation_lineages", "dataset": dataset["id"],
                     "train_sha256": manifest["projected_hashes"]["train"],
                     "validation_sha256": manifest["projected_hashes"]["validation"],
                     "projection_sha256": manifest["projection_sha256"],
                     "configurations": tune_density.candidates(entry),
                     "fit_seed": 11, "budget_seconds": tune_copula.BUDGET_SECONDS})
    if len(jobs) != 100:
        raise ValueError("GaussianCopula S3 validation matrix must have 100 cells")
    return {"format": "dope-benchmark-copula-native-round", "version": 1,
            "validation_only": True, "s3_data_lock_sha256": sha256(data_path),
            "methods_sha256": sha256(methods_path),
            "tuner_sha256": sha256(Path(tune_copula.__file__)),
            "adapter_sha256": sha256(Path(adapters.__file__)),
            "objective_sha256": entry["native_objective"]["implementation_sha256"],
            "python": str(python), "environment": versions, "jobs": jobs}


def main() -> None:
    lock = freeze()
    tune_density.write_once(OUTPUT, lock)
    print(json.dumps({"jobs": len(lock["jobs"]), "sha256": sha256(OUTPUT)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
