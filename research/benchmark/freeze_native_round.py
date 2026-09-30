"""Freeze the compact density validation matrix before tuning starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import adapters, tune_density
from .score import sha256


def freeze(cohort_path: Path, worker_root: Path, methods_lock: Path) -> dict:
    cohort = json.loads(cohort_path.read_text())
    methods = json.loads(methods_lock.read_text())["methods"]
    if (cohort.get("format") != "dope-benchmark-gpu-cohort-lock"
            or cohort.get("validation_only") is not True):
        raise ValueError("expected frozen validation cohort")
    jobs = []
    for stage in ("discovery", "confirmation"):
        for dataset in cohort[stage]:
            worker = worker_root / dataset["id"]
            manifest = json.loads((worker / "worker-manifest.json").read_text())
            if (manifest["dataset_id"] != dataset["id"]
                    or manifest["projected_hashes"]["train"] != dataset["train_sha256"]
                    or manifest["projected_hashes"]["validation"] != dataset["validation_sha256"]):
                raise ValueError("worker and cohort differ")
            for name in ("train", "validation"):
                if sha256(worker / f"{name}.csv") != manifest["projected_hashes"][name]:
                    raise ValueError("worker partition changed")
            if sha256(worker / "projection.json") != manifest["projection_sha256"]:
                raise ValueError("worker projection changed")
            for method in tune_density.METHODS:
                entry = methods[method]
                if (entry["status"] != "locked"
                        or entry["adapter_sha256"] != sha256(Path(adapters.__file__))):
                    raise ValueError("compact reference method changed")
                jobs.append({"stage": stage, "dataset": dataset["id"], "method": method,
                             "train_sha256": manifest["projected_hashes"]["train"],
                             "validation_sha256": manifest["projected_hashes"]["validation"],
                             "projection_sha256": manifest["projection_sha256"],
                             "configurations": tune_density.candidates(entry),
                             "fit_seed": 11, "budget_seconds": tune_density.BUDGET_SECONDS})
    if len(jobs) != 24:
        raise ValueError("compact validation matrix must contain 24 cells")
    return {"format": "dope-benchmark-compact-native-round", "version": 1,
            "validation_only": True, "cohort_sha256": sha256(cohort_path),
            "method_lock_sha256": sha256(methods_lock),
            "adapter_sha256": sha256(Path(adapters.__file__)),
            "tuner_sha256": sha256(Path(tune_density.__file__)), "jobs": jobs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path)
    parser.add_argument("worker_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--methods-lock", type=Path,
                        default=Path(__file__).with_name("methods.lock.json"))
    args = parser.parse_args()
    lock = freeze(args.cohort, args.worker_root, args.methods_lock)
    tune_density.write_once(args.output, lock)
    print(json.dumps({"jobs": len(lock["jobs"]), "sha256": sha256(args.output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
