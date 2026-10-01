"""Freeze the 24-hour pilot's author-objective CTGAN/TVAE GPU grid."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from . import sdv_round
from .score import sha256


PILOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")


def configs(entry: dict) -> list[dict]:
    default = entry["default_config"]
    search = entry["tuning_search_space"]
    values = [dict(default)]
    for batch, embedding in itertools.product(search["batch_size"],
                                               search["embedding_dim"]):
        config = {**default, "batch_size": batch, "embedding_dim": embedding}
        if config not in values:
            values.append(config)
    if len(values) != 4:
        raise ValueError("SDV pilot grid differs from four frozen author parameters")
    return values


def freeze(root: Path = PILOT) -> Path:
    here = Path(__file__).parent
    methods_path = here / "methods.lock.json"
    runtime_path = here / "sdv-runtime.lock.json"
    methods = json.loads(methods_path.read_text())["methods"]
    worker_root = root / "prepared-v2/worker"
    jobs = []
    for method in ("CTGAN", "TVAE"):
        entry = methods[method]
        if (entry["status"] != "locked"
                or entry["source_sha256"] != sha256(Path(entry["source_archive"]))
                or entry["adapter_sha256"] != sha256(here / "sdv_adapter.py")
                or entry["worker_sha256"] != sha256(here / "adapter_worker.py")
                or entry["dependency_or_container_digest"] != sha256(runtime_path)
                or entry["native_objective"]["status"] != "locked"
                or entry["native_objective"]["name"] != "sdmetrics_mean_regression_r2"):
            raise ValueError("SDV method or native KPI differs from lock")
        grid = configs(entry)
        for dataset in ("California", "News"):
            worker = sdv_round.worker_for_job({"worker_root": str(worker_root)},
                                              {"dataset": dataset})
            manifest = json.loads((worker / "worker-manifest.json").read_text())
            if (manifest["dataset_id"] != dataset
                    or json.loads((worker / "projection.json").read_text())["task"] != "regression"):
                raise ValueError("SDV pilot objective requires regression inputs")
            for part in ("train", "validation"):
                if sha256(worker / f"{part}.csv") != manifest["projected_hashes"][part]:
                    raise ValueError("SDV pilot worker partition changed")
            for trial, config in enumerate(grid):
                jobs.append({"dataset": dataset, "stage": "pilot", "method": method,
                             "trial": trial, "config": config,
                             "fit_seed": 11, "sample_seed": 101,
                             "size_multiplier": 1,
                             "train_rows": manifest["train_rows"],
                             "train_sha256": manifest["projected_hashes"]["train"],
                             "validation_sha256": manifest["projected_hashes"]["validation"],
                             "projection_sha256": manifest["projection_sha256"]})
    source_names = ("sdv_adapter.py", "sdv_round.py", "freeze_pilot_sdv.py",
                    "score.py", "contract.json", "fetch_jope.py", "gpu_probe.py",
                    "inventory_hosts.py", "manifest.py", "methods.lock.json",
                    "sdv-runtime.lock.json")
    prior = root / "sdv-native-v1/round.lock.json"
    if not prior.exists() or list((prior.parent / "jobs").glob("*/attempt-*")):
        raise ValueError("expected blocked v1 admission without a launched fit")
    lock = {"format": "dope-sdv-validation-round", "version": 3,
            "validation_only": True, "official_tests_opened": False,
            "worker_root": str(worker_root),
            "cpu_slot": list(range(32, 48)),
            "supersedes_sha256": sha256(prior),
            "pilot_dataset_lock_sha256": sha256(here / "pilot-24h-datasets.lock.json"),
            "methods_lock_sha256": sha256(methods_path),
            "adult_binary_status": "not_applicable_until_author_classification_objective_locked",
            "jobs": jobs, "environment": sdv_round.environment(),
            "source_files": {name: sha256(here / name) for name in source_names},
            "dependency_files": {
                path.relative_to(sdv_round.ROOT / "deps").as_posix(): sha256(path)
                for path in sorted((sdv_round.ROOT / "deps").rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts},
            "budgets": {"gpu_fit_seconds": 600, "gpu_vram_bytes": 16 * 1024**3,
                        "cpu_cores_per_host": 16, "gpu_fits_per_host": 1,
                        "cell_max_attempts": 8, "cell_wall_seconds": 43_200,
                        "planned_trials_per_cell": 4,
                        "scratch_ceiling_bytes": 200_000_000_000},
            "production_ptf_v1": None, "mfs_v2": None}
    path = root / "sdv-native-v2/round.lock.json"
    sdv_round.write_once(path, lock)
    return path


if __name__ == "__main__":
    path = freeze()
    print(json.dumps({"jobs": 16, "lock_sha256": sha256(path)}, sort_keys=True))
