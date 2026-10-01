"""Freeze the author-library GPU validation grid before fitting any S3 cell."""

from __future__ import annotations

import inspect
import itertools
import json
import shutil
from pathlib import Path

from .score import sha256
from .sdv_round import ROOT, SCRATCH, WORKERS, environment, write_once


def freeze() -> dict:
    from ctgan import CTGAN, TVAE
    here = Path(__file__).parent
    cohort_path = SCRATCH / "s3-v1/gpu-cohort.lock.json"
    cohort = json.loads(cohort_path.read_text())
    source = SCRATCH / "source-snapshots/ctgan-826da23f"
    upstream = next(source.iterdir())
    for path in (upstream / "ctgan").rglob("*.py"):
        installed = ROOT / "deps" / path.relative_to(upstream)
        if sha256(path) != sha256(installed):
            raise ValueError("installed CTGAN differs from author release source")
    methods = {}
    for model in (CTGAN, TVAE):
        default = {name: param.default for name, param in inspect.signature(model).parameters.items()}
        default = json.loads(json.dumps(default))
        configs = [default]
        for batch, embedding in itertools.product((500, 100), (128, 64)):
            config = {**default, "batch_size": batch, "embedding_dim": embedding}
            if config not in configs:
                configs.append(config)
        methods[model.__name__] = {
            "upstream_url": "https://github.com/sdv-dev/CTGAN",
            "upstream_commit": "826da23f8f9385ad15fd206ecad691e04cb0ccdc",
            "source_archive_sha256": sha256(SCRATCH / "source-snapshots/ctgan-826da23f.tar.gz"),
            "license": "BUSL-1.1", "license_sha256": sha256(upstream / "LICENSE"),
            "license_evidence": "https://github.com/sdv-dev/CTGAN/blob/826da23f8f9385ad15fd206ecad691e04cb0ccdc/LICENSE",
            "use": "non-production benchmark research",
            "default_config": default, "configurations": configs,
            "search_space_origin": "study grid of author-supported batch_size and embedding_dim; other author defaults retained",
            "track": "common_numeric", "supported_task": "regression",
            "native_objective": {"name": "sdmetrics_mean_regression_r2", "direction": "maximize",
                "components": ["LinearRegression", "MLPRegressor"],
                "aggregation": "study-prespecified arithmetic mean of the two author-library efficacy metrics",
                "source": "https://docs.sdv.dev/sdmetrics/data-metrics/metrics-in-beta/ml-efficacy-single-table/regression",
                "upstream_commit": "6e75fb1ac4469152abe4dabebeb87eb6828d10a5",
                "implementation_sha256": sha256(here / "sdv_adapter.py"),
                "partition": "validation", "metric_seed": 1729,
                "tie_breaks": ["artifact_bytes_ascending", "config_sha256_ascending"]}}
    jobs = []
    for trial in range(4):
        for stage in ("discovery", "confirmation"):
            for dataset in cohort[stage]:
                worker = WORKERS / dataset["id"]
                manifest = json.loads((worker / "worker-manifest.json").read_text())
                for part in ("train", "validation"):
                    if sha256(worker / f"{part}.csv") != dataset[f"{part}_sha256"]:
                        raise ValueError("cohort worker hash mismatch")
                if json.loads((worker / "projection.json").read_text())["task"] != "regression":
                    raise ValueError("native efficacy only admitted for regression")
                for method, entry in methods.items():
                    jobs.append({"dataset": dataset["id"], "stage": stage, "method": method,
                                 "trial": trial, "config": entry["configurations"][trial],
                                 "fit_seed": 11, "sample_seed": 101, "size_multiplier": 1,
                                 "train_rows": manifest["train_rows"],
                                 "train_sha256": dataset["train_sha256"],
                                 "validation_sha256": dataset["validation_sha256"],
                                 "projection_sha256": manifest["projection_sha256"]})
    source_names = ("sdv_adapter.py", "sdv_round.py", "freeze_sdv_round.py", "score.py",
                    "contract.json", "fetch_jope.py", "gpu_probe.py", "inventory_hosts.py", "manifest.py")
    lock = {"format": "dope-sdv-validation-round", "version": 1, "validation_only": True,
            "official_tests_opened": False, "cohort_sha256": sha256(cohort_path),
            "methods": methods, "jobs": jobs, "environment": environment(),
            "source_files": {name: sha256(here / name) for name in source_names},
            "dependency_files": {p.relative_to(ROOT / "deps").as_posix(): sha256(p)
                                 for p in sorted((ROOT / "deps").rglob("*"))
                                 if p.is_file() and "__pycache__" not in p.parts},
            "budgets": {"gpu_fit_seconds": 600, "gpu_vram_bytes": 16 * 1024**3,
                        "cpu_cores_per_host": 16, "gpu_fits_per_host": 1,
                        "cell_max_attempts": 8, "cell_wall_seconds": 43_200,
                        "planned_trials_per_cell": 4, "scratch_ceiling_bytes": 200_000_000_000},
            "production_ptf_v1": None, "mfs_v2": None}
    write_once(ROOT / "round.lock.json", lock)
    package = ROOT / "package/research/benchmark"
    package.mkdir(parents=True, exist_ok=True)
    for name in source_names:
        shutil.copy2(here / name, package / name)
    return lock


if __name__ == "__main__":
    result = freeze()
    print(json.dumps({"jobs": len(result["jobs"]), "lock_sha256": sha256(ROOT / "round.lock.json")}))
