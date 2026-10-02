"""Publish complete bounded GPU research with failures and matched native references."""

from __future__ import annotations
import json
from pathlib import Path
from research.benchmark.manifest import digest
from research.benchmark.score import sha256, artifact_inventory
from research.benchmark.publish_native_neural_fivefit import sealed, evidence

HERE = Path(__file__).resolve().parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h/dope-target-refinement-v1")
DATASETS = ("Adult", "California", "News")
PROFILES = (
    "features12_steps512",
    "features12_steps2048",
    "features24_steps512",
    "features24_steps2048",
)
FIT_SEEDS = (11, 23)
SAMPLE_SEEDS = (101, 211, 307)
SIZES = (1, 4)
AUDITORS = ("catboost", "linear", "mlp")
PINS = {
    "fit": "1897f312e68de13124c5ac78b73e0320c61c222ed8cfd119e8bb8a971e56a0c0",
    "raw": "3fd6b707b9fabd78d408a3723ca4182e06eabb521e03eb2126c1ce5797ed903f",
    "pack": "238ddb0d9b9126442ea41d117943670dc9d9a6b95b1d230aa56cf35ddcee0304",
    "packed": "89fb505a07991cfdcd4c8882252154fd6e068daf0b59d16b832de3c374654bc2",
    "reference": "2908181c30df8871bc58968d89df163bae770c80fc8c6a23f26e30fa0cfcaa9b",
}


def read(p):
    return json.loads(Path(p).read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def reference(p):
    return {"path": str(p), "sha256": sha256(p)}


def keys(row):
    return (
        row["dataset"],
        row["profile"],
        row["fit_seed"],
        row["sample_seed"],
        row["size_multiplier"],
    )


def fit_attempts(lock):
    rows = []
    index = {}
    require(
        len(lock["gpu_jobs"]) == 24
        and {(j["dataset"], j["research_profile"], j["seed"]) for j in lock["gpu_jobs"]}
        == {(d, p, f) for d in DATASETS for p in PROFILES for f in FIT_SEEDS},
        "frozen GPU grid changed",
    )
    for job in lock["gpu_jobs"]:
        base = ROOT / "dispatch" / digest(job)
        path = base / "receipt.json"
        dispatch = read(path)
        sealed(dispatch)
        require(
            dispatch["job"] == job
            and dispatch["round_sha256"] == PINS["fit"]
            and dispatch["source_sha256"] == lock["source_sha256"],
            "GPU dispatch identity changed",
        )
        admission = base / "admission-0001.json"
        require(
            sha256(admission) == dispatch["admission_sha256"]
            and sha256(base / "dispatch.log") == dispatch["log_sha256"],
            "dispatch evidence changed",
        )
        observed = read(admission)
        host = observed["inventory"]
        require(
            observed["admitted"] is True
            and host["host"] == dispatch["host"]
            and not host["active_gpu_processes"]
            and host["gpus"][0]["memory_free_mib"] >= 17 * 1024
            and host["memory"]["MemAvailable"] >= lock["ram_reservation_bytes"]
            and observed["benchmark_scratch_bytes"] + lock["scratch_reservation_bytes"]
            <= 200_000_000_000,
            "GPU admission changed",
        )
        child = Path(dispatch["child"]["receipt"])
        receipt = read(child)
        model = child.with_suffix(".dpk")
        identity = receipt["identity"]
        worker = Path(lock["workers"][job["dataset"]]["path"])
        worker_files = lock["workers"][job["dataset"]]["files"]
        worker_manifest = read(worker / "worker-manifest.json")
        require(
            sha256(child) == dispatch["child"]["receipt_sha256"]
            and receipt["status"] == dispatch["status"]
            and receipt["exit_code"] == 0
            and receipt["validation_only"] is True
            and receipt["ptf_v1"] is None,
            "GPU fit evidence changed",
        )
        require(
            identity["dataset"] == job["dataset"]
            and identity["seed"] == job["seed"]
            and identity["research_profile"] == job["research_profile"]
            and identity["round_lock_sha256"] == PINS["fit"]
            and identity["binary_sha256"] == lock["gpu_binary_sha256"]
            and identity["probe_source_sha256"] == lock["probe_source_sha256"]
            and identity["host_runtime_lock_sha256"]
            == lock["host_runtime_locks"][dispatch["host"]]["sha256"]
            and identity["sample_seeds"] == list(SAMPLE_SEEDS)
            and identity["validation_size_multipliers"] == list(SIZES),
            "GPU fit identity changed",
        )
        require(
            all(
                identity[name] == job[name]
                for name in ("candidate", "target_weight", "structural_penalty")
            )
            and identity["train_sha256"] == worker_files["train.csv"]
            and identity["validation_sha256"] == worker_files["validation.csv"]
            and identity["projection_sha256"] == worker_files["projection.json"]
            and identity["split"] == worker_manifest["split_hashes"]
            and identity["track"] == "common-numeric"
            and identity["tier"] == "l3"
            and identity["dp_epsilon"] is None
            and identity["fit_deadline_seconds"] == 600,
            "GPU data or configuration identity changed",
        )
        require(
            receipt["artifact_sha256"] == sha256(model)
            and receipt["artifact_bytes"]
            == model.stat().st_size + (worker / "projection.json").stat().st_size
            and receipt["elapsed_seconds"] <= 600
            and receipt["peak_gpu_used_mib"] <= 16384,
            "GPU resource or byte charge changed",
        )
        inspection = child.with_suffix(".inspection.json")
        require(
            sha256(inspection) == receipt["inspection_sha256"]
            and sha256(child.with_suffix(".log")) == receipt["log_sha256"]
            and read(inspection)["target"] == "compact_neural_residual"
            and receipt["selected_target_operator"] == "compact_neural_residual",
            "trained GPU target missing",
        )
        require(
            receipt["status"]
            == ("charged_artifact_cap" if job["dataset"] == "News" else "ok"),
            "raw fit status changed",
        )
        packed = None
        if job["dataset"] == "News":
            packed_path = ROOT / "news-packed-v1/cells" / sha256(child) / "receipt.json"
            packed = read(packed_path)
            sealed(packed)
            require(
                packed["round_sha256"] == PINS["pack"]
                and packed["job"]["fit_receipt_sha256"] == sha256(child)
                and packed["status"] == "l3_byte_eligible"
                and packed["learned_bytes_unchanged"]
                and packed["projection_exactly_reconstructible"]
                and packed["original_failed_fit_preserved"],
                "packed artifact identity changed",
            )
            artifact = Path(packed["artifact_path"])
            inventory, charged = artifact_inventory(
                artifact, ["model.dpk", "projection.compact.json"]
            )
            require(
                inventory == packed["artifact_inventory"]
                and charged == packed["artifact_bytes"]
                and charged <= 10240
                and sha256(artifact / "model.dpk") == sha256(model),
                "packed byte charge changed",
            )
            from research.benchmark.refine_projection import unpacked_projection

            require(
                unpacked_projection((artifact / "projection.compact.json").read_bytes())
                == read(worker / "projection.json"),
                "packed projection changed",
            )
            require(
                len(packed["sample_parity_checks"]) == 3
                and {r["sample_seed"] for r in packed["sample_parity_checks"]}
                == set(SAMPLE_SEEDS),
                "packing parity matrix incomplete",
            )
            for check in packed["sample_parity_checks"]:
                require(
                    check["exact_original_replay"] is True
                    and check["rows"] == 64
                    and all(
                        sha256(
                            packed_path.parent / f'{label}-{check["sample_seed"]}.csv'
                        )
                        == check["sample_sha256"]
                        for label in ["original", "packed"]
                    ),
                    "packing parity changed",
                )
        row = {
            "dataset": job["dataset"],
            "profile": job["research_profile"],
            "fit_seed": job["seed"],
            "host": dispatch["host"],
            "raw_status": receipt["status"],
            "raw_artifact_bytes": receipt["artifact_bytes"],
            "model_bytes": model.stat().st_size,
            "packed_artifact_bytes": packed["artifact_bytes"] if packed else None,
            "artifact_bytes": (
                packed["artifact_bytes"] if packed else receipt["artifact_bytes"]
            ),
            "elapsed_gpu_fit_seconds": receipt["elapsed_seconds"],
            "dispatch_elapsed_seconds": dispatch["elapsed_seconds"],
            "peak_gpu_used_mib": receipt["peak_gpu_used_mib"],
            "energy_joules_estimate": receipt["energy_joules_estimate"],
            "gpu_target_operator": "compact_neural_residual",
            "original_target_verified_field": receipt["trained_gpu_target_verified"],
            "model_sha256": sha256(model),
            "fit_receipt": reference(child),
            "dispatch_receipt": reference(path),
            "admission": reference(admission),
            "packing_receipt": reference(packed_path) if packed else None,
            "source_unavailable": False,
            "contributes_dope_win": False,
            "release_safe": None,
        }
        rows.append(row)
        index[digest(job)] = row
    return rows, index


def validation_cells(name, lock, fit_index):
    packed = name == "news-packed-validation-v1"
    root = ROOT / name
    rows = []
    require(len(lock["jobs"]) == (48 if packed else 144), "validation matrix changed")
    expected = {
        (d, p, f, s, z)
        for d in (("News",) if packed else DATASETS)
        for p in PROFILES
        for f in FIT_SEEDS
        for s in SAMPLE_SEEDS
        for z in SIZES
    }
    actual = [
        (
            j["fit_job"]["dataset"],
            j["fit_job"]["research_profile"],
            j["fit_job"]["seed"],
            j["sample_seed"],
            j["size_multiplier"],
        )
        for j in lock["jobs"]
    ]
    require(
        len(actual) == len(set(actual)) and set(actual) == expected,
        "raw or packed validation matrix incomplete or duplicated",
    )
    for job in lock["jobs"]:
        path = root / "cells" / digest(job) / "receipt.json"
        receipt = read(path)
        sealed(receipt)
        require(
            receipt["job"] == job
            and receipt["round_sha256"] == sha256(root / "round.lock.json")
            and receipt["new_fits_started"] == 0
            and receipt["source_unavailable"] is False
            and receipt["counts_as_dope_win"] is False,
            "validation lineage changed",
        )
        fit = fit_index[job["fit_job_key"]]
        require(job["fit_job_key"] == digest(job["fit_job"]), "fit key changed")
        worker = Path(job["worker"]["path"])
        manifest = read(worker / "worker-manifest.json")
        require(
            job["auditor_fit_seed"] == job["sample_seed"]
            and job["rows"] == manifest["train_rows"] * job["size_multiplier"],
            "auditor seed or sample size changed",
        )
        row = {
            "dataset": fit["dataset"],
            "profile": fit["profile"],
            "fit_seed": fit["fit_seed"],
            "sample_seed": job["sample_seed"],
            "size_multiplier": job["size_multiplier"],
            "artifact_recipe": "lossless_packed" if packed else "raw",
            "status": receipt["status"],
            "artifact_bytes": (
                fit["artifact_bytes"]
                if packed or fit["raw_status"] == "ok"
                else fit["raw_artifact_bytes"]
            ),
            "receipt": reference(path),
            "metrics": None,
            "sample_sha256": None,
            "metric_receipt": None,
            "operations": receipt.get("operations", []),
            "source_unavailable": False,
            "contributes_dope_win": False,
            "release_safe": None,
        }
        if receipt["status"] == "ok":
            evidence(path.parent, receipt)
            metric_path = path.parent / "metrics.json"
            metric = read(metric_path)
            repeat = read(path.parent / "metrics-replay.json")
            payload = lambda r: {k: v for k, v in r.items() if k != "metric_seconds"}
            require(
                payload(metric) == payload(repeat)
                and metric["implementation_sha256"] == lock["metric_source_sha256"]
                and metric["dependencies"] == lock["metric_environment"]
                and metric["mfs_v2"] is None
                and metric["gate_profile_complete"] is False
                and metric["rows"]["synthetic"] == job["rows"]
                and metric["rows"]["train"] == manifest["train_rows"]
                and metric["task"] == read(worker / "projection.json")["task"]
                and len(receipt["operations"]) == 3
                and all(
                    op["status"] == "ok"
                    and op["exit_code"] == 0
                    and 0 <= op["elapsed_seconds"] <= 600
                    for op in receipt["operations"]
                ),
                "shared metric replay changed",
            )
            require(
                sha256(path.parent / "sample.csv") == sha256(path.parent / "repeat.csv")
                and receipt["sample_replay_exact"]
                and receipt["metric_replay_exact"]
                and receipt["artifact_bytes"] == row["artifact_bytes"],
                "sample parity or byte charge changed",
            )
            require(
                receipt["fit_receipt_sha256"] == fit["fit_receipt"]["sha256"]
                and receipt["model_sha256"] == fit["model_sha256"],
                "sample model lineage changed",
            )
            if packed:
                require(
                    receipt["packed_receipt_sha256"]
                    == job["packed_receipt_sha256"]
                    == fit["packing_receipt"]["sha256"],
                    "packed validation changed",
                )
            row.update(
                metrics=metric,
                sample_sha256=sha256(path.parent / "sample.csv"),
                metric_receipt=reference(metric_path),
            )
        else:
            require(
                receipt["status"]
                in (
                    "fit_unavailable",
                    "sample_timeout",
                    "sample_failed",
                    "metric_timeout",
                    "metric_failed",
                    "sample_replay_mismatch",
                    "pilot_deadline_unstarted",
                ),
                "unknown failed cell status",
            )
            if receipt["status"] == "fit_unavailable":
                require(
                    not packed
                    and fit["dataset"] == "News"
                    and fit["raw_status"] == "charged_artifact_cap",
                    "unexpected unavailable fit",
                )
            if "evidence_files" in receipt:
                evidence(path.parent, receipt)
        rows.append(row)
    return rows
