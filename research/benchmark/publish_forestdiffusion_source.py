"""Publish verified original-package rights and generated-input contract evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from .manifest import digest
from .score import sha256

HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/method-source-audits/forestdiffusion-v1")
GPU = ROOT / "gpu-adapter-contract-v1"
CPU = ROOT / "adapter-contract-v1"
PORTABLE = Path("/mnt/fast-scratch/dope-benchmark/envs/forestdiffusion-python311-portable-v1")
OUTPUT = HERE / "forestdiffusion-source.lock.json"
COMMIT = "818ac3b9c8df7b7c763470c9a0691dacca8d8b37"
RUNTIME = "94a3e51bb79cec5c799fd96aec3a6e3c0b577e9cd06cccf7ce52c72c629d2285"
ROUND = "1deff71bd61869d69d265a25052f36f4002080d21825f93a437dd3c25a458f33"


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    return {"path": str(path), "sha256": sha256(path)}


def inventory(root):
    rows = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "contract artifact contains a symlink")
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(),
                         "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def native_mean(kpi):
    scores = kpi["auditor_seed_scores"]
    require(kpi["direction"] == "maximize" and kpi["shared_kpi_used_for_selection"] is False,
            "native objective changed")
    require(kpi["auditor_fit_seeds"] == list(range(5))
            and set(scores) == {"linear", "adaboost", "random_forest", "xgboost"}
            and all(len(v) == 5 and all(math.isfinite(x) for x in v) for v in scores.values()),
            "author four-model five-seed evidence incomplete")
    require(abs(mean(mean(v) for v in scores.values()) - kpi["value"]) <= 1e-12,
            "native mean does not reproduce")


def build():
    runtime_path = GPU / "runtime.lock.json"
    runtime = read(runtime_path)
    round_path = GPU / "round.lock.json"
    lock, completion = read(round_path), read(GPU / "completion.json")
    adapter_hash = sha256(HERE / "forestdiffusion_adapter.py")
    require(sha256(runtime_path) == RUNTIME and sha256(round_path) == ROUND
            and adapter_hash == runtime["adapter_sha256"] == lock["adapter_sha256"]
            and runtime["upstream_commit"] == COMMIT
            and completion["round_sha256"] == ROUND and completion["all_operations_passed"] is True
            and completion["generated_gpu_fits"] == 2 and completion["benchmark_fits"] == 0,
            "original GPU runtime or contract identity changed")
    require(runtime["python_package_license"] == "MIT"
            and runtime["repository_root_experiment_code_license"] == "unresolved_not_imported"
            and runtime["author_faithful_paper_dependency_reproduction"] is False,
            "source rights or reproduction scope changed")
    for root_key, files_key in (("package_root", "author_package_files"),
                               ("dependency_root", "dependency_files")):
        root = Path(runtime[root_key])
        for name, expected in runtime[files_key].items():
            require(sha256(root / name) == expected, "original source or dependency changed")
    preaudit, blobs = read(ROOT / "preaudit.json"), read(ROOT / "git-blob-verification.json")
    require(blobs["commit"] == preaudit["commit"] == COMMIT
            and blobs["preaudit_sha256"] == sha256(ROOT / "preaudit.json")
            and blobs["no_dataset_or_test_rows_read"] is True
            and sha256(ROOT / "source.tar.gz") == preaudit["archive_sha256"],
            "author archive or git-blob verification changed")
    metric = runtime["native_metric_reference"]
    require(sha256(ROOT / "source/metrics.py") == metric["sha256"], "author objective source changed")
    portable = read(PORTABLE / "runtime-manifest.json")
    require(sha256(PORTABLE / "runtime-manifest.json") == lock["portable_manifest_sha256"],
            "portable interpreter identity changed")
    for item in portable["files"]:
        require(sha256(PORTABLE / item["path"]) == item["sha256"],
                "portable interpreter file changed")
    contracts = []
    for task in ("regression", "binary"):
        root = GPU / task
        fit = read(root / "fit.receipt.json")
        sample, repeat = read(root / "sample.receipt.json"), read(root / "repeat.receipt.json")
        verification = read(root / "verification.json")
        files = inventory(root / "artifact")
        require(files == fit["artifact_inventory"]
                and sum(r["bytes"] for r in files) == fit["artifact_bytes"] == verification["full_artifact_bytes"]
                and fit["retained_training_row_containers"] is False
                and fit["original_sample_parity"] == "exact"
                and all(d.startswith("cuda") for d in fit["trained_booster_devices"])
                and verification["round_sha256"] == ROUND
                and verification["row_container_removal_parity_exact"] is True
                and verification["sample_replay_exact"] is True,
                "generated fit/sample or charged-byte contract changed")
        require(sample == repeat and sample["rows"] == 49 and sample["status"] == "ok"
                and sha256(root / "sample.csv") == sha256(root / "repeat.csv") == sample["sample_sha256"],
                "generated sample replay differs")
        native_mean(read(root / "native.receipt.json"))
        contracts.append({"task": task, "status": "ok", "artifact_bytes": fit["artifact_bytes"],
                          "artifact_inventory": files, "sample_rows": 49,
                          "sample_replay": "exact", "row_removal_replay": "exact",
                          "trained_devices": sorted(set(fit["trained_booster_devices"])),
                          "native_mean_verified": True,
                          "receipts": [reference(root / name) for name in
                                       ("fit.receipt.json", "sample.receipt.json", "repeat.receipt.json",
                                        "native.receipt.json", "verification.json")]})
    costs = []
    for root in (CPU, GPU):
        for task in ("regression", "binary"):
            for path in sorted((root / task).glob("*.attempt.json")):
                row = read(path)
                require(row["official_tests_opened"] is False and row["mfs_v2"] is None
                        and row["ptf_v1"] is None and row["exit_code"] == 0,
                        "generated contract operation failed or opened tests")
                costs.append({**reference(path), "operation": row["operation"],
                              "elapsed_seconds": row["elapsed_seconds"],
                              "peak_gpu_used_mib": row.get("peak_gpu_used_mib")})
    require(sum(r["operation"] == "fit" for r in costs) == 4, "generated fit cost inventory incomplete")
    gap = read(GPU / "aggregate-admission-gap.json")
    require(gap["round_sha256"] == ROUND and gap["full_aggregate_admission_verified"] is False,
            "historical resource-admission gap concealed")
    return {
        "format": "dope-forestdiffusion-source-lock", "version": 1, "status": "pilot_locked",
        "method": "ForestDiffusion/Forest-Flow", "full_matrix_admission": False,
        "upstream_url": "https://github.com/SamsungSAILMontreal/ForestDiffusion", "upstream_commit": COMMIT,
        "source_archive_sha256": preaudit["archive_sha256"],
        "git_blob_verification": reference(ROOT / "git-blob-verification.json"),
        "license": "MIT", "license_scope": "original Python package only",
        "license_evidence": {"url": f"https://github.com/SamsungSAILMontreal/ForestDiffusion/blob/{COMMIT}/Python-Package/base-ForestDiffusion/LICENSE.txt",
                             "sha256": runtime["python_package_license_sha256"]},
        "author_package_files": runtime["author_package_files"],
        "repository_root_experiment_code_license": "unresolved_not_imported",
        "implementation": "original_author_generator_with_study_fit_sample_wrapper",
        "independent_generator_implementation": False, "reported_experiment_reproductions_claimed": 0,
        "author_faithful_paper_dependency_reproduction": False,
        "adapter_source_sha256": adapter_hash, "gpu_runtime": reference(runtime_path),
        "cpu_runtime": reference(CPU / "runtime.lock.json"),
        "environment": runtime["environment"],
        "dependency_inventory": {"files": len(runtime["dependency_files"]),
                                 "sha256": digest(runtime["dependency_files"]), "complete_runtime_reference": reference(runtime_path)},
        "portable_interpreter": {"manifest": reference(PORTABLE / "runtime-manifest.json"),
                                 "version": portable["cpython_version"], "files": len(portable["files"])},
        "supported_tasks": ["binary", "regression"], "track": "common_numeric",
        "generation_process": "flow", "author_default_config": runtime["author_library_defaults"],
        "resource_overrides": runtime["resource_overrides"], "n_batch": 1,
        "native_objective": metric, "native_metric_implementation": runtime["native_metric_implementation"],
        "cpu_native_search_space": runtime["native_search_space"],
        "gpu_native_search_space": [runtime["native_search_space"][i] for i in (0, 1)],
        "total_trial_cap_per_dataset": 8, "method_dataset_seconds_cap": 43200,
        "fit_seconds_cap": 600, "gpu_vram_mib_cap": 16384,
        "artifact_policy": "charge sampler, adapter metadata and projection; remove retained row containers with exact original/serialized sample parity",
        "generated_input_contracts": contracts,
        "generated_contract_costs": {"fits": 4, "operations": costs,
                                     "summed_operation_seconds": sum(r["elapsed_seconds"] for r in costs),
                                     "scope": "contract development, no real benchmark fitting or reported-experiment reproduction"},
        "historical_aggregate_resource_admission_complete": False,
        "historical_admission_correction": reference(GPU / "aggregate-admission-gap.json"),
        "independent_cpu_sample_and_initializer_checks": reference(CPU / "independent-verification-v1/manifest.json"),
        "formal_dp_claim": False, "official_tests_opened": False,
        "mfs_v2": None, "ptf_v1": None, "release_safe": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--schema", type=Path)
    args = parser.parse_args()
    document = build()
    args.output.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")
    if args.schema:
        args.schema.write_text(json.dumps(source_schema(document), sort_keys=True, indent=2) + "\n")
    print(args.output)


def source_schema(document):
    """Strict shape plus fixed source identity, release gates, and compute caps."""
    def shape(value, key=""):
        if value is None:
            return {"type": "null"}
        if isinstance(value, bool):
            return {"const": value}
        if isinstance(value, str):
            result = {"type": "string", "minLength": 1}
            if key.endswith("sha256") or len(value) == 64 and all(c in "0123456789abcdef" for c in value):
                result["pattern"] = "^[0-9a-f]{64}$"
            if key == "path" and value.startswith("/mnt/"):
                result["pattern"] = "^/mnt/fast-scratch/dope-benchmark/"
            return result
        if isinstance(value, (int, float)):
            return {"type": "integer" if isinstance(value, int) else "number", "minimum": 0}
        if isinstance(value, list):
            variants = {json.dumps(shape(row), sort_keys=True) for row in value}
            item = {"anyOf": [json.loads(row) for row in sorted(variants)]} if variants else {}
            return {"type": "array", "minItems": len(value), "maxItems": len(value), "items": item}
        return {"type": "object", "additionalProperties": False, "required": sorted(value),
                "properties": {k: shape(v, k) for k, v in value.items()}}

    result = shape(document)
    result["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    for name in ("format", "version", "status", "method", "upstream_commit", "license",
                 "implementation", "reported_experiment_reproductions_claimed",
                 "total_trial_cap_per_dataset", "method_dataset_seconds_cap", "fit_seconds_cap",
                 "gpu_vram_mib_cap", "track", "generation_process", "repository_root_experiment_code_license"):
        result["properties"][name] = {"const": document[name]}
    return result


if __name__ == "__main__":
    main()
