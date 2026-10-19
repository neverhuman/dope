"""Execute the signed, offline W01 native model qualification cases.

Each model/task case runs in a fresh subprocess with an externally observed
wall and RSS cap. The result records measured gates and leaves independent
review pending; execution alone never confers registry admission.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from research.transfer.native_qualification import (
    QualificationError, new_result, sha256_file, verify_checkpoints,
    verify_signed_manifest,
)
from research.transfer.native_qualification_runtime import (
    RUNTIME_FILE, _adapter_parity, _case_inputs, _hash, _run_native,
)

RUNNER_FILE = Path(__file__).resolve()
WORKER_EXCEPTIONS = (
    QualificationError, OSError, ValueError, KeyError, RuntimeError, TypeError,
    ImportError, AttributeError, IndexError, ArithmeticError, LookupError,
)


def _evidence(gate: dict, value: dict, status: str = "passed") -> None:
    gate.update(status=status, evidence_sha256=_hash(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()))
    gate["observation"] = value


def _worker(case: dict, root: Path, output: Path) -> int:
    try:
        import socket
        def denied(*args, **kwargs):
            raise OSError("network disabled for W01 native run")
        socket.create_connection = denied
        socket.socket.connect = denied
        os.environ.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1",
                           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                           "NUMEXPR_NUM_THREADS": "1"})
        result = _run_native(case, root, repeat=True, runner_file=RUNNER_FILE)
        import psutil
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        effective_threads = max([1, *[int(pool["num_threads"]) for pool in pools]])
        try:
            import torch
            effective_threads = max(effective_threads, int(torch.get_num_threads()))
            gpu_available = bool(torch.cuda.is_available())
        except ImportError:
            gpu_available = False
        result.update({"peak_rss_bytes": psutil.Process().memory_info().rss,
                       "native_process_threads": psutil.Process().num_threads(),
                       "effective_compute_threads": effective_threads,
                       "gpu_available": gpu_available,
                       "python_version": platform.python_version(), "status": "passed"})
        output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        return 0
    except WORKER_EXCEPTIONS as exc:
        output.write_text(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                                      "error": str(exc)[:200],
                                      "traceback": traceback.format_exc()[-1200:]}, sort_keys=True) + "\n")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--trust-key", required=True, type=Path)
    parser.add_argument("--trust-key-sha256", required=True)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worker", nargs=2, metavar=("CASE_JSON", "OUT"))
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(json.loads(args.worker[0]), args.checkpoint_root, Path(args.worker[1]))
    try:
        if platform.python_version() != "3.12.3" or sys.prefix == sys.base_prefix:
            raise QualificationError("isolated_python_version_mismatch")
        cfg = Path(sys.prefix) / "pyvenv.cfg"
        if "include-system-site-packages = false" not in cfg.read_text().lower():
            raise QualificationError("system_site_packages_enabled")
        payload = verify_signed_manifest(args.manifest, args.trust_key, args.trust_key_sha256)
        checkpoints = verify_checkpoints(payload, args.checkpoint_root)
        closure_path = args.checkpoint_root / "w01-cert-venv-closure.json"
        closure = json.loads(closure_path.read_text())
        closure_sha = sha256_file(closure_path)
        if closure_sha != payload["cases"][0]["dependency_closure_sha256"]:
            raise QualificationError("dependency_closure_pin_mismatch")
        if closure["python_version"] != platform.python_version() or closure["include_system_site_packages"]:
            raise QualificationError("dependency_closure_environment_mismatch")
        for dist in closure["distributions"]:
            installed = metadata.distribution(dist["name"])
            record = next((f for f in (installed.files or ()) if str(f).endswith("RECORD")), None)
            if installed.version != dist["version"] or record is None:
                raise QualificationError("dependency_closure_mismatch")
            if _hash(installed.locate_file(record).read_bytes()) != dist["record_sha256"]:
                raise QualificationError("dependency_closure_mismatch")
    except (QualificationError, OSError, ValueError, KeyError, StopIteration) as error:
        reason = str(error) if isinstance(error, QualificationError) else "preflight_input_unavailable"
        print(json.dumps({"status": "refused", "reason": reason}, sort_keys=True))
        return 2

    result = new_result(payload, checkpoints)
    result["environment"] = {"python_version": platform.python_version(),
                             "venv": sys.prefix, "dependency_closure_sha256": closure_sha,
                             "network": "disabled", "gpu": "disabled"}
    source_root = RUNNER_FILE.parents[2]
    result["parity_provenance"] = {
        "harness_base_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
            capture_output=True, text=True).stdout.strip(),
        "signed_source_commit_preserved": payload["source_commit"],
        "module_source_sha256": {
            "tabicl_adapter": sha256_file(RUNNER_FILE.parent / "adapters/auditors/tabicl.py"),
            "kumo_adapter": sha256_file(RUNNER_FILE.parent / "adapters/auditors/kumo_tabular.py"),
            "native_qualification": sha256_file(RUNNER_FILE.with_name("native_qualification.py")),
            "native_qualification_runtime": sha256_file(RUNTIME_FILE),
            "run_native_qualification": sha256_file(RUNNER_FILE),
        },
        "comparison_admission": "local identity binding only; does not alter registry status",
    }
    result["cases"] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import psutil
    except ImportError:
        print(json.dumps({"status": "refused", "reason": "resource_observer_unavailable"}))
        return 2
    for case in payload["cases"]:
        gates = {f"Q{i}": {"status": "pending", "evidence_sha256": None} for i in range(10)}
        item = {"model": case["model"], "task": case["task"],
                "recipe_sha256": case["recipe_sha256"],
                "class_order": case["class_order"], "gates": gates}
        _evidence(gates["Q0"], {"signed_admission": True, "role": case["role"]})
        try:
            context, query, recipe = _case_inputs(case, args.checkpoint_root)
            _evidence(gates["Q1"], {"context_sha256": case["context_sha256"],
                                    "query_sha256": case["query_sha256"],
                                    "recipe_sha256": case["recipe_sha256"],
                                    "closure_sha256": closure_sha})
            # The query input is checked to contain feature columns only.
            target_names = {context["target_column"], "y_cls", "y_reg"}
            if any(target_names.intersection(row) for row in query["rows"]):
                raise QualificationError("query_target_leakage")
            _evidence(gates["Q4"], {"query_has_targets": False, "target_column_used_for_fit_only": context["target_column"]})
            worker_out = args.out.with_name(args.out.stem + f"-{case['model']}-{case['task']}.worker.json")
            command = [sys.executable, "-B", "-m", "research.transfer.run_native_qualification",
                       "--manifest", str(args.manifest), "--trust-key", str(args.trust_key),
                       "--trust-key-sha256", args.trust_key_sha256,
                       "--checkpoint-root", str(args.checkpoint_root), "--out", str(args.out),
                       "--worker", json.dumps(case, sort_keys=True), str(worker_out)]
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1", "PYTHONPATH": os.environ.get("PYTHONPATH", ".")})
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, cwd=RUNNER_FILE.parents[2])
            proc = psutil.Process(process.pid)
            start = time.monotonic()
            peak_rss = 0
            failure = None
            while process.poll() is None:
                try:
                    rss = proc.memory_info().rss
                    peak_rss = max(peak_rss, rss)
                except psutil.Error:
                    rss = peak_rss
                if time.monotonic() - start > case["resource_caps"]["wall_seconds"]:
                    failure = "wall_cap_exceeded"
                    process.kill()
                    break
                if peak_rss > case["resource_caps"]["memory_bytes"]:
                    failure = "memory_cap_exceeded"
                    process.kill()
                    break
                time.sleep(0.25)
            stdout, stderr = process.communicate()
            wall = time.monotonic() - start
            if failure:
                raise QualificationError(failure)
            if process.returncode != 0 or not worker_out.exists():
                raise QualificationError("native_execution_failed")
            observed = json.loads(worker_out.read_text())
            if observed.get("status") != "passed":
                raise QualificationError("native_execution_failed")
            observed["peak_rss_bytes"] = max(peak_rss, observed.get("peak_rss_bytes", 0))
            observed["wall_seconds"] = wall
            q2 = {"native_checkpoint_loaded": observed["weights_loaded_strictly"],
                  "checkpoint_sha256": checkpoints[f"{case['model']}:{case['task']}"]["sha256"],
                  "model": case["model"], **observed["q2_parity"]}
            q2_passed = q2["adapter_output_parity"] == "established"
            _evidence(gates["Q2"], q2, "passed" if q2_passed else "failed")
            _evidence(gates["Q3"], {"output_shape": q2["adapter_output_shape"],
                                    "output_kind": observed["output_kind"],
                                    "output_sha256": q2["adapter_output_sha256"],
                                    "finite_and_validated": True})
            _evidence(gates["Q5"], {"query_permutation_max_abs_delta": observed["query_permutation_max_abs_delta"],
                                    "policy": "batch_order_observed"})
            context_ok = observed["context_rows"] == len(context["rows"])
            _evidence(gates["Q6"], {"context_rows_consumed": observed["context_rows"],
                                    "context_rows_admitted": len(context["rows"]),
                                    "all_context_consumed": context_ok}, "passed" if context_ok else "failed")
            replay_ok = observed["seeded_replay_passed"]
            _evidence(gates["Q7"], {"seeded_replay_max_abs_delta": observed["seeded_replay_max_abs_delta"],
                                    "tolerance": observed["tolerance"],
                                    "passed": replay_ok}, "passed" if replay_ok else "failed")
            caps = case["resource_caps"]
            within = (wall <= caps["wall_seconds"] and observed["peak_rss_bytes"] <= caps["memory_bytes"]
                      and observed["effective_compute_threads"] <= caps["threads"]
                      and not observed["gpu_available"])
            _evidence(gates["Q8"], {"wall_seconds": wall,
                                    "peak_rss_bytes": observed["peak_rss_bytes"],
                                    "effective_compute_threads": observed["effective_compute_threads"],
                                    "process_threads": observed["native_process_threads"],
                                    "gpu_available": observed["gpu_available"], "gpu_bytes": 0,
                                    "within_signed_caps": within}, "passed" if within else "failed")
            gates["Q9"] = {"status": "pending_independent_review", "evidence_sha256": None,
                           "reason": "no_independent_review_receipt_supplied"}
            item["native_observation"] = observed
        except WORKER_EXCEPTIONS as error:
            item["execution_error"] = str(error)[:160]
            for gate in ("Q2", "Q3", "Q5", "Q6", "Q7", "Q8"):
                if gates[gate]["status"] == "pending":
                    gates[gate]["status"] = "failed"
                    gates[gate]["reason"] = "native_case_not_completed"
            gates["Q9"] = {"status": "pending_independent_review", "evidence_sha256": None,
                           "reason": "no_independent_review_receipt_supplied"}
        result["cases"].append(item)
        result["status"] = "incomplete"
        parity_complete = len(result["cases"]) == 4 and all(
            case_item.get("gates", {}).get("Q2", {}).get("status") == "passed" and
            case_item.get("gates", {}).get("Q2", {}).get("observation", {}).get("adapter_output_parity") == "established"
            for case_item in result["cases"]
        )
        result["qualification_scope_status"] = (
            "q0_q8_sdk_parity_established_pending_independent_q9_review"
            if parity_complete else "technical_smoke_or_parity_incomplete"
        )
        result["blockers"] = []
        if not parity_complete:
            result["blockers"].append("official_adapter_sdk_output_parity_incomplete")
        if any(case_item.get("gates", {}).get("Q9", {}).get("status") != "passed"
               for case_item in result["cases"]):
            result["blockers"].append("independent_Q9_review_missing")
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        tmp.replace(args.out)
        print(json.dumps({"model": case["model"], "task": case["task"],
                          "status": gates["Q2"]["status"],
                          "wall_seconds": item.get("native_observation", {}).get("wall_seconds"),
                          "peak_rss_bytes": item.get("native_observation", {}).get("peak_rss_bytes")}, sort_keys=True), flush=True)
    print(json.dumps({"status": "incomplete", "case_count": len(result["cases"]),
                      "output": str(args.out), "blockers": result["blockers"]}, sort_keys=True))
    return 3


if __name__ == "__main__":
    sys.exit(main())
