"""Bounded validation-only DOPE neural fit with an immutable resource receipt."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from .fetch_jope import LIMIT, used_bytes
from .manifest import digest
from .score import sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark")
CANDIDATES = {"micro_tvae_4_16", "micro_tvae_8_24", "tiny_mat_16_2_32", "tabsyn",
              "compact_neural_residual_symbolic", "symbolic_autoregressive_residual"}


def gpu_snapshot() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.used,power.draw",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True,
                         text=True, timeout=10).stdout.strip().splitlines()
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                "--format=csv,noheader,nounits"], check=True,
                               capture_output=True, text=True, timeout=10).stdout.strip()
    if not gpu:
        raise ValueError("no GPU available")
    fields = [item.strip() for item in gpu[0].split(",")]
    power = None if fields[4] in ("[N/A]", "N/A") else float(fields[4])
    return {"index": int(fields[0]), "name": fields[1], "free_mib": int(fields[2]),
            "used_mib": int(fields[3]), "power_watts": power,
            "active_processes": bool(processes)}


def run(worker: Path, binary: Path, candidate: str, seed: int,
        target_weight: float, structural_penalty: float, output_root: Path,
        round_lock: Path | None = None) -> dict:
    if candidate not in CANDIDATES or target_weight not in (2.0, 4.0) or structural_penalty not in (0.0, 0.1):
        raise ValueError("trial is outside frozen neural grid")
    if not output_root.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("GPU evidence must stay on benchmark scratch")
    manifest = json.loads((worker / "worker-manifest.json").read_text())
    projection = json.loads((worker / "projection.json").read_text())
    if any(sha256(worker / f"{part}.csv") != manifest["projected_hashes"][part]
           for part in ("train", "validation")) or sha256(worker / "projection.json") != manifest["projection_sha256"]:
        raise ValueError("worker input digest mismatch")
    if used_bytes(ROOT) + 1_000_000_000 > LIMIT:
        raise ValueError("benchmark scratch reservation unavailable")
    gpu = gpu_snapshot()
    if gpu["index"] != 0 or gpu["active_processes"] or gpu["free_mib"] < 17 * 1024:
        raise ValueError("GPU owner or VRAM admission failed")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 16:
        raise ValueError("16 CPU cores unavailable")
    import torch
    torch_lib = str(Path(torch.__file__).parent / "lib")
    round_sha256 = None
    if round_lock is not None:
        locked = json.loads(round_lock.read_text())
        trial = {"dataset": manifest["dataset_id"], "candidate": candidate, "seed": seed,
                 "target_weight": target_weight, "structural_penalty": structural_penalty}
        if (trial not in locked.get("gpu_jobs", [])
                or locked.get("gpu_binary_sha256") != sha256(binary)
                or locked.get("probe_source_sha256") != sha256(Path(__file__))):
            raise ValueError("GPU trial differs from round lock")
        round_sha256 = sha256(round_lock)
    identity = {"dataset": manifest["dataset_id"], "split": manifest["split_hashes"],
                "probe_source_sha256": sha256(Path(__file__)),
                "round_lock_sha256": round_sha256,
                "projection_sha256": manifest["projection_sha256"],
                "train_sha256": manifest["projected_hashes"]["train"],
                "validation_sha256": manifest["projected_hashes"]["validation"],
                "binary_sha256": sha256(binary), "candidate": candidate, "seed": seed,
                "target_weight": target_weight, "structural_penalty": structural_penalty,
                "tier": "l3", "fit_deadline_seconds": 600}
    trial_dir = output_root / digest(identity)
    trial_dir.mkdir(parents=True, exist_ok=True)
    attempt = len(list(trial_dir.glob("attempt-*.json"))) + 1
    prefix = trial_dir / f"attempt-{attempt:04d}"
    model = Path(f"{prefix}.dpk")
    log = Path(f"{prefix}.log")
    command = [str(binary), "compile", "--dataset-dir", str(worker),
               "--task", projection["task"], "--out", str(model),
               "--tier", "l3", "--seed", str(seed), "--candidate", candidate,
               "--neural-target-weight", str(target_weight),
               "--neural-structural-penalty", str(structural_penalty),
               "--deadline-seconds", "600"]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "LD_LIBRARY_PATH": torch_lib + ":" + env.get("LD_LIBRARY_PATH", ""),
                "OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16",
                "MKL_NUM_THREADS": "16", "RAYON_NUM_THREADS": "16"})
    started = time.monotonic()
    last = started
    energy = 0.0
    power_samples = 0
    peak_mib = gpu["used_mib"]
    stopped_for = None
    with log.open("w") as stream:
        process = subprocess.Popen(["timeout", "--signal=KILL", "600", "taskset", "-c",
                                    ",".join(map(str, allowed[:16])), *command],
                                   stdout=stream, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
        while process.poll() is None:
            time.sleep(min(2, max(0, 600 - (time.monotonic() - started))))
            now = time.monotonic()
            try:
                sample = gpu_snapshot()
                peak_mib = max(peak_mib, sample["used_mib"])
                if sample["power_watts"] is not None:
                    energy += sample["power_watts"] * (now - last)
                    power_samples += 1
                if sample["used_mib"] > 16 * 1024:
                    stopped_for = "memory_limit"
            except (OSError, subprocess.SubprocessError, ValueError):
                stopped_for = "resource_monitor_failed"
            last = now
            if time.monotonic() - started >= 600:
                stopped_for = "timeout"
            if stopped_for:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
        return_code = process.wait()
    elapsed = time.monotonic() - started
    status = stopped_for or ("ok" if return_code == 0 and model.is_file() else
                             "timeout" if elapsed >= 600 else "failed")
    byte_failure = re.search(r"smallest observed size: (\d+) bytes", log.read_text())
    receipt = {"format": "dope-benchmark-gpu-research-attempt", "version": 1,
               "identity": identity, "attempt": attempt, "status": status,
               "host": os.uname().nodename, "gpu": gpu,
               "cpu_affinity": allowed[:16], "elapsed_seconds": elapsed,
               "exit_code": return_code, "peak_gpu_used_mib": peak_mib,
               "energy_joules_estimate": energy if power_samples else None,
               "power_samples": power_samples, "log_sha256": sha256(log),
               "failure_code": "byte_cap" if byte_failure else None,
               "encoded_candidate_bytes": int(byte_failure.group(1)) if byte_failure else None,
               "artifact_bytes": model.stat().st_size + (worker / "projection.json").stat().st_size
               if model.is_file() else None,
               "artifact_sha256": sha256(model) if model.is_file() else None,
               "validation_only": True, "ptf_v1": None}
    receipt_path = Path(f"{prefix}.json")
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return {"receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path),
            "status": status, "artifact_bytes": receipt["artifact_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("candidate", choices=sorted(CANDIDATES))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--target-weight", type=float, default=2.0)
    parser.add_argument("--structural-penalty", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "gpu-research-v1")
    parser.add_argument("--round-lock", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.worker, args.binary, args.candidate, args.seed,
                         args.target_weight, args.structural_penalty, args.output_root,
                         args.round_lock),
                     sort_keys=True))


if __name__ == "__main__":
    main()
