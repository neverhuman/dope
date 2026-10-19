"""Supervised native kernel calls; no model or scientific qualification."""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import resource
import subprocess
import time

from research.transfer.campaign_artifacts import create_private, file_hash, read_regular
from research.transfer.campaign_runner import CampaignRunnerError, _require
from research.transfer.evidence import EvidenceError, parse_json


def invoke(argv, root, label, wall_ms, ram_bytes):
    """Only this call's new process session may be terminated."""
    import psutil

    stdout, stderr = root / (label + ".stdout"), root / (label + ".stderr")
    create_private(stdout, b"")
    create_private(stderr, b"")
    start = time.monotonic()
    peak, process = 0, None
    environment = {"PATH": os.defpath, "LC_ALL": "C", "RAYON_NUM_THREADS": "1",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    def child_limits():
        priority = os.nice(0)
        if priority < 10:
            os.nice(10 - priority)
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_AS, (ram_bytes, ram_bytes))
        seconds = max(1, math.ceil(wall_ms / 1000) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 << 20, 32 << 20))

    def stop_owned():
        if process is not None and process.poll() is None:
            try:
                _require(os.getpgid(process.pid) == process.pid, "native_process_identity_changed")
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    outcome = "failed"
    try:
        with stdout.open("ab", buffering=0) as out, stderr.open("ab", buffering=0) as err:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       cwd=root, env=environment, start_new_session=True,
                                       preexec_fn=child_limits, close_fds=True)
            observed = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    members = [observed, *observed.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in members if p.is_running())
                    peak = max(peak, rss)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
                if peak > ram_bytes:
                    outcome = "over_budget"
                    stop_owned()
                    break
                if (time.monotonic() - start) * 1000 >= wall_ms:
                    outcome = "timeout"
                    stop_owned()
                    break
                time.sleep(0.05)
            else:
                outcome = "completed" if process.returncode == 0 else "failed"
            os.fsync(out.fileno())
            os.fsync(err.fileno())
        peak = max(peak, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
        elapsed = math.ceil((time.monotonic() - start) * 1000)
        if elapsed > wall_ms and outcome == "completed":
            outcome = "over_budget"
        return {"status": outcome, "wall_ms": elapsed, "peak_ram_bytes": peak,
                "returncode": process.returncode,
                "rss_measurement": "polled_tree_and_lifetime_child_rusage_upper_bound", "stdout_sha256": file_hash(stdout),
                "stderr_sha256": file_hash(stderr)}
    except (OSError, subprocess.SubprocessError):
        raise CampaignRunnerError("native_process_unavailable") from None
    finally:
        stop_owned()


def output_document(root, label):
    try:
        result = parse_json(read_regular(root / (label + ".stdout")))
        _require(type(result) is dict, "native_output_invalid")
        return result
    except EvidenceError:
        raise CampaignRunnerError("native_output_invalid") from None


def native_arguments(binary, stage, plan, root):
    value = plan.document()
    artifact = str(root / "model.raw")
    if stage == "compile":
        return [str(binary), "compile-raw", "--csv", str(root / "fit.csv"),
                "--schema", str(root / "schema.json"), "--out", artifact,
                "--seed", str(value["fit_seed"]), "--candidate", "chow_liu",
                "--max-complete-bytes", str(value["byte_budget"]),
                "--deadline-seconds", str(value["deadline_seconds"])]
    if stage == "reload":
        return [str(binary), "inspect-raw", "--artifact", artifact]
    _require(stage == "sample", "native_stage_unsupported")
    return [str(binary), "sample-raw-isolated", "--artifact", artifact,
            "--rows", str(value["rows"]), "--seed", str(value["draw_seed"])]
