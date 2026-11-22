"""Zero-dependency stdlib host probe for DOPE multi-node training, Arena, and GPU telemetry.

Piped over SSH (`ssh -T xbabeN python3 -u -`) to emit bounded JSONL snapshots every 2 seconds.
Read-only: never modifies training jobs, jsr processes, or files.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


POLL_SECONDS = 2.0
TAIL_TRAIN = 180
TAIL_EVAL = 100
DEFAULT_RUN_ROOTS = ("/home/ubuntu/dope-data/runs", "/home/ubuntu/dope-data/arena")


def _cmd(args: list[str], timeout: float = 2.0) -> str | None:
    try:
        res = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return res.stdout if res.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _float_or_none(raw: str) -> float | None:
    try:
        v = float(raw.strip())
        return v if (v == v and abs(v) < 1e15) else None
    except ValueError:
        return None


def _cmdline(pid: int) -> str:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _proc_ddp_env(pid: int) -> dict[str, object] | None:
    if not sys.platform.startswith("linux"):
        return None
    keys = {b"RANK", b"WORLD_SIZE", b"MASTER_ADDR", b"MASTER_PORT"}
    try:
        entries = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        env = {k: v for e in entries for k, _, v in [e.partition(b"=")] if k in keys}
        rank, ws = int(env[b"RANK"]), int(env[b"WORLD_SIZE"])
        if not (0 <= rank < ws and ws > 1):
            return None
        ident = b":".join((env[b"MASTER_ADDR"], env[b"MASTER_PORT"], env[b"WORLD_SIZE"]))
        return {
            "group": hashlib.sha256(ident).hexdigest()[:16],
            "rank": rank,
            "world_size": ws,
            "master": env[b"MASTER_ADDR"].decode("utf-8", errors="replace"),
            "port": env[b"MASTER_PORT"].decode("utf-8", errors="replace"),
        }
    except (OSError, KeyError, ValueError):
        return None


def _scan_dope_procs() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not Path("/proc").is_dir():
        return out
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        cmd = _cmdline(pid)
        if not cmd:
            continue
        low = cmd.lower()
        role = None
        if "s3_ddp_trainer" in low or "s3_v7_fast_trainer" in low:
            role = "ddp_trainer"
        elif "s3_kpi_evaluator" in low:
            role = "kpi_evaluator"
        elif "dope-arena" in low or "dope_format2_arena" in low:
            role = "rust_arena"
        if role is None:
            continue
        ddp = _proc_ddp_env(pid)
        out.append({"pid": pid, "role": role, "ddp": ddp})
    return out


def _scan_gpus() -> list[dict[str, object]]:
    raw = _cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.free,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        return []
    app_raw = _cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    dope_mib: dict[str, float] = {}
    jsr_mib: dict[str, float] = {}
    if app_raw:
        for line in app_raw.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            uuid, pid_s, mib_s = parts[0], parts[1], parts[2]
            mib = _float_or_none(mib_s) or 0.0
            try:
                cmd = _cmdline(int(pid_s)).lower()
            except ValueError:
                cmd = ""
            if "dope" in cmd or "s3_ddp_trainer" in cmd or "s3_kpi_evaluator" in cmd:
                dope_mib[uuid] = dope_mib.get(uuid, 0.0) + mib
            else:
                jsr_mib[uuid] = jsr_mib.get(uuid, 0.0) + mib

    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        uuid = parts[1]
        gpus.append(
            {
                "index": int(parts[0]) if parts[0].isdigit() else len(gpus),
                "name": parts[2][:64],
                "util_pct": _float_or_none(parts[3]),
                "memory_used_mib": _float_or_none(parts[4]),
                "memory_free_mib": _float_or_none(parts[5]),
                "memory_total_mib": _float_or_none(parts[6]),
                "temperature_c": _float_or_none(parts[7]),
                "power_w": _float_or_none(parts[8]),
                "dope_vram_mib": round(dope_mib.get(uuid, 0.0), 1),
                "other_vram_mib": round(jsr_mib.get(uuid, 0.0), 1),
            }
        )
    return gpus


def _host_resources() -> dict[str, object]:
    load1 = None
    try:
        load1 = round(os.getloadavg()[0], 2)
    except OSError:
        pass
    ram_total_gib, ram_free_gib = None, None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            vals = {}
            for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
                k, _, rest = line.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    vals[k] = float(rest.strip().split()[0]) / (1024.0 * 1024.0)
            ram_total_gib = round(vals["MemTotal"], 1) if "MemTotal" in vals else None
            ram_free_gib = round(vals["MemAvailable"], 1) if "MemAvailable" in vals else None
        except (OSError, ValueError):
            pass
    disk_path = Path("/home/ubuntu/dope-data") if Path("/home/ubuntu/dope-data").exists() else Path.home()
    disk_free_gib = None
    try:
        disk_free_gib = round(shutil.disk_usage(disk_path).free / (1024.0 ** 3), 1)
    except OSError:
        pass
    return {
        "cpus": os.cpu_count() or 0,
        "load1": load1,
        "ram_total_gib": ram_total_gib,
        "ram_free_gib": ram_free_gib,
        "disk_free_gib": disk_free_gib,
    }


def _tail_jsonl(path: Path, limit: int) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    buf: deque[str] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    buf.append(line)
    except OSError:
        return []
    rows = []
    for line in buf:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except ValueError:
            continue
    return rows


def _scan_checkpoints(run_dir: Path) -> list[dict[str, object]]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    out: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    # 1. Promoted best_main_kpi.json first
    best_json = ckpt_dir / "best_main_kpi.json"
    if best_json.is_file():
        try:
            obj = json.loads(best_json.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                obj["checkpoint"] = "best_main_kpi.safetensors"
                obj["role"] = "best_main_kpi"
                out.append(obj)
                seen_keys.add(f"best:{obj.get('step')}")
        except (OSError, ValueError):
            pass
    # 2. Immutable checkpoints/ledger.jsonl (tail 40, newest first)
    ledger = _tail_jsonl(ckpt_dir / "ledger.jsonl", 40)
    for row in reversed(ledger):
        key = f"{row.get('role')}:{row.get('checkpoint')}:{row.get('step')}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(row)
        if len(out) >= 32:
            break
    # 3. Fallback: scan *.safetensors files directly if ledger is absent (e.g., v1.3.1 runs)
    if not out:
        for sf in sorted(ckpt_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
            try:
                st = sf.stat()
                out.append({
                    "checkpoint": sf.name,
                    "role": "best_main_kpi" if "best" in sf.name else ("valley_soup" if "soup" in sf.name else "step_ema"),
                    "bytes": st.st_size,
                    "time": round(st.st_mtime, 3),
                })
            except OSError:
                continue
    return out


def _scan_runs(run_roots: tuple[str, ...]) -> list[dict[str, object]]:
    runs = []
    for root_s in run_roots:
        root = Path(root_s)
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            train_rows = _tail_jsonl(child / "train.jsonl", TAIL_TRAIN)
            eval_rows = _tail_jsonl(child / "eval.jsonl", TAIL_EVAL)
            ckpts = _scan_checkpoints(child)
            if not train_rows and not eval_rows and not ckpts:
                continue
            cfg = {}
            cfg_path = child / "config.json"
            if cfg_path.is_file():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    cfg = {}
            regret = {}
            reg_path = child / "curriculum_regret.json"
            if reg_path.is_file():
                try:
                    regret = json.loads(reg_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    regret = {}
            runs.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "config": cfg,
                    "regret": regret,
                    "checkpoints": ckpts,
                    "train": train_rows,
                    "eval": eval_rows,
                }
            )
    return runs


def collect_snapshot(run_roots: tuple[str, ...] = DEFAULT_RUN_ROOTS) -> dict[str, object]:
    return {
        "host": socket.gethostname().split(".")[0],
        "at": round(time.time(), 3),
        "resources": _host_resources(),
        "gpus": _scan_gpus(),
        "procs": _scan_dope_procs(),
        "runs": _scan_runs(run_roots),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--run-root", action="append", default=[])
    args = ap.parse_args()
    roots = tuple(args.run_root) if args.run_root else DEFAULT_RUN_ROOTS
    while True:
        snap = collect_snapshot(roots)
        sys.stdout.write(json.dumps(snap, separators=(",", ":"), allow_nan=False) + "\n")
        sys.stdout.flush()
        if args.once:
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
