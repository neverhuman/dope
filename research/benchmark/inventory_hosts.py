"""Read-only three-host capacity snapshot; sends no AWS credentials to workers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=15).stdout.strip()


def local() -> dict:
    disk = shutil.disk_usage("/home/ubuntu")
    scratch = shutil.disk_usage("/mnt/fast-scratch/dope-benchmark")
    with Path("/proc/meminfo").open() as stream:
        memory = {parts[0].rstrip(":"): int(parts[1]) * 1024
                  for line in stream if (parts := line.split()) and parts[0] in ("MemAvailable:", "MemTotal:")}
    gpu = []
    output = command(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
                      "--format=csv,noheader,nounits"])
    for line in output.splitlines():
        index, name, total, free, driver = (part.strip() for part in line.split(","))
        gpu.append({"index": int(index), "name": name, "memory_total_mib": int(total),
                    "memory_free_mib": int(free), "driver": driver})
    active = command(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                      "--format=csv,noheader,nounits"]).splitlines()
    return {"host": socket.gethostname(), "observed_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_count": os.cpu_count(), "allowed_cpus": sorted(os.sched_getaffinity(0)),
            "load_average": os.getloadavg(),
            "memory": memory, "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
            "scratch_disk": {"total_bytes": scratch.total, "free_bytes": scratch.free},
            "gpus": gpu, "active_gpu_processes": active,
            "per_job_cap": {"gpus": 1, "cpu_cores": 16}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.local:
        print(json.dumps(local(), sort_keys=True))
        return
    source = Path(__file__).read_text()
    hosts = [local()]
    for name in ("xbabe1", "xbabe3"):
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                                 name, "python3", "-", "--local"], input=source,
                                capture_output=True, text=True, timeout=20, check=True)
        hosts.append(json.loads(result.stdout))
    report = {"format": "dope-benchmark-host-inventory", "version": 1,
              "hosts": hosts, "container_digest": None,
              "container_status": "not_pinned; pilot dispatch blocked",
              "aws_credentials_distributed": False}
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
