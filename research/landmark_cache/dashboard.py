"""Publish aggregate coverage only; private measurements stay outside the dashboard."""
from __future__ import annotations

import json
import time
import os
import re
import subprocess
from pathlib import Path

from .runner import write_private


def summarize(paths):
    runs = [json.loads(Path(path).read_bytes()) for path in paths]
    cells, seen = [], set()
    for run in runs:
        for row in run["cells"]:
            key = row["key"]
            if key in seen:
                continue
            seen.add(key)
            cells.append({k: row[k] for k in ("key", "dataset_id", "state", "host", "elapsed_seconds", "code", "cell_states") if k in row})
    counts = {s: sum(c["state"] == s for c in cells)
              for s in ("complete", "cached", "unsupported", "failed", "overbudget", "interrupted")}
    total = max((r["total_catalog"] for r in runs), default=0)
    scheduled = sum(r["scheduled"] for r in runs)
    return dict(format="dope-landmark-dashboard-v1", updated_unix=time.time(),
                catalog_total=max((r["total_catalog"] for r in runs), default=0),
                scheduled=sum(r["scheduled"] for r in runs), terminal=len(cells), counts=counts,
                pending=sum(r["pending"] for r in runs),
                unassigned=max(0, total - scheduled),
                runs=[{k: r[k] for k in ("host", "shard", "shards", "scheduled", "terminal", "pending", "counts", "elapsed_seconds", "workers", "threads_per_worker", "updated_unix", "plan_sha256")} for r in runs],
                cells=sorted(cells, key=lambda c: c["key"]), official_test_access=False,
                model_training=False, certification="unqualified_private_development_measurements")


def render(paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private(output / "status.json", summarize(paths))
    (output / "index.html").write_text(Path(__file__).with_name("dashboard.html").read_text())


def watch(remotes, paths, output, interval=0):
    """Pull only explicitly named progress snapshots over existing SSH access."""
    output = Path(output)
    private = output / "worker-snapshots"
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    for remote in remotes:
        if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*:/[a-zA-Z0-9_./-]+", remote) is None:
            raise ValueError("invalid_progress_remote")
    while True:
        available = list(paths)
        for index, remote in enumerate(remotes):
            target = private / f"worker-{index}.json"
            tmp = target.with_suffix(".partial")
            try:
                result = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                                         remote, str(tmp)], stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        timeout=20, check=False)
                if result.returncode == 0:
                    json.loads(tmp.read_bytes())
                    os.chmod(tmp, 0o600)
                    os.replace(tmp, target)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass  # Keep the last complete snapshot; worker timestamps expose staleness.
            finally:
                tmp.unlink(missing_ok=True)
            if target.exists():
                available.append(target)
        if available:
            render(available, output)
        if interval == 0:
            return
        if interval < 5:
            raise ValueError("dashboard_interval_must_be_at_least_5_seconds")
        time.sleep(interval)
