"""How one block B operation ended, read from its attempt directory (tabsyn_seeds_v2.py).

An attempt directory holds the controller's request and the worker's receipts.
Beside it, outside the worker's working directory, the controller keeps its own
record of the spawn: the request digest, the boot id, and the deadline before
the spawn, then the worker's pid and start ticks after it. A cause the
controller confirms rests on that record, never on a file the worker could
rewrite. A receipt counts only once it parses. Every status except a confirmed
infrastructure failure on the first attempt is final, and no attempt key is
reused.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

from research.benchmark.review_fixes import tabsyn_seeds_worker as W

# Written when the close passed before the spawn; no worker ran.
REFUSED = "spawn.refused.receipt.json"
LISTED_CAUSES = ("host_reboot", "foreign_gpu_owner", "storage_refusal", "runtime_lock_mismatch")
GRACE_SECONDS = 5


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def read_receipt(path: Path) -> dict | None:
    """A receipt that exists and parses as a JSON object; a missing or half-written file is None."""
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def controller_record(attempt: Path) -> Path:
    """The controller's own record of one attempt, beside it and outside the worker's working directory."""
    return attempt.parent / f"{attempt.name}.controller.json"


def attempt_number(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[1])


def number(attempt: Path) -> int:
    """An attempt's key: N in attempt-N/fit, or N in attempt-M/sample-N."""
    return attempt_number(attempt.parent if attempt.name == "fit" else attempt)


def attempts(cell: Path, mode: str) -> list[Path]:
    """Every attempt directory of one mode in a cell, oldest first."""
    pattern = "attempt-*/fit" if mode == "fit" else "attempt-*/sample-*"
    found = [path for path in cell.glob(pattern) if path.is_dir()]
    return sorted(found, key=lambda path: (number(path), str(path)))


def next_number(cell: Path, mode: str, previous: dict | None) -> int:
    """One past both the recorded attempt and every attempt directory present, so no key is reused."""
    return 1 + max([previous["attempts"] if previous else 0] + [number(path) for path in attempts(cell, mode)])


def failure_receipt(attempt: Path, mode: str) -> dict | None:
    for name in (f"{mode}.failure.receipt.json", "request.failure.receipt.json"):
        receipt = read_receipt(attempt / name)
        if receipt is not None:
            return receipt
    return None


def outcome(attempt: Path, mode: str, code: int | None, confirm) -> tuple[str, str | None]:
    """Status of one operation from its exit code and receipts.

    A retryable infrastructure failure needs a predeclared cause that ``confirm``
    establishes on the controller's side; the worker's receipt alone is a claim.
    An operation that leaves no final receipt in the same boot is unknown and final.
    """
    if (attempt / REFUSED).is_file():
        return "not_started", "campaign_close"
    if code in (0, None) and read_receipt(attempt / f"{mode}.complete.receipt.json") is not None:
        return "ok", None
    if code == 124:
        return "timeout", "wall"
    receipt = failure_receipt(attempt, mode)
    if receipt is not None:
        if receipt.get("exception_type") == "TimeoutError":
            return "timeout", "deadline"
        cause = receipt.get("cause")
        if cause in LISTED_CAUSES and confirm(cause, attempt):
            return "infrastructure_failure", cause
        return "failed", receipt.get("exception_type")
    if confirm("host_reboot", attempt):
        return "infrastructure_failure", "host_reboot"
    return "unknown", "no_final_receipt"


def needs_attempt(record: dict | None) -> bool:
    """Run when absent; one retry after a confirmed infrastructure failure; every other status is final."""
    if record is None:
        return True
    return record["status"] == "infrastructure_failure" and record["attempts"] < 2


def sample_cells(row: dict, cell: Path, sizes: list[int], seeds: list[int]) -> list[dict]:
    return [{"seed": seed, "size": size, "rows": row["train_rows"] * size,
             "path": str(cell / f"sample-{seed}-{size}n.csv")} for size in sizes for seed in seeds]


def sample_status(attempt: Path, cells: list[dict], operation: str, code: int | None) -> list[dict]:
    """Per-cell status from the worker's per-cell receipt; a cell without one takes the operation's status.

    A deadline cut, whether the worker's own TimeoutError or the controller's kill,
    leaves its unfinished cells as timeouts, not failures.
    """
    unfinished = "unknown" if operation == "ok" else operation
    statuses = []
    for item in cells:
        receipt = read_receipt(attempt / W.cell_receipt_name(item))
        found = receipt or {**item, "status": unfinished, "reason": f"exit_{code}", "synthetic_sha256": None}
        statuses.append({key: value for key, value in found.items() if key != "path"})
    return statuses


def _alive(pid: int, ticks: str) -> bool:
    """The same process is still running: its start ticks match and it has not exited into a zombie."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return False
    return len(fields) > 19 and fields[0] not in ("Z", "X") and fields[19] == ticks


def _stop(pid: int, ticks: str, sleep) -> None:
    for sent in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sent)
        except ProcessLookupError:
            return
        for _ in range(GRACE_SECONDS * 10):
            if not _alive(pid, ticks):
                return
            sleep(0.1)


def settle(attempt: Path, boot: str, clock=time.monotonic, sleep=time.sleep) -> int | None:
    """Wait for a worker a stopped controller left running; stop its group once its deadline has passed.

    The controller's record names the worker; the worker's own start receipt stands in
    only when the controller stopped before recording the pid. Returns 124 when the
    worker had to be stopped, otherwise None: a process this controller did not start
    has no exit code it can read.
    """
    record = read_receipt(controller_record(attempt)) or {}
    started = read_receipt(attempt / "worker.before-imports.receipt.json") or {}
    named = record if record.get("pid") else started
    pid, ticks = named.get("pid"), named.get("start_ticks")
    if not isinstance(pid, int) or not isinstance(ticks, str) or record.get("boot_id", boot) != boot:
        return None
    deadline = record.get("deadline_monotonic", clock()) + GRACE_SECONDS
    while _alive(pid, ticks) and clock() < deadline:
        sleep(1)
    if not _alive(pid, ticks):
        return None
    _stop(pid, ticks, sleep)
    return 124
