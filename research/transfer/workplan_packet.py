"""Explicit workplan snapshot qualification; portable fixtures never qualify it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
W01_PIN = "3b2910b10fede74882b953e3df4855d74a9891dd5f3f24753df6a7157ae1af49"
EXPECTED_INPUT_PINS = {
    "tasks.json": "d5150648a51bb5971125dfd6fbb81711e3dba5349b3ee1801b3a6c6913d11146",
    "progress.json": "a7f835d4b1c473de6741ba89cfc8f1781ffb2b0e9947e111a47205ca300cec3c",
    "validate_plan.py": "82819a10280756b4c8422580f5ef01d4eee168af9e0dcf896d2631d9683e4671",
}
NOT_RUN = "CLEAN_CHECKOUT_PACKET_ABSENT_NOT_QUALIFIED"


class PacketError(RuntimeError):
    """A typed refusal that contains no caller-supplied values."""


def _safe_is_dir(path: Path | None) -> bool:
    try:
        return path is not None and path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path | None) -> bool:
    try:
        return path is not None and path.is_file()
    except OSError:
        return False


def packet_mode(mode: str | None = None) -> str:
    selected = mode if mode is not None else os.environ.get("DOPE_PACKET_MODE", "auto")
    if selected not in ("auto", "unit", "qualify"):
        raise PacketError("PACKET_MODE_INVALID")
    if os.environ.get("DOPE_REQUIRE_PACKET", "0") not in ("0", "1"):
        raise PacketError("PACKET_REQUIRE_FLAG_INVALID")
    if os.environ.get("DOPE_REQUIRE_PACKET") == "1":
        if selected == "unit":
            raise PacketError("PACKET_MODE_CONFLICT")
        return "qualify"
    return selected


def _regular_bytes(root: Path, name: str, limit: int = 8 * 1024 * 1024) -> bytes:
    path = root / name
    try:
        for part in (path, *path.parents):
            if part.is_symlink():
                raise PacketError("PACKET_SYMLINK_REFUSED")
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise PacketError("PACKET_FILE_INVALID")
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
            if not stat.S_ISREG(before.st_mode) or identity(info) != identity(before):
                raise PacketError("PACKET_FILE_CHANGED")
            data = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        if identity(before) != identity(after) or identity(after) != identity(path.stat(follow_symlinks=False)) or len(data) > limit:
            raise PacketError("PACKET_FILE_CHANGED")
        return data
    except (OSError, ValueError) as exc:
        raise PacketError("PACKET_INPUT_UNAVAILABLE") from exc


def resolve_packet_dir(env_pkt: str | None = None, *, mode: str | None = None) -> Path | None:
    selected = packet_mode(mode)
    if selected == "unit":
        return None
    raw = os.environ.get("DOPE_PACKET_DIR") if env_pkt is None else env_pkt
    if not raw:
        if selected == "qualify":
            raise PacketError("PACKET_REQUIRED_NOT_QUALIFIED")
        return None
    root = Path(raw)
    try:
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise PacketError("PACKET_DIRECTORY_INVALID")
        for name in ("tasks.json", "progress.json", "validate_plan.py"):
            _regular_bytes(root, name)
    except OSError as exc:
        raise PacketError("PACKET_INPUT_UNAVAILABLE") from exc
    return root


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise PacketError("PACKET_JSON_INVALID")
        out[key] = value
    return out


def _json_bytes(data: bytes):
    try:
        return json.loads(data, object_pairs_hook=_unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(PacketError("PACKET_JSON_INVALID")))
    except (ValueError, UnicodeError) as exc:
        raise PacketError("PACKET_JSON_INVALID") from exc


def assert_canonical_acceptance_truth(root: Path) -> dict:
    """Pin the reported provisional snapshot; this supplies no scientific grant."""
    captured = {name: _regular_bytes(root, name) for name in ("tasks.json", "progress.json")}
    if any(hashlib.sha256(data).hexdigest() != EXPECTED_INPUT_PINS[name] for name, data in captured.items()):
        raise PacketError("PACKET_SNAPSHOT_PIN_CHANGED")
    tasks_doc = _json_bytes(captured["tasks.json"])
    progress = _json_bytes(captured["progress.json"])
    try:
        if not isinstance(tasks_doc, dict) or not isinstance(progress, dict):
            raise PacketError("PACKET_SCHEMA_INVALID")
        tasks = tasks_doc["tasks"]
        if not isinstance(tasks, list) or len(tasks) != 65:
            raise PacketError("PACKET_TASK_IDENTITIES_INVALID")
        if any(not isinstance(t, dict) or type(t.get("id")) is not str or type(t.get("status")) is not str for t in tasks):
            raise PacketError("PACKET_SCHEMA_INVALID")
        if len({task["id"] for task in tasks}) != 65:
            raise PacketError("PACKET_TASK_IDENTITIES_INVALID")
        expected = {"done": 61, "blocked": 4, "in_progress": 0}
        summaries = [tasks_doc["summary_counts"], progress["counts"], progress["status_summary"]]
        expected_summaries = [expected] + [{**expected, "total": 65}] * 2
        # Every other schema state must be zero; exact pins bind its field names.
        for actual, wanted in zip(summaries, expected_summaries):
            if (not isinstance(actual, dict) or any(type(v) is not int for v in actual.values())
                    or {k: v for k, v in actual.items() if k in expected or v != 0} != wanted):
                raise PacketError("PACKET_REPORTED_COUNTS_CHANGED")
        blocked = ["E03", "E04", "E05", "E06"]
        if (sorted(t["status"] for t in tasks) != ["blocked"] * 4 + ["done"] * 61
                or progress["blocked_tasks"] != blocked
                or sorted(t["id"] for t in tasks if t["status"] == "blocked") != blocked):
            raise PacketError("PACKET_TASK_STATES_CHANGED")
        w01 = next(t for t in tasks if t["id"] == "W01")
        if w01["verification_receipt_sha256"] != W01_PIN:
            raise PacketError("PACKET_W01_BINDING_CHANGED")
        binding = "verify/main_241746ea/W01.json"
        captured[binding] = _regular_bytes(root, binding)
        if hashlib.sha256(captured[binding]).hexdigest() != W01_PIN:
            raise PacketError("PACKET_W01_BINDING_CHANGED")
    except (KeyError, TypeError, StopIteration) as exc:
        raise PacketError("PACKET_SCHEMA_INVALID") from exc
    if any(_regular_bytes(root, name) != data for name, data in captured.items()):
        raise PacketError("PACKET_CHANGED_DURING_VALIDATION")
    return {"tasks_sha256": hashlib.sha256(captured["tasks.json"]).hexdigest(),
            "progress_sha256": hashlib.sha256(captured["progress.json"]).hexdigest(),
            "W01_sha256": W01_PIN, "validator_sha256": EXPECTED_INPUT_PINS["validate_plan.py"]}


def run_validate_plan() -> str:
    root = resolve_packet_dir()
    if root is None:
        return NOT_RUN
    before = {n: hashlib.sha256(_regular_bytes(root, n)).hexdigest()
              for n in ("tasks.json", "progress.json", "validate_plan.py")}
    if before != EXPECTED_INPUT_PINS:
        raise PacketError("PACKET_SNAPSHOT_OR_VALIDATOR_PIN_CHANGED")
    try:
        result = subprocess.run([sys.executable, str(root / "validate_plan.py")], cwd=root,
                                capture_output=True, timeout=60, check=False)
        if result.returncode:
            raise PacketError("PACKET_VALIDATOR_FAILED")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PacketError("PACKET_VALIDATOR_UNAVAILABLE") from exc
    after = {n: hashlib.sha256(_regular_bytes(root, n)).hexdigest() for n in before}
    if before != after:
        raise PacketError("PACKET_CHANGED_DURING_VALIDATION")
    return "PACKET_VALIDATOR_PASSED"


def check_dashboard_200(path: str = "/") -> str:
    if resolve_packet_dir() is None:
        return "DOPE_DASHBOARD_NOT_RUN_NOT_QUALIFIED"
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:18884{path}", timeout=3) as response:
            if response.status != 200:
                raise PacketError("PACKET_DASHBOARD_UNAVAILABLE")
            return response.read().decode("utf-8", errors="replace")
    except (OSError, ValueError) as exc:
        raise PacketError("PACKET_DASHBOARD_UNAVAILABLE") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "unit", "qualify"), default=None)
    parser.add_argument("--require-packet", action="store_true")
    args = parser.parse_args(argv)
    report = {"format": "dope-workplan-packet-qualification-v1", "status": "not_run",
              "scope": "reported_provisional_snapshot_only", "release_authorized": False,
              "scientific_qualification": "not_established"}
    try:
        mode = "qualify" if args.require_packet else args.mode
        root = resolve_packet_dir(mode=mode)
        if root is None:
            report["reason"] = NOT_RUN
        else:
            report["input_hashes"] = assert_canonical_acceptance_truth(root)
            # Use the same explicit mode for the legacy validator path.
            previous = os.environ.get("DOPE_PACKET_MODE")
            os.environ["DOPE_PACKET_MODE"] = "qualify"
            try:
                run_validate_plan()
                if report["input_hashes"] != assert_canonical_acceptance_truth(root):
                    raise PacketError("PACKET_CHANGED_DURING_VALIDATION")
            finally:
                if previous is None:
                    os.environ.pop("DOPE_PACKET_MODE", None)
                else:
                    os.environ["DOPE_PACKET_MODE"] = previous
            report["status"] = "qualified_snapshot"
            report["reason"] = "REPORTED_PROVISIONAL_PACKET_VERIFIED"
    except PacketError as exc:
        report["reason"] = str(exc)
        print(json.dumps(report, sort_keys=True))
        return 2
    report["module_file"] = __file__
    report["module_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
