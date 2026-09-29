#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PYTHON'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def main() -> None:
    path = Path("target/jankurai/repo-score.json")
    if not path.is_file():
        return
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("auditor_version") != "1.7.1":
        raise SystemExit("audit receipt requires pinned Jankurai v1.7.1")
    destination = path.parent
    queue = {
        "format": "dope-audit-repair-queue",
        "version": 1,
        "report_fingerprint": report["report_fingerprint"],
        "findings": [
            {
                "fingerprint": finding["fingerprint"],
                "check_id": finding["check_id"],
                "severity": finding["severity"],
                "path": finding["path"],
                "repair_hint": finding["agent_fix"],
                "rerun_command": finding.get("rerun_command") or "just score",
            }
            for finding in report["findings"]
        ],
    }
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    attestation = {
        "format": "dope-audit-report-attestation",
        "version": 1,
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "report_fingerprint": report["report_fingerprint"],
        "auditor_version": report["auditor_version"],
        "source_commit": source_commit,
        "score": report["score"],
        "decision": report["decision"]["status"],
        "signature_kind": "none",
    }
    for name, value in (("repair-queue.json", queue), ("report-attestation.json", attestation)):
        (destination / name).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


main()
PYTHON
