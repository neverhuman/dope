"""Redacted development planning/inspection CLI; native execution stays denied."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat

from research.transfer.campaign_runner import (
    CampaignRunSpec, CampaignRunnerError, MAX_SPEC_BYTES, StageJournal,
)


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise CampaignRunnerError("invalid_cli_arguments")


def _load(path):
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_SPEC_BYTES:
            raise CampaignRunnerError("run_spec_unavailable")
        chunks, total = [], 0
        while total <= MAX_SPEC_BYTES:
            chunk = os.read(descriptor, min(65536, MAX_SPEC_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            chunks.append(chunk)
        return CampaignRunSpec.from_bytes(b"".join(chunks))
    except OSError:
        raise CampaignRunnerError("run_spec_unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("command", choices=("plan", "inspect", "pilot", "confirm"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-dir")
    try:
        args = parser.parse_args(argv)
        if args.command in {"pilot", "confirm"}:
            raise CampaignRunnerError("native_execution_admission_not_connected")
        spec = _load(args.manifest)
        if args.command == "plan":
            result = spec.public_summary()
        else:
            if not args.evidence_dir:
                raise CampaignRunnerError("evidence_directory_required")
            with StageJournal(Path(args.evidence_dir), spec, read_only=True) as journal:
                result = journal.public_summary()
        print(json.dumps({"status": "ok", **result}, sort_keys=True))
        return 0
    except CampaignRunnerError as error:
        print(json.dumps({"status": "refused", "reason": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
