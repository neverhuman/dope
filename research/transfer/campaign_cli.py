"""Redacted development CLI; signed native engineering only, no pilot/confirmation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat

from research.transfer.campaign_runner import (
    CampaignRunSpec, CampaignRunnerError, MAX_SPEC_BYTES, NativeDopeCommandPlan, StageJournal,
)


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise CampaignRunnerError("invalid_cli_arguments")


def _load(path, *, native_commands=False):
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
        decoder = NativeDopeCommandPlan if native_commands else CampaignRunSpec
        return decoder.from_bytes(b"".join(chunks))
    except OSError:
        raise CampaignRunnerError("run_spec_unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("command", choices=("plan", "inspect", "native", "pilot", "confirm"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-dir")
    parser.add_argument("--commands")
    for flag in ("execution-plan", "admission", "public-key", "trusted-key-sha256"):
        parser.add_argument("--" + flag)
    try:
        args = parser.parse_args(argv)
        if args.command in {"pilot", "confirm"}:
            raise CampaignRunnerError("native_execution_admission_not_connected")
        if args.commands is not None and args.command != "plan":
            raise CampaignRunnerError("command_plan_requires_plan")
        native_fields = (args.execution_plan, args.admission, args.public_key, args.trusted_key_sha256)
        if args.command == "native":
            if not args.evidence_dir or not all(native_fields):
                raise CampaignRunnerError("native_authorization_required")
        elif any(native_fields):
            raise CampaignRunnerError("native_arguments_require_native")
        spec = _load(args.manifest)
        if args.command == "native":
            from research.transfer.campaign_artifacts import read_regular
            from research.transfer.campaign_execution import NativeExecutionPlan, execute_native
            from research.transfer.evidence import EvidenceError, parse_json
            try:
                plan = NativeExecutionPlan.from_document(parse_json(read_regular(args.execution_plan)))
            except EvidenceError:
                raise CampaignRunnerError("native_plan_invalid") from None
            result = execute_native(spec, plan, Path(args.evidence_dir), admission=args.admission,
                                    public_key=args.public_key,
                                    trusted_key_sha256=args.trusted_key_sha256)
        elif args.command == "plan":
            result = spec.public_summary()
            if args.commands is not None:
                commands = _load(args.commands, native_commands=True)
                commands.verify_binding(spec)
                result["native_commands"] = commands.public_summary()
        else:
            if not args.evidence_dir:
                raise CampaignRunnerError("evidence_directory_required")
            with StageJournal(Path(args.evidence_dir), spec, read_only=True) as journal:
                result = journal.public_summary()
        print(json.dumps({"status": "ok", **result}, sort_keys=True))
        return 0 if result.get("status", "ok") == "ok" else 2
    except CampaignRunnerError as error:
        print(json.dumps({"status": "refused", "reason": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
