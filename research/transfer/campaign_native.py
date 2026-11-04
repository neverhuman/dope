"""Supervised native kernel calls; no model or scientific qualification."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

from research.transfer.campaign_artifacts import read_regular
from research.transfer.campaign_runner import CampaignRunnerError, _require
from research.transfer.evidence import EvidenceError, parse_json


def invoke(argv, root, label, wall_ms, ram_bytes):
    """Transport only; build native-process explicitly before entering measured work."""
    from research.transfer.evidence import canonical_json

    helper = os.environ.get("DOPE_NATIVE_PROCESS")
    if helper is None:
        helper = str(Path(__file__).resolve().parents[2] / "target/debug/native-process")
    request = {"argv": [str(x) for x in argv], "root": str(Path(root).absolute()),
               "label": label, "wall_ms": wall_ms, "ram_bytes": ram_bytes}
    try:
        _require(Path(helper).is_absolute(), "native_process_unavailable")
        result = subprocess.run([helper], input=canonical_json(request),
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        _require(len(result.stdout) <= 4096, "native_process_unavailable")
        value = parse_json(result.stdout)
        if result.returncode != 0:
            code = value.get("error") if type(value) is dict else None
            _require(code in {"native_process_unavailable", "native_output_unavailable",
                              "native_file_unavailable", "native_file_limit"},
                     "native_process_unavailable")
            raise CampaignRunnerError(code) from None
        _require(type(value) is dict and set(value) == {
            "status", "wall_ms", "peak_ram_bytes", "returncode", "rss_measurement",
            "stdout_sha256", "stderr_sha256"}, "native_process_unavailable")
        return value
    except (OSError, subprocess.SubprocessError, EvidenceError):
        raise CampaignRunnerError("native_process_unavailable") from None


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
