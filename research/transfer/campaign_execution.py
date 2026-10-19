"""Signed native development slice, not an E03 pipeline or confirmation runner.

Prepare, compile, reload and isolated sampling execute real native code.
Learners, drivers, Rust metric receipt and report stages remain separate work.
Old v1 command plans remain read-only; their public-facade argv is not executed.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import resource
from pathlib import Path
import subprocess
import tempfile
import time

from research.transfer.campaign_artifacts import (
    MAX_BINARY, create_private, directory_pin, file_hash, private_root, read_regular, snapshot,
    sync_artifact,
)
from research.transfer.campaign_native import invoke, native_arguments, output_document
from research.transfer.campaign_runner import (
    CampaignRunSpec, CampaignRunnerError, StageJournal, _command_path, _hash, _integer, _require,
)
from research.transfer.evidence import EvidenceError, canonical_json, digest, parse_json

FORMAT = "dope-development-native-execution-v1"
SCOPE = "native_engineering_stages_only"
NATIVE_STAGES = ("prepare", "compile", "reload", "sample")


@dataclass(frozen=True)
class NativeExecutionPlan:
    _encoded: bytes

    @classmethod
    def from_document(cls, value):
        try:
            raw = canonical_json(value)
            value = parse_json(raw)
        except (EvidenceError, UnicodeError):
            raise CampaignRunnerError("native_plan_invalid") from None
        fields = {"format", "version", "cell_id", "binary", "binary_sha256", "csv", "csv_sha256",
                  "schema", "schema_sha256", "fit_seed", "draw_seed", "rows",
                  "byte_budget", "deadline_seconds", "stage_wall_ms"}
        _require(len(raw) <= 1 << 20 and type(value) is dict and set(value) == fields,
                 "native_plan_invalid")
        _require(value["format"] == FORMAT and type(value["version"]) is int
                 and value["version"] == 1, "native_plan_invalid")
        _require(all(_hash(value[k]) for k in ("cell_id", "binary_sha256", "csv_sha256",
                                               "schema_sha256")), "native_plan_pin_invalid")
        _require(all(_command_path(value[k]) for k in ("binary", "csv", "schema"))
                 and Path(value["binary"]).name == "dope-kernel"
                 and len({value[k] for k in ("binary", "csv", "schema")}) == 3,
                 "native_plan_path_invalid")
        _require(all(_integer(value[k]) for k in ("fit_seed", "draw_seed"))
                 and all(_integer(value[k], 1) for k in ("rows", "byte_budget", "deadline_seconds"))
                 and value["deadline_seconds"] <= 60 and value["rows"] <= 4096,
                 "native_plan_limits_invalid")
        limits = value["stage_wall_ms"]
        _require(type(limits) is dict and set(limits) == set(NATIVE_STAGES)
                 and all(_integer(v, 1) and v <= 120000 for v in limits.values()),
                 "native_plan_limits_invalid")
        _require(limits["compile"] >= value["deadline_seconds"] * 1000,
                 "native_plan_limits_invalid")
        return cls(raw)

    def document(self):
        return parse_json(NativeExecutionPlan.from_document(parse_json(self._encoded))._encoded)

    @property
    def sha256(self):
        return digest(self.document())

    @property
    def input_sha256(self):
        value = self.document()
        return digest({k: value[k] for k in ("csv_sha256", "schema_sha256")})

    def verify_binding(self, spec):
        spec = CampaignRunSpec.from_bytes(spec._encoded)
        cells = {cell["cell_id"]: cell for cell in spec.document()["cells"]}
        value = self.document()
        _require(len(cells) == 1 and value["cell_id"] in cells, "native_plan_cell_mismatch")
        cell = cells[value["cell_id"]]
        _require(cell["command_sha256"] == self.sha256
                 and cell["input_sha256"] == self.input_sha256, "native_plan_binding_mismatch")
        _require(sum(value["stage_wall_ms"].values()) <= spec.document()["budget"]["wall_ms"],
                 "native_plan_budget_mismatch")


def authorization(path, public_key, trusted_key_sha256, spec, plan, expected_directory_sha256):
    """Out-of-band key pin; a key carried only in the envelope has no authority."""
    try:
        key_bytes = read_regular(public_key)
        _require(_hash(trusted_key_sha256) and hashlib.sha256(key_bytes).hexdigest() == trusted_key_sha256,
                 "native_authorization_key_mismatch")
        envelope = parse_json(read_regular(path))
        _require(type(envelope) is dict and set(envelope) == {"payload", "signature_ed25519_base64"},
                 "native_authorization_invalid")
        payload = envelope["payload"]
        fields = {"scope", "phase", "scientific_admission", "run_spec_sha256",
                  "execution_plan_sha256", "rights_admission_sha256", "resource_lease_sha256",
                  "evidence_directory_sha256"}
        _require(type(payload) is dict and set(payload) == fields
                 and payload["scope"] == SCOPE and payload["phase"] == "development"
                 and payload["scientific_admission"] is False
                 and payload["run_spec_sha256"] == spec.sha256
                 and payload["execution_plan_sha256"] == plan.sha256
                 and payload["evidence_directory_sha256"] == expected_directory_sha256
                 and all(_hash(payload[k]) for k in ("rights_admission_sha256", "resource_lease_sha256")),
                 "native_authorization_invalid")
        signature = base64.b64decode(envelope["signature_ed25519_base64"], validate=True)
        _require(len(signature) == 64, "native_authorization_invalid")
        with tempfile.TemporaryDirectory(prefix="dope-native-auth-") as directory:
            message, sig = Path(directory) / "message", Path(directory) / "signature"
            verify_key = Path(directory) / "public.pem"
            create_private(verify_key, key_bytes)
            create_private(message, canonical_json(payload))
            create_private(sig, signature)
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(verify_key),
                 "-rawin", "-in", str(message), "-sigfile", str(sig)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=15, check=False)
        _require(result.returncode == 0, "native_authorization_invalid")
        return {"payload_sha256": digest(payload), "trusted_key_sha256": trusted_key_sha256}
    except (EvidenceError, ValueError, TypeError, OSError, subprocess.SubprocessError):
        raise CampaignRunnerError("native_authorization_invalid") from None


def execute_native(spec, plan, evidence_dir, *, admission, public_key, trusted_key_sha256):
    spec = CampaignRunSpec.from_bytes(spec._encoded)
    plan = NativeExecutionPlan.from_document(plan.document())
    plan.verify_binding(spec)
    root = private_root(evidence_dir)
    auth = authorization(admission, public_key, trusted_key_sha256, spec, plan, directory_pin(root))
    value, limits = plan.document(), spec.document()["budget"]
    cell = value["cell_id"]
    _require(not (root / "native-execution.json").exists(), "native_execution_already_attempted")
    receipt = {"format": FORMAT, "scope": SCOPE, "plan_sha256": plan.sha256,
               "run_spec_sha256": spec.sha256, "authorization": auth, "stages": [],
               "source_commit_declared": spec.document()["source_commit"],
               "source_tree_declared": spec.document()["source_tree"],
               "source_execution_qualification": "not_established",
               "scientific_qualification": "unmeasured", "release_authorization": "not_authorized"}
    processes, artifact_pin = 0, None
    with StageJournal(root, spec, create=True) as journal:
        for stage in NATIVE_STAGES:
            reservation = value["stage_wall_ms"][stage]
            journal.begin(cell, stage, reservation)
            started = time.monotonic()
            output, observed = None, {}
            try:
                if stage == "prepare":
                    output = {
                        "binary": snapshot(value["binary"], root / "dope-kernel",
                                           value["binary_sha256"], executable=True),
                        "csv": snapshot(value["csv"], root / "fit.csv", value["csv_sha256"]),
                        "schema": snapshot(value["schema"], root / "schema.json", value["schema_sha256"])}
                    observed = invoke([str(root / "dope-kernel"), "doctor"], root, "doctor",
                                      reservation, limits["peak_ram_bytes"])
                    processes += 1
                    if observed["status"] == "completed":
                        doctor = output_document(root, "doctor")
                        _require(doctor.get("name") == "dope-kernel" and
                                 doctor.get("implementation") == "native_rust" and
                                 doctor.get("gpu_training") is False and
                                 doctor.get("executable_sha256") == value["binary_sha256"],
                                 "native_identity_mismatch")
                        output["doctor"] = doctor
                    observed["wall_ms"] = max(observed["wall_ms"],
                        int((time.monotonic() - started) * 1000))
                    observed["peak_ram_bytes"] = max(observed["peak_ram_bytes"],
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                    observed["rss_measurement"] = "process_lifetime_rusage_upper_bound"
                else:
                    _require(file_hash(root / "dope-kernel", MAX_BINARY) == value["binary_sha256"],
                             "native_binary_changed")
                    if stage == "compile":
                        _require(file_hash(root / "fit.csv") == value["csv_sha256"] and
                                 file_hash(root / "schema.json") == value["schema_sha256"],
                                 "native_input_changed")
                    else:
                        _require(artifact_pin is not None and
                                 file_hash(root / "model.raw") == artifact_pin,
                                 "native_artifact_changed")
                    observed = invoke(native_arguments(root / "dope-kernel", stage, plan, root),
                                      root, stage, reservation, limits["peak_ram_bytes"])
                    processes += 1
                    if observed["status"] == "completed":
                        output = output_document(root, stage)
                        _require(type(output) is dict, "native_output_invalid")
                        if stage == "compile":
                            _require((root / "model.raw").stat().st_size <= value["byte_budget"],
                                     "native_artifact_over_budget")
                            sync_artifact(root / "model.raw")
                            artifact_pin = file_hash(root / "model.raw")
                        if stage == "reload":
                            _require(output.get("accounting", {}).get("complete_sampling_bytes") ==
                                     (root / "model.raw").stat().st_size,
                                     "native_reload_proof_missing")
                        if stage == "sample":
                            _require(_hash(output.get("sample_sha256")) and
                                     output.get("filesystem_network_exec") == "denied" and
                                     output.get("inherited_source_fds") == "closed" and
                                     output.get("rows") == value["rows"] and
                                     output.get("seed") == value["draw_seed"],
                                     "native_sampling_proof_missing")
                status = observed["status"]
                if observed["wall_ms"] > reservation or observed["peak_ram_bytes"] > limits["peak_ram_bytes"]:
                    status = "over_budget"
                row = {"stage": stage, **observed, "native_status": observed["status"], "status": status,
                       "output": output, "artifact_sha256":
                       file_hash(root / "model.raw") if (root / "model.raw").is_file() else None}
            except (CampaignRunnerError, OSError, EvidenceError):
                status = "over_budget" if (observed.get("wall_ms", 0) > reservation or
                    observed.get("peak_ram_bytes", 0) > limits["peak_ram_bytes"]) else "failed"
                row = {"stage": stage, **observed, "native_status": observed.get("status"),
                       "status": status, "reason": "native_stage_failed"}
            # A journal/fsync failure must escape: never double-finish or retry it.
            journal.finish(cell, stage, status, observed_wall_ms=row.get("wall_ms"),
                           peak_ram_bytes=row.get("peak_ram_bytes"),
                           output_sha256=digest(row) if status == "completed" else None)
            receipt["stages"].append(row)
            if status != "completed":
                break
        receipt["journal_sha256"] = file_hash(root / "development-stages.jsonl", 64 << 20)
        receipt["native_processes_executed"] = processes
        receipt["pipeline_complete"] = False
        receipt["missing_pipeline_stages"] = ["learners", "drivers", "receipt", "report"]
        create_private(root / "native-execution.json", canonical_json(receipt) + b"\n")
    return {"status": "ok" if len(receipt["stages"]) == 4 and
            all(r["status"] == "completed" for r in receipt["stages"]) else "failed",
            "completion_kind": "native_engineering_slice_only", "worker_executed": processes > 0,
            "native_processes_executed": processes, "pipeline_complete": False,
            "scientific_qualification": "unmeasured", "release_authorization": "not_authorized"}
