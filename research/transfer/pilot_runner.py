"""Signed 8-stage pilot execution bridge; development/pilot only, no release authority."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import subprocess
import tempfile
import time

from research.transfer.campaign_admission import (
    GATES, PilotContractError, PilotExecutionPlan, _identifier, inspect_admission_contract, seed_uint64,
)
from research.transfer.campaign_artifacts import (
    MAX_BINARY, MAX_INPUT, create_private, directory_pin, file_hash,
    private_root, read_regular, sync_artifact,
)
from research.transfer.campaign_native import invoke, native_arguments
from research.transfer.campaign_runner import (
    CampaignRunSpec, CampaignRunnerError, STAGES, StageJournal, _command_path, _require,
)
from research.transfer.evidence import EvidenceError, canonical_json, digest, parse_json
from .pilot_cell_plan import (
    FORMAT, COMPARATORS, SCALES, LEARNERS, CONTEXTS, LEARNER_PATHS,
    PilotCellExecutionPlan, _hex, _seed,
)

RECEIPT_FORMAT = "dope-pilot-stage-execution-receipt"
SCOPE = "pilot_full_8stage_execution"
# Worker response shape checks do not establish these implementations.
PIPELINE_PREREQUISITES = (
    "real_control_artifact_routes_and_complete_byte_charges",
    "materialized_primary_and_4n_draw_handoff",
    "bound_learner_driver_metric_runtime_implementation",
)


def _verify_pilot_authorization(
    root, spec, pilot_plan, admission_contract, cell_plan, admission, public_key, trusted_key_sha256,
):
    contract_sha256, expected_dir_pin = digest(admission_contract), directory_pin(root)
    try:
        key_bytes = read_regular(public_key)
        _require(
            type(trusted_key_sha256) is str and len(trusted_key_sha256) == 64
            and hashlib.sha256(key_bytes).hexdigest() == trusted_key_sha256,
            "pilot_authorization_key_mismatch",
        )
        envelope = parse_json(read_regular(admission))
        _require(type(envelope) is dict and set(envelope) == {"payload", "signature_ed25519_base64"}, "pilot_authorization_invalid")
        payload = envelope["payload"]
        fields = {
            "scope", "phase", "scientific_admission", "run_spec_sha256",
            "pilot_execution_plan_sha256", "pilot_admission_contract_sha256",
            "cell_execution_plan_sha256", "rights_admission_sha256",
            "resource_lease_sha256", "evidence_directory_sha256",
        }
        _require(
            type(payload) is dict and set(payload) == fields and payload["scope"] == SCOPE
            and payload["phase"] == "development" and payload["scientific_admission"] is False
            and payload["run_spec_sha256"] == spec.sha256
            and payload["pilot_execution_plan_sha256"] == pilot_plan.sha256
            and payload["pilot_admission_contract_sha256"] == contract_sha256
            and payload["cell_execution_plan_sha256"] == cell_plan.sha256
            and payload["evidence_directory_sha256"] == expected_dir_pin,
            "pilot_authorization_invalid",
        )
        _hex(payload["rights_admission_sha256"], 64, "pilot_authorization_invalid")
        _hex(payload["resource_lease_sha256"], 64, "pilot_authorization_invalid")
        signature = base64.b64decode(envelope["signature_ed25519_base64"], validate=True)
        _require(len(signature) == 64, "pilot_authorization_invalid")
        with tempfile.TemporaryDirectory(prefix="dope-pilot-auth-") as directory:
            message, sig, verify_key = Path(directory) / "message", Path(directory) / "signature", Path(directory) / "public.pem"
            create_private(verify_key, key_bytes)
            create_private(message, canonical_json(payload))
            create_private(sig, signature)
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(verify_key), "-rawin", "-in", str(message), "-sigfile", str(sig)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False,
            )
        _require(result.returncode == 0, "pilot_authorization_invalid")
        return {
            "trusted_key_sha256": trusted_key_sha256, "admission_sha256": digest(payload),
            "pilot_admission_contract_sha256": contract_sha256,
            "rights_admission_sha256": payload["rights_admission_sha256"],
            "resource_lease_sha256": payload["resource_lease_sha256"],
        }
    except CampaignRunnerError:
        raise
    except (EvidenceError, ValueError, TypeError, OSError, subprocess.SubprocessError):
        raise CampaignRunnerError("pilot_authorization_invalid") from None


def _count_csv_rows(raw: bytes) -> int:
    return max(0, len([line for line in raw.splitlines() if line.strip()]) - 1)


def _snapshot_pilot_inputs(root, doc):
    for name in ("dope-kernel", "pilot-worker", "fit.csv", "selection.csv", "query.csv", "schema.json",
                 "model.raw", "private-learners.json", "private-drivers.json", "stage-receipt.json", "stage-report.json"):
        _require(not (root / name).exists(), "pilot_workspace_not_clean")
    blobs = {k: read_regular(doc[k], MAX_BINARY if "binary" in k else MAX_INPUT)
             for k in ("binary", "worker_binary", "csv", "selection_csv", "query_csv", "schema")}
    _require(all(hashlib.sha256(blobs[k]).hexdigest() == doc[k + "_sha256"] for k in blobs), "pilot_input_hash_mismatch")
    _require(
        _count_csv_rows(blobs["csv"]) >= doc["fit_rows"]
        and _count_csv_rows(blobs["selection_csv"]) >= doc["selection_rows"]
        and _count_csv_rows(blobs["query_csv"]) >= doc["query_rows"],
        "pilot_input_rows_insufficient",
    )
    paths = {
        "binary": root / "dope-kernel", "worker": root / "pilot-worker",
        "csv": root / "fit.csv", "selection_csv": root / "selection.csv",
        "query_csv": root / "query.csv", "schema": root / "schema.json",
        "artifact": root / "model.raw", "learners": root / "private-learners.json",
        "drivers": root / "private-drivers.json", "receipt": root / "stage-receipt.json",
        "report": root / "stage-report.json",
    }
    create_private(paths["binary"], blobs["binary"], 0o500)
    create_private(paths["worker"], blobs["worker_binary"], 0o500)
    for key in ("csv", "selection_csv", "query_csv", "schema"):
        create_private(paths[key], blobs[key], 0o400)
    return paths


def _verify_stage_payload(label, payload, doc, artifact_sha, artifact_size, draw_manifest_sha256=None):
    _require(type(payload) is dict, "pilot_stage_output_invalid")
    if label == "doctor":
        _require(
            payload.get("name") == "dope-kernel" and payload.get("implementation") == "native_rust"
            and payload.get("gpu_training") is False and payload.get("executable_sha256") == doc["binary_sha256"],
            "pilot_stage_output_invalid",
        )
    elif label == "compile":
        _require(0 < artifact_size <= doc["byte_budget"], "pilot_stage_output_invalid")
        if doc["comparator"] != "dope":
            _require(
                payload.get("comparator") == doc["comparator"] and payload.get("candidate") == doc["candidate"],
                "pilot_stage_output_invalid",
            )
    elif label == "reload":
        accounting = payload.get("accounting")
        _require(
            type(accounting) is dict and accounting.get("complete_sampling_bytes") == artifact_size
            and 0 < artifact_size <= doc["byte_budget"],
            "pilot_stage_output_invalid",
        )
        if doc["comparator"] != "dope":
            _require(payload.get("comparator") == doc["comparator"], "pilot_stage_output_invalid")
    elif label == "sample":
        _hex(payload.get("sample_sha256"), 64, "pilot_stage_output_invalid")
        _require(
            payload.get("filesystem_network_exec") == "denied" and payload.get("inherited_source_fds") == "closed"
            and payload.get("rows") == doc["rows"] and payload.get("seed") == doc["draw_seed"],
            "pilot_stage_output_invalid",
        )
    elif label == "learners":
        if doc["version"] == 2:
            _require(draw_manifest_sha256 is not None
                     and payload.get("materialized_draw_manifest_sha256") == draw_manifest_sha256,
                     "pilot_stage_output_invalid")
        _require(
            payload.get("comparator") == doc["comparator"] and payload.get("candidate") == doc["candidate"]
            and tuple(payload.get("training_scales", ())) == SCALES
            and tuple(payload.get("learners", ())) == LEARNERS and tuple(payload.get("contexts", ())) == CONTEXTS
            and tuple(payload.get("learner_paths", ())) == LEARNER_PATHS
            and payload.get("evaluation_count") == len(SCALES) * len(LEARNERS) * len(CONTEXTS)
            and payload.get("artifact_sha256") == artifact_sha
            and payload.get("selection_csv_sha256") == doc["selection_csv_sha256"]
            and payload.get("query_csv_sha256") == doc["query_csv_sha256"],
            "pilot_stage_output_invalid",
        )
        _hex(payload.get("paired_loss_digest"), 64, "pilot_stage_output_invalid")
    elif label == "drivers":
        _require(
            payload.get("path") == "drivers-v1" and payload.get("evidence_kind") == "permutation_importance"
            and payload.get("seed") == doc["driver_seed"] and tuple(payload.get("learners", ())) == LEARNERS
            and payload.get("query_csv_sha256") == doc["query_csv_sha256"],
            "pilot_stage_output_invalid",
        )
        _hex(payload.get("driver_digest"), 64, "pilot_stage_output_invalid")
    elif label == "receipt":
        _require(
            payload.get("artifact_sha256") == artifact_sha and payload.get("comparator") == doc["comparator"]
            and payload.get("byte_budget") == doc["byte_budget"],
            "pilot_stage_output_invalid",
        )
        _hex(payload.get("receipt_digest"), 64, "pilot_stage_output_invalid")
    elif label == "report":
        _require(
            payload.get("redacted") is True and payload.get("sealed_holdout_opened") is False
            and payload.get("scientific_qualification") == "unmeasured"
            and payload.get("release_authorization") == "not_authorized",
            "pilot_stage_output_invalid",
        )
        _hex(payload.get("report_digest"), 64, "pilot_stage_output_invalid")


def _record_pilot_stage(journal, cell_id, stage, wall_ms, ram_limit, fn):
    journal.begin(cell_id, stage, wall_ms)
    try:
        outcome, payload, artifact_sha = fn()
    except CampaignRunnerError as error:
        row = {"stage": stage, "status": "failed", "reason": str(error), "wall_ms": wall_ms}
        journal.finish(cell_id, stage, "failed", observed_wall_ms=wall_ms, peak_ram_bytes=0)
        return row, None, None
    status = "completed" if outcome["status"] == "completed" else "failed"
    if outcome["wall_ms"] > wall_ms or outcome["peak_ram_bytes"] > ram_limit:
        status = "over_budget"
    row = {
        "stage": stage, "status": status, "native_status": outcome["status"],
        "wall_ms": outcome["wall_ms"], "peak_ram_bytes": outcome["peak_ram_bytes"],
        "returncode": outcome["returncode"],
    }
    for key in ("stdout_sha256", "stderr_sha256"):
        if key in outcome:
            row[key] = outcome[key]
    if artifact_sha is not None:
        row["artifact_sha256"] = artifact_sha
    journal.finish(
        cell_id, stage, status, observed_wall_ms=outcome["wall_ms"],
        peak_ram_bytes=outcome["peak_ram_bytes"],
        output_sha256=digest(row) if status == "completed" else None,
    )
    return row, (payload if status == "completed" else None), artifact_sha


def _stage_command(paths, plan, stage):
    doc = plan.document()
    if doc["comparator"] == "dope":
        argv = native_arguments(paths["binary"], stage, plan, paths["artifact"].parent)
        if stage == "compile":
            # The shared engineering route defaults to chow_liu. A pilot must
            # retain the candidate bound into its signed cell plan instead.
            argv[argv.index("--candidate") + 1] = doc["candidate"]
        if doc["version"] == 2 and stage in ("compile", "sample"):
            key = "fit_seed" if stage == "compile" else "draw_seed"
            argv[argv.index("--seed") + 1] = str(_seed(doc, key))
            if stage == "sample":
                argv.append("--private-samples-f64le")
        return paths["binary"], argv[1:]
    _require(stage in ("compile", "reload", "sample"), "native_stage_unsupported")
    exe = paths["worker"]
    if stage == "compile":
        return exe, [
            "control-compile", "--comparator", doc["comparator"],
            "--csv", str(paths["csv"]), "--schema", str(paths["schema"]),
            "--out", str(paths["artifact"]), "--seed", str(doc["fit_seed"]),
            "--candidate", doc["candidate"], "--max-complete-bytes", str(doc["byte_budget"]),
            "--deadline-seconds", str(doc["deadline_seconds"]),
        ]
    if stage == "reload":
        return exe, ["control-reload", "--comparator", doc["comparator"], "--artifact", str(paths["artifact"])]
    return exe, ["control-sample", "--comparator", doc["comparator"],
                 "--artifact", str(paths["artifact"]), "--rows", str(doc["rows"]), "--seed", str(doc["draw_seed"])]


def execute_pilot_cell(
    spec: CampaignRunSpec, pilot_plan: PilotExecutionPlan, admission_contract: dict,
    cell_plan: PilotCellExecutionPlan, evidence_dir: Path, *,
    admission, public_key, trusted_key_sha256: str,
) -> dict:
    """Execute one admitted 8-stage pilot cell with durable StageJournal reservations."""
    root = private_root(evidence_dir)
    cell_plan.verify_binding(spec, pilot_plan, admission_contract)
    auth = _verify_pilot_authorization(
        root, spec, pilot_plan, admission_contract, cell_plan, admission, public_key, trusted_key_sha256,
    )
    _require(not (root / "pilot-execution.json").exists(), "pilot_replay_refused")
    doc = cell_plan.document()
    cell_id, ram_limit = doc["cell_id"], spec.document()["budget"]["peak_ram_bytes"]
    stage_rows, processes, paths, compiled_sha = [], 0, None, None
    draw_manifest_sha256 = None

    with StageJournal(root, spec, create=True) as journal:
        def run_step(binary_path, args, label, expected_artifact=None, private_sink=None):
            nonlocal processes
            if file_hash(paths["binary"], MAX_BINARY) != doc["binary_sha256"]:
                raise CampaignRunnerError("pilot_binary_modified")
            if file_hash(paths["worker"], MAX_BINARY) != doc["worker_binary_sha256"]:
                raise CampaignRunnerError("pilot_worker_modified")
            if expected_artifact is not None and file_hash(paths["artifact"], MAX_BINARY) != expected_artifact:
                raise CampaignRunnerError("pilot_artifact_modified")
            processes += 1
            wall = doc["stage_wall_ms"]["prepare" if label == "doctor" else label]
            outcome = invoke([str(binary_path), *args], root, label, wall, ram_limit)
            if outcome["status"] != "completed":
                return outcome, None, None
            try:
                payload = parse_json(read_regular(root / (label + ".stdout")))
            except EvidenceError:
                raise CampaignRunnerError("pilot_stage_output_invalid") from None
            artifact_sha, artifact_size = expected_artifact, 0
            if label == "compile":
                sync_artifact(paths["artifact"])
                artifact_sha = file_hash(paths["artifact"], MAX_BINARY)
            if paths["artifact"].is_file():
                artifact_size = paths["artifact"].stat().st_size
            _verify_stage_payload(label, payload, doc, artifact_sha, artifact_size, draw_manifest_sha256)
            if private_sink is not None:
                create_private(private_sink, canonical_json(payload), 0o400)
                artifact_sha = file_hash(private_sink, MAX_INPUT)
            return outcome, payload, artifact_sha

        def prepare():
            nonlocal paths
            paths = _snapshot_pilot_inputs(root, doc)
            return run_step(paths["binary"], ["doctor"], "doctor")

        def sample_pair():
            # Called only after _record_pilot_stage has durably reserved SAMPLE.
            # Both physical draws consume that ONE wall allowance. No retry/reset.
            nonlocal processes, draw_manifest_sha256
            from .pilot_materialized_draws import DrawSpec, decode_draw, store_pair
            start = time.monotonic()
            wall = doc["stage_wall_ms"]["sample"]
            schema_raw = read_regular(paths["schema"])
            schema = parse_json(schema_raw)
            _require(type(schema) is dict and type(schema.get("columns")) is list,
                     "pilot_schema_invalid")
            draws, outcomes = [], []
            for scale, factor, key in ((SCALES[0], 1, "draw_seed"), (SCALES[1], 4, "draw_4n_seed")):
                remaining = wall - int((time.monotonic() - start) * 1000 + 1)
                _require(remaining > 0, "pilot_sample_budget_exhausted")
                _require(file_hash(paths["binary"], MAX_BINARY) == doc["binary_sha256"]
                         and file_hash(paths["artifact"], MAX_BINARY) == compiled_sha,
                         "pilot_sample_input_modified")
                exe, args = _stage_command(paths, cell_plan, "sample")
                args[args.index("--rows") + 1] = str(doc["rows"] * factor)
                args[args.index("--seed") + 1] = str(_seed(doc, key))
                label = "sample_" + scale
                processes += 1
                outcome = invoke([str(exe), *args], root, label, remaining, ram_limit)
                outcomes.append(outcome)
                if outcome["status"] != "completed":
                    return {**outcome, "wall_ms": int((time.monotonic() - start) * 1000 + 1),
                            "peak_ram_bytes": max(o["peak_ram_bytes"] for o in outcomes)}, None, compiled_sha
                spec_draw = DrawSpec(scale, "dope", doc["rows"] * factor, len(schema["columns"]),
                                     doc[key], compiled_sha, doc["schema_sha256"],
                                     cell_plan.input_sha256, doc["byte_budget"])
                draws.append(decode_draw(read_regular(root / (label + ".stdout")),
                                         read_regular(root / (label + ".stderr")), spec_draw,
                                         schema_document=schema_raw))
            _require(file_hash(paths["binary"], MAX_BINARY) == doc["binary_sha256"],
                     "pilot_sample_input_modified")
            manifest = store_pair(root, *draws, artifact=paths["artifact"])
            draw_manifest_sha256 = file_hash(root / "materialized-draws/manifest.json")
            return {"status": "completed", "returncode": 0,
                    "wall_ms": int((time.monotonic() - start) * 1000 + 1),
                    "peak_ram_bytes": max(o["peak_ram_bytes"] for o in outcomes)}, manifest, compiled_sha

        def learners():
            def verify_inputs():
                _require(all(file_hash(paths[key]) == doc[key + "_sha256"]
                             for key in ("csv", "schema", "selection_csv", "query_csv")),
                         "pilot_learner_input_modified")
            verify_inputs()
            sample_args = (["--sample-stdout", str(root / "sample.stdout")] if doc["version"] == 1 else
                           ["--materialized-draw-manifest", str(root / "materialized-draws/manifest.json"),
                            "--materialized-draw-manifest-sha256", draw_manifest_sha256])
            result = run_step(paths["worker"], [
                "learners", "--artifact", str(paths["artifact"]), *sample_args,
                "--fit-csv", str(paths["csv"]), "--fit-csv-sha256", doc["csv_sha256"],
                "--schema", str(paths["schema"]), "--schema-sha256", doc["schema_sha256"],
                "--selection-csv", str(paths["selection_csv"]), "--query-csv", str(paths["query_csv"]),
                "--comparator", doc["comparator"], "--candidate", doc["candidate"],
                "--rows-primary", str(doc["rows"]), "--rows-4n", str(doc["rows"] * 4),
                "--control-seed", str(doc["control_seed"]),
            ], "learners", compiled_sha, paths["learners"])

            verify_inputs()
            return result

        stage_fns = (
            ("prepare", prepare),
            ("compile", lambda: run_step(*_stage_command(paths, cell_plan, "compile"), "compile")),
            ("reload", lambda: run_step(*_stage_command(paths, cell_plan, "reload"), "reload", compiled_sha)),
            ("sample", sample_pair if doc["version"] == 2 else
             lambda: run_step(*_stage_command(paths, cell_plan, "sample"), "sample", compiled_sha)),
            ("learners", learners),
            ("drivers", lambda: run_step(paths["worker"], [
                "drivers", "--learners-evidence", str(paths["learners"]),
                "--query-csv", str(paths["query_csv"]), "--driver-seed", str(doc["driver_seed"]),
            ], "drivers", compiled_sha, paths["drivers"])),
            ("receipt", lambda: run_step(paths["worker"], [
                "receipt", "--artifact", str(paths["artifact"]),
                "--learners-evidence", str(paths["learners"]), "--drivers-evidence", str(paths["drivers"]),
            ], "receipt", compiled_sha, paths["receipt"])),
            ("report", lambda: run_step(paths["worker"], [
                "report", "--stage-receipt", str(paths["receipt"]),
            ], "report", compiled_sha, paths["report"])),
        )
        for stage, fn in stage_fns:
            row, payload, artifact_sha = _record_pilot_stage(
                journal, cell_id, stage, doc["stage_wall_ms"][stage], ram_limit, fn,
            )
            stage_rows.append(row)
            if payload is None:
                break
            if stage == "compile":
                compiled_sha = artifact_sha
        summary = journal.public_summary()

    ok = (
        len(stage_rows) == len(STAGES)
        and all(r["status"] == "completed" for r in stage_rows)
        and summary["pending_stages"] == 0
    )
    receipt = {
        "format": RECEIPT_FORMAT, "version": 1, "status": "ok" if ok else "failed",
        "scope": SCOPE, "run_spec_sha256": spec.sha256,
        "pilot_execution_plan_sha256": pilot_plan.sha256,
        "cell_execution_plan_sha256": cell_plan.sha256, **auth,
        "cell_id": cell_id, "task_id": doc["task_id"], "comparator": doc["comparator"],
        "candidate": doc["candidate"], "byte_budget": doc["byte_budget"],
        "processes_executed": processes, "stages": stage_rows, "journal": summary,
        "bridge_stages_complete": ok, "pipeline_complete": False,
        "pipeline_prerequisites": [p for p in PIPELINE_PREREQUISITES if not
                                   (p == "materialized_primary_and_4n_draw_handoff" and draw_manifest_sha256
                                    and any(r["stage"] == "sample" and r["status"] == "completed" for r in stage_rows))],
        "scientific_admission": False, "scientific_qualification": "unmeasured",
        "release_authorization": "not_authorized",
    }
    if doc["version"] == 2:
        receipt["materialized_draw_manifest_sha256"] = draw_manifest_sha256
    create_private(root / "pilot-execution.json", canonical_json(receipt), 0o400)
    return receipt
