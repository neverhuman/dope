"""Fail-closed evidence validation for the W01 native qualification runner.

This module validates the signed admission envelope and immutable local inputs.
It does not turn installation, fixture tests, or a partial run into a model
qualification. Q0--Q9 remain individually visible in the emitted receipt.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any


SCHEMA = "dope.w01-native-qualification-manifest.v1"
GATES = tuple(f"Q{i}" for i in range(10))
TASKS = ("regression", "classification")
MODEL_FILES = {
    "tabicl_v2": {
        "revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
        "files": {
            "tabicl-classifier-v2-20260212.ckpt": (110368038, "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"),
            "tabicl-regressor-v2-20260212.ckpt": (114324594, "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"),
        },
    },
    "kumo_tabular": {
        "revision": "3520742f7f3b937a2751266cea22d7fb6686882c",
        "files": {
            "large/classifier.pt": (854850034, "cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613"),
            "large/regressor.pt": (862909259, "745e0e5d97e4256d8a24cb61c2eecbd1128da6c83938c196b7589ea6f62e7421"),
        },
    },
}
MODEL_SOURCES = {
    "tabicl_v2": ("soda-inria/tabicl", "10719e21fd72b8945d4a4f9ae077e350ba8650d9",
                  "jingang/TabICL", "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"),
    "kumo_tabular": ("NVIDIA/structured-data-models", "d50589692dbbbad4852eb1d9627f3b40c2673b63",
                     "nvidia/Kumo-Tabular", "3520742f7f3b937a2751266cea22d7fb6686882c"),
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


class QualificationError(ValueError):
    """A bounded, non-sensitive qualification input error."""


def sha256_file(path: str | Path) -> str:
    """Hash a regular, non-symlink file without following its final component."""
    path = Path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise QualificationError("not_regular_file")
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
            return h.hexdigest()
    except QualificationError:
        raise
    except OSError:
        raise QualificationError("file_unavailable") from None


def _read_json(path: str | Path, limit: int = 16 * 1024 * 1024) -> dict[str, Any]:
    try:
        p = Path(path)
        if p.is_symlink() or not p.is_file() or p.stat().st_size > limit:
            raise QualificationError("manifest_unavailable")
        value = json.loads(p.read_text(encoding="utf-8"))
    except QualificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise QualificationError("manifest_unavailable") from None
    if type(value) is not dict:
        raise QualificationError("manifest_invalid")
    return value


def canonical_payload(envelope: dict[str, Any]) -> bytes:
    if set(envelope) != {"payload", "signature_ed25519_base64"}:
        raise QualificationError("signature_envelope_invalid")
    payload = envelope["payload"]
    if type(payload) is not dict:
        raise QualificationError("manifest_invalid")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def verify_signed_manifest(manifest_path: str | Path, public_key_path: str | Path,
                           trusted_key_sha256: str) -> dict[str, Any]:
    """Verify an Ed25519 envelope against an out-of-band pinned public key.

    The trusted-key digest is an explicit CLI input. A key embedded only in the
    manifest is never trusted. OpenSSL is invoked without a shell.
    """
    if not _SHA.fullmatch(trusted_key_sha256):
        raise QualificationError("trusted_key_pin_invalid")
    if sha256_file(public_key_path) != trusted_key_sha256:
        raise QualificationError("trusted_key_pin_mismatch")
    envelope = _read_json(manifest_path)
    payload_bytes = canonical_payload(envelope)
    try:
        signature = base64.b64decode(envelope["signature_ed25519_base64"], validate=True)
    except (ValueError, TypeError):
        raise QualificationError("signature_invalid") from None
    try:
        with tempfile.TemporaryDirectory(prefix="dope-w01-verify-", dir=Path(manifest_path).parent) as td:
            message = Path(td) / "manifest.json"
            sig = Path(td) / "manifest.sig"
            message.write_bytes(payload_bytes)
            sig.write_bytes(signature)
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(public_key_path),
                 "-rawin", "-in", str(message), "-sigfile", str(sig)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False, timeout=15,
            )
    except (OSError, subprocess.SubprocessError):
        raise QualificationError("signature_verifier_unavailable") from None
    if result.returncode:
        raise QualificationError("signature_invalid")
    validate_admission_payload(envelope["payload"])
    return envelope["payload"]


def validate_admission_payload(payload: dict[str, Any]) -> None:
    """Validate the exact W01 signed admission scope before any model loading."""
    required = {"schema", "status", "lease_id", "source_commit", "admission", "cases"}
    if set(payload) != required or payload.get("schema") != SCHEMA:
        raise QualificationError("manifest_schema_invalid")
    if payload.get("status") != "admitted" or not isinstance(payload.get("lease_id"), str) or not payload["lease_id"]:
        raise QualificationError("manifest_not_admitted")
    if not re.fullmatch(r"[0-9a-f]{40}", str(payload.get("source_commit", ""))):
        raise QualificationError("source_commit_invalid")
    admission = payload["admission"]
    if type(admission) is not dict or set(admission) != {"rights", "resource", "roles"}:
        raise QualificationError("admission_incomplete")
    if any(admission[k] != "approved" for k in ("rights", "resource", "roles")):
        raise QualificationError("admission_incomplete")
    cases = payload["cases"]
    expected = {(model, task) for model in MODEL_FILES for task in TASKS}
    try:
        actual = {(case["model"], case["task"]) for case in cases}
    except (TypeError, KeyError):
        raise QualificationError("case_matrix_invalid") from None
    if type(cases) is not list or len(cases) != 4 or actual != expected:
        raise QualificationError("case_matrix_incomplete")
    for case in cases:
        if type(case) is not dict or set(case) != {
                "model", "task", "code_repo", "code_revision", "checkpoint_repo",
                "checkpoint_revision", "checkpoint", "dependency_closure_sha256",
                "code_license_sha256", "weights_license_sha256", "role",
                "class_order", "recipe_sha256", "context_sha256", "query_sha256", "resource_caps"}:
            raise QualificationError("case_invalid")
        model = case["model"]
        if model not in MODEL_FILES or case["task"] not in TASKS:
            raise QualificationError("case_invalid")
        code_repo, code_revision, checkpoint_repo, checkpoint_revision = MODEL_SOURCES[model]
        if (case["code_repo"], case["code_revision"], case["checkpoint_repo"], case["checkpoint_revision"]) != (code_repo, code_revision, checkpoint_repo, checkpoint_revision):
            raise QualificationError("model_identity_mismatch")
        if not isinstance(case["role"], str) or not case["role"]:
            raise QualificationError("model_role_unresolved")
        classes = case["class_order"]
        if (case["task"] == "classification" and
                (type(classes) is not list or len(classes) != 2 or
                 not all(type(label) is str and label for label in classes) or len(set(classes)) != 2)):
            raise QualificationError("binary_class_order_invalid")
        if case["task"] == "regression" and classes != []:
            raise QualificationError("regression_class_order_invalid")
        expected_name = ("large/classifier.pt" if case["task"] == "classification" else "large/regressor.pt") if model == "kumo_tabular" else ("tabicl-classifier-v2-20260212.ckpt" if case["task"] == "classification" else "tabicl-regressor-v2-20260212.ckpt")
        if case["checkpoint"] != expected_name:
            raise QualificationError("checkpoint_task_mismatch")
        for key in ("dependency_closure_sha256", "code_license_sha256", "weights_license_sha256",
                    "recipe_sha256", "context_sha256", "query_sha256"):
            if not isinstance(case[key], str) or not _SHA.fullmatch(case[key]):
                raise QualificationError("case_pin_invalid")
        caps = case["resource_caps"]
        if type(caps) is not dict or set(caps) != {"wall_seconds", "memory_bytes", "gpu_bytes", "threads"}:
            raise QualificationError("resource_caps_invalid")
        if (any(type(v) is not int or v < 0 for v in caps.values()) or
                any(caps[k] == 0 for k in ("wall_seconds", "memory_bytes", "threads"))):
            raise QualificationError("resource_caps_invalid")


def verify_checkpoints(payload: dict[str, Any], checkpoint_root: str | Path) -> dict[str, dict[str, Any]]:
    """Verify only the four admitted large/v2 task checkpoints, by byte count and hash."""
    root = Path(checkpoint_root).resolve(strict=True)
    results: dict[str, dict[str, Any]] = {}
    cases = {(case["model"], case["task"]): case for case in payload["cases"]}
    for model, spec in MODEL_FILES.items():
        for task in TASKS:
            case = cases[(model, task)]
            rel = Path(case["checkpoint"])
            if rel.is_absolute() or ".." in rel.parts:
                raise QualificationError("checkpoint_path_invalid")
            candidate = root / model / rel
            cursor = root
            for part in Path(model, rel).parts:
                cursor = cursor / part
                try:
                    mode = cursor.lstat().st_mode
                except OSError:
                    raise QualificationError("checkpoint_unavailable") from None
                if stat.S_ISLNK(mode):
                    raise QualificationError("checkpoint_path_invalid")
            path = candidate.resolve(strict=True)
            if root not in path.parents:
                raise QualificationError("checkpoint_path_invalid")
            expected_size, expected_sha = spec["files"][str(rel)]
            try:
                size = path.stat().st_size
            except OSError:
                raise QualificationError("checkpoint_unavailable") from None
            if size != expected_size or sha256_file(path) != expected_sha:
                raise QualificationError("checkpoint_pin_mismatch")
            results[f"{model}:{task}"] = {"bytes": size, "sha256": expected_sha, "verified": True}
    return results


def new_result(payload: dict[str, Any], checkpoint_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create an honestly incomplete receipt; gates pass only with bound evidence."""
    cases = []
    for case in payload["cases"]:
        cases.append({
            "model": case["model"], "task": case["task"],
            "class_order": case["class_order"],
            "recipe_sha256": case["recipe_sha256"],
            "gates": {gate: {"status": "pending", "evidence_sha256": None} for gate in GATES},
        })
    return {
        "schema": "dope.w01-native-qualification-result.v1",
        "status": "incomplete",
        "lease_id": payload["lease_id"],
        "source_commit": payload["source_commit"],
        "admission": payload["admission"],
        "checkpoint_verification": checkpoint_receipt,
        "cases": cases,
        "blockers": ["native_execution_and_Q0_Q9_evidence_not_recorded"],
        "release_authorization": "not_authorized",
    }


def assess_result(result: dict[str, Any]) -> str:
    """Require output parity as well as complete Q0--Q9 evidence for qualification."""
    cases = result.get("cases")
    if type(cases) is not list or len(cases) != 4:
        return "incomplete"
    expected = {(model, task) for model in MODEL_FILES for task in TASKS}
    found = set()
    parity_all = True
    for case in cases:
        if type(case) is not dict or (case.get("model"), case.get("task")) not in expected:
            return "incomplete"
        found.add((case["model"], case["task"]))
        gates = case.get("gates")
        if type(gates) is not dict or set(gates) != set(GATES):
            return "incomplete"
        for gate in GATES:
            evidence = gates[gate]
            if (type(evidence) is not dict or evidence.get("status") != "passed" or
                    not _SHA.fullmatch(str(evidence.get("evidence_sha256", "")))):
                return "incomplete"
            if gate == "Q9":
                review = evidence.get("independent_review")
                if (type(review) is not dict or set(review) != {"reviewer_id", "disposition", "receipt_sha256"} or
                        not review["reviewer_id"] or review["disposition"] != "accepted" or
                        not _SHA.fullmatch(str(review["receipt_sha256"]))):
                    return "incomplete"
            if gate == "Q2":
                observation = evidence.get("observation")
                if type(observation) is not dict or observation.get("native_checkpoint_loaded") is not True:
                    return "incomplete"
                parity_state = observation.get("adapter_output_parity")
                if parity_state in (None, "not_established"):
                    parity_all = False
                    continue
                if parity_state != "established":
                    return "incomplete"
                tolerance = observation.get("tolerance")
                delta = observation.get("adapter_vs_sdk_max_abs_delta")
                if (type(tolerance) not in (int, float) or tolerance < 0 or
                        type(delta) not in (int, float) or delta < 0 or delta > tolerance or
                        observation.get("adapter_output_sha256") != observation.get("sdk_output_sha256") or
                        observation.get("adapter_output_shape") != observation.get("sdk_output_shape") or
                        observation.get("adapter_class_order") != observation.get("sdk_class_order") or
                        observation.get("class_order") != case.get("class_order") or
                        observation.get("adapter_class_order") != case.get("class_order") or
                        observation.get("recipe_sha256") != case.get("recipe_sha256") or
                        not _SHA.fullmatch(str(observation.get("recipe_sha256", ""))) or
                        not _SHA.fullmatch(str(observation.get("preprocessing_sha256", ""))) or
                        not _SHA.fullmatch(str(observation.get("output_aggregation_sha256", ""))) or
                        type(observation.get("seed")) is not int or
                        observation.get("precision") != "float32"):
                    return "incomplete"
    if found != expected:
        return "incomplete"
    # A complete checkpoint-load review remains useful technical smoke evidence,
    # but is deliberately distinct from official output parity qualification.
    if not parity_all:
        return "technical_smoke_reviewed"
    return "qualified"


def bind_independent_review(result_path: str | Path, review_path: str | Path,
                            output_path: str | Path, *,
                            expected_result_sha256: str,
                            expected_review_sha256: str,
                            expected_reviewer_id: str) -> dict[str, Any]:
    """Bind a pinned independent Q9 receipt without changing its execution input.

    Both expected hashes are out-of-band pins supplied by the dispatch. The
    review must name the exact capped result, lease, reviewer identity and
    accepted disposition. Only Q9 is populated; Q0--Q8 must already pass.
    """
    result_path, review_path, output_path = map(Path, (result_path, review_path, output_path))
    if output_path.resolve() == result_path.resolve():
        raise QualificationError("qualified_result_must_be_distinct")
    if not _SHA.fullmatch(expected_result_sha256) or not _SHA.fullmatch(expected_review_sha256):
        raise QualificationError("review_pin_invalid")
    actual_result_sha256 = sha256_file(result_path)
    if actual_result_sha256 != expected_result_sha256:
        raise QualificationError("capped_result_pin_mismatch")
    actual_review_sha256 = sha256_file(review_path)
    if actual_review_sha256 != expected_review_sha256:
        raise QualificationError("review_receipt_pin_mismatch")
    result = _read_json(result_path)
    review = _read_json(review_path)
    if result.get("schema") != "dope.w01-native-qualification-result.v1":
        raise QualificationError("result_schema_invalid")
    if review.get("schema") != "dope.independent_cross_family_review.f65_01_w01_q9_signoff.v1":
        raise QualificationError("review_schema_invalid")
    bound = review.get("bound_execution_artifacts")
    reviewer = review.get("reviewer")
    if (type(bound) is not dict or bound.get("capped_result_sha256") != actual_result_sha256 or
            bound.get("lease_id") != result.get("lease_id") or
            bound.get("source_commit") != result.get("source_commit") or
            bound.get("dependency_closure_sha256") != result.get("environment", {}).get("dependency_closure_sha256")):
        raise QualificationError("review_execution_binding_mismatch")
    if (type(reviewer) is not dict or reviewer.get("reviewer_id") != expected_reviewer_id or
            reviewer.get("author_separation_verified") is not True or
            reviewer.get("authored_by_reviewer") is not False or
            reviewer.get("role") != "independent_cross_family_reviewer" or
            review.get("disposition") != "accepted"):
        raise QualificationError("independent_review_not_accepted")
    cases = result.get("cases")
    expected_cases = {(model, task) for model in MODEL_FILES for task in TASKS}
    if type(cases) is not list or len(cases) != len(expected_cases):
        raise QualificationError("result_case_matrix_invalid")
    seen = set()
    for case in cases:
        key = (case.get("model"), case.get("task"))
        if key not in expected_cases or key in seen:
            raise QualificationError("result_case_matrix_invalid")
        seen.add(key)
        gates = case.get("gates")
        if type(gates) is not dict or set(gates) != set(GATES):
            raise QualificationError("result_gate_matrix_invalid")
        for gate in GATES[:-1]:
            evidence = gates[gate]
            if (type(evidence) is not dict or evidence.get("status") != "passed" or
                    not _SHA.fullmatch(str(evidence.get("evidence_sha256", "")))):
                raise QualificationError("native_gate_evidence_incomplete")
        if gates["Q9"].get("status") != "pending_independent_review":
            raise QualificationError("q9_state_invalid")
    if seen != expected_cases:
        raise QualificationError("result_case_matrix_invalid")
    for case in cases:
        case["gates"]["Q9"] = {
            "status": "passed",
            "evidence_sha256": actual_review_sha256,
            "independent_review": {
                "reviewer_id": expected_reviewer_id,
                "disposition": "accepted",
                "receipt_sha256": actual_review_sha256,
            },
        }
    result["independent_review"] = {
        "reviewer_id": expected_reviewer_id,
        "disposition": "accepted",
        "receipt_sha256": actual_review_sha256,
    }
    result["status"] = assess_result(result)
    if result["status"] not in {"qualified", "technical_smoke_reviewed"}:
        raise QualificationError("q9_binding_incomplete")
    result["blockers"] = ([] if result["status"] == "qualified" else
                           ["official_adapter_sdk_output_parity_not_established"])
    # Scientific qualification and release authorization are separate gates.
    result["release_authorization"] = "not_authorized"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise QualificationError("qualified_result_already_exists")
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return result
