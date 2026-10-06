"""Metadata contract for the bounded GPU research grid; no artifact decoding.

The caller supplies inspector and TRAIN-derived validation-log bytes plus their
previously frozen pins. Source/binary/runtime/artifact pins are custody inputs,
not claims that this helper read those bodies or proved runtime qualification.
Width, fit seed and readout settings remain requested configuration; inspection
does not independently attest them. Actual training needs execution receipts.
"""

import hashlib
import json
import math
import re
from enum import Enum


class ActualTargetOpcode(str, Enum):
    SPARSE_LINEAR = "sparse_linear"
    SPARSE_LOGISTIC = "sparse_logistic"
    COMPACT_NEURAL_RESIDUAL = "compact_neural_residual"


class BindingGap(ValueError):
    pass


def pin(ref):
    if (type(ref) is not dict or set(ref) != {"bytes", "sha256"}
            or type(ref["bytes"]) is not int or ref["bytes"] <= 0
            or type(ref["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None):
        raise BindingGap("invalid metadata pin")
    return dict(ref)


def pinned_bytes(data, ref):
    ref = pin(ref)
    if (type(data) is not bytes or len(data) != ref["bytes"]
            or hashlib.sha256(data).hexdigest() != ref["sha256"]):
        raise BindingGap("metadata byte pin mismatch")
    return data


def profile_settings(profile):
    if type(profile) is not str:
        raise BindingGap("invalid bounded research profile")
    match = re.fullmatch(
        r"grid_features(6|12|24)_width(linear|8|16)_steps(512|2048|8192)_readout(on|off)",
        profile,
    )
    if match is None:
        raise BindingGap("invalid bounded research profile")
    features, width, steps, readout = match.groups()
    return {
        "requested_architecture": "linear" if width == "linear" else "neural_residual",
        "requested_width": width,
        "feature_limit": int(features),
        "optimizer_steps": int(steps),
        "readout_refit": readout == "on",
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BindingGap("duplicate inspection field")
        result[key] = value
    return result


def bind_ablation_metadata(request, inspection_bytes, loss_log_bytes):
    required = {"requested_profile", "fit_seed", "candidate_slot", "source_bundle_ref",
                "binary_ref", "runtime_ref", "artifact_ref", "inspection_ref", "loss_log_ref"}
    if type(request) is not dict or set(request) != required:
        raise BindingGap("invalid ablation metadata request")
    if (type(request["fit_seed"]) is not int or not 0 <= request["fit_seed"] < 2**64
            or type(request["candidate_slot"]) is not str
            or request["candidate_slot"] != "compact_neural_residual"):
        raise BindingGap("invalid requested identity")
    settings = profile_settings(request["requested_profile"])
    refs = {key: pin(request[key]) for key in required if key.endswith("_ref")}
    # Verify both emitted metadata byte pins BEFORE either payload is decoded.
    inspection_bytes = pinned_bytes(inspection_bytes, refs["inspection_ref"])
    loss_log_bytes = pinned_bytes(loss_log_bytes, refs["loss_log_ref"])
    try:
        def reject_constant(_):
            raise BindingGap("nonfinite inspection metadata")
        inspection = json.loads(inspection_bytes, object_pairs_hook=unique_object,
                                parse_constant=reject_constant)
        if type(inspection) is not dict:
            raise BindingGap("invalid native inspection binding")
        opcode = ActualTargetOpcode(inspection["target"])
        version = inspection["version"]
        if (type(version) is not int or version not in (2, 3)
                or inspection["format"] != f"dope-kernel-v{version}"
                or type(inspection["decoder_id"]) is not int or inspection["decoder_id"] != 1
                or type(inspection["rows_fitted"]) is not int or inspection["rows_fitted"] <= 0
                or type(inspection["artifact_bytes"]) is not int
                or inspection["artifact_bytes"] != refs["artifact_ref"]["bytes"]):
            raise BindingGap("invalid native inspection binding")
        if settings["requested_architecture"] == "linear":
            if opcode not in (ActualTargetOpcode.SPARSE_LINEAR, ActualTargetOpcode.SPARSE_LOGISTIC):
                raise BindingGap("requested architecture disagrees with actual opcode")
        elif opcode is not ActualTargetOpcode.COMPACT_NEURAL_RESIDUAL:
            raise BindingGap("requested architecture disagrees with actual opcode")
        records = loss_log_bytes.decode("ascii").splitlines()
        if len(records) != settings["optimizer_steps"] or not loss_log_bytes.endswith(b"\n"):
            raise BindingGap("incomplete validation log")
        for step, record in enumerate(records):
            fields = record.split("\t")
            if len(fields) != 3 or fields[0] != str(step):
                raise BindingGap("invalid validation step sequence")
            if any(not math.isfinite(float(value)) for value in fields[1:]):
                raise BindingGap("nonfinite validation loss")
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        if isinstance(error, BindingGap):
            raise
        raise BindingGap("invalid emitted ablation metadata") from None
    # Never forward schema names, symbolic expressions, learned coefficients or log values.
    return {
        "version": "research-ablation-binding.v1",
        "requested_profile": request["requested_profile"], **settings,
        "fit_seed": request["fit_seed"], "candidate_slot": request["candidate_slot"],
        "actual_target_opcode": opcode.value, **refs,
        "validation_log_records": len(records),
        "validation_log_phase": "train_pre_optimizer_validation_post_optimizer_pre_readout_refit",
        "identity_scope": "requested_configuration_and_emitted_opcode_only",
        "production_eligibility": None,
    }
