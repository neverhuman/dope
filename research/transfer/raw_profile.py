"""Charge the native raw-complete profile after the inspect-raw digest check.

DOPERAW1 and DOPERAW2 share 8 magic bytes, a little-endian state length, a
little-endian DPK length, and 32 bytes of unkeyed BLAKE3 over the following
state||DPK. The carrier charge is that whole artifact. Generic runtime bytes
and the mapped cold closure stay None: sample-raw-isolated measures those
separately, and a missing number is not a zero. DOPERAW2 requires DPK 3.3
regression and an uncompressed noise section of opcode 4, version 1, kind 0-3.
The kind fields are counted and discarded. This check does not decompress,
decode the kernel, reload affine columns, or publish offsets, scales, or knots.
DOPMIX01 is a separate carrier and is charged only by account_mixed_artifact.
A passed charge is not confirmation, a right, an official asset, a release,
formal DP, or HIPAA de-identification. Historical DPK tier totals are a
different account, and inventory-only does not replay this check.
"""

from types import MappingProxyType

from research.transfer.blake3_digest import blake3_hash

RAW_MAGIC = b"DOPERAW1"
RESIDUAL_MAGIC = b"DOPERAW2"
RAW_HEADER = 56
RAW_MAX_BYTES = 32 * 1024 * 1024
RAW_HANDLER = "native-raw-complete-v1"
RESIDUAL_BYTE_PROFILE = "native-raw-complete-residual-v1"
RAW_KIND = "doperaw1"
RAW_CAPABILITIES = ["byte-inventory", "native-raw-sample"]
MIXED_MAGIC = b"DOPMIX01"
MIXED_HEADER = 56
MIXED_MAX_BYTES = 32 * 1024 * 1024
MIXED_MAX_STATE = 2 * 1024 * 1024
MIXED_BYTE_PROFILE = "native-mixed-complete-v1"
_MIXED_PROFILES = frozenset({"independent_v1", "numeric_split_v1"})
_MIXED_STATE_KEYS = frozenset({
    "format", "version", "profile", "constant_target", "affine", "features",
})
_MIXED_FEATURE_KEYS = frozenset({
    "id", "start", "categories", "completion", "probabilities", "split",
})
_MIXED_SPLIT_KEYS = frozenset({"anchor", "threshold", "left", "right"})
_NOISE_TAG = 5
_RESIDUAL_FIELDS = {0: 0, 1: 2, 2: 4, 3: 17}
_FEATURE_ROLES = frozenset({"feature", "target"})
_KNOWN_ROLES = _FEATURE_ROLES | {
    "entity_key",
    "timestamp",
    "group_split_key",
    "excluded_sensitive_identifier",
}
_PUBLIC_KEYS = frozenset({"id", "role", "dtype", "mask_states", "label", "units"})


def _fail(code):
    from research.transfer.bundle import _fail as fail

    fail(code)


def profile_limits():
    """Fixed claim boundary for raw-continuous-symbolic-v1. Not a measurement."""
    return {
        "format": "dope-raw-sampling-artifact",
        "version": 1,
        "profile": "raw-continuous-symbolic-v1",
        "capability": "fully_observed_continuous_regression",
        "feature_support": "fit_range_bounded_historical_native_marginals",
        "target_support": "unclipped_affine_regression_not_tail_qualified",
        "constant_target_rule": "unit_scale_with_fitted_offset",
        "privacy": "unmeasured_private_transforms_retained",
        "certification": "unqualified_development_artifact",
        "content_hash": "unverified_native_inspect",
        "symbolic_backend": "unverified_native_inspect",
        "mixed_columns": "unsupported",
        "missing_mask": "unsupported",
        "confirmation": "unqualified",
        "rights": "unqualified",
        "official_assets": "unqualified",
        "formal_dp": False,
        "hipaa_deidentification": False,
        "release_authorized": False,
        "tail_fidelity": "unqualified",
        "learner_transfer": "unqualified",
    }


def _uncompressed_noise(payload):
    """Return the tag-5 body. A compressed tag 5 is closed, not decoded."""
    from research.transfer.bundle import BundleError
    from research.transfer.bundle_io import _varint

    minor = payload[5]
    maximum = 6 if payload[:4] == b"DPK3" and minor >= 2 else 5
    data = payload[RAW_HEADER:]
    offset = 0
    prior = 0
    noise = None
    compressed = False
    while offset < len(data):
        tag_byte = data[offset]
        offset += 1
        tag = tag_byte & 0x7F
        if tag == 0 or tag > maximum or tag <= prior:
            _fail("raw_payload_limits_invalid")
        prior = tag
        try:
            encoded_length, offset = _varint(data, offset)
            if tag_byte & 0x80:
                _, offset = _varint(data, offset)
        except BundleError:
            _fail("raw_payload_limits_invalid")
        if encoded_length > len(data) - offset:
            _fail("raw_payload_limits_invalid")
        if tag == _NOISE_TAG:
            if tag_byte & 0x80:
                compressed = True
            else:
                noise = data[offset:offset + encoded_length]
        offset += encoded_length
    return noise, compressed


def _residual_fields_fit(body):
    """Consume opcode 4, version 1, and kind 0-3. Discard every integer."""
    from research.transfer.bundle import BundleError
    from research.transfer.bundle_io import _varint

    if len(body) < 3 or body[0] != 4 or body[1] != 1 or body[2] not in _RESIDUAL_FIELDS:
        return False
    offset = 5 if body[2] == 2 else 3
    if offset > len(body):
        return False
    try:
        for _ in range(_RESIDUAL_FIELDS[body[2]]):
            _, offset = _varint(body, offset)
    except BundleError:
        return False
    return offset == len(body)


def _require_residual_agreement(magic, payload):
    noise, compressed = _uncompressed_noise(payload)
    residual_payload = (
        payload[:4] == b"DPK3"
        and payload[4] == 3
        and payload[5] == 3
        and payload[6] == 1
        and payload[7] == 0
        and not compressed
        and noise is not None
        and _residual_fields_fit(noise)
    )
    if magic == RESIDUAL_MAGIC:
        if not residual_payload:
            _fail("raw_residual_profile_mismatch")
    elif noise is not None and noise[:1] == b"\x04":
        # DOPERAW1 still carries uncompressed noise opcodes 0-3. Opcode 4 is residual.
        _fail("raw_residual_profile_mismatch")


def account_raw_artifact(artifact, limits=None):
    """Split one self-contained artifact after BLAKE3(state||DPK) matches bytes 24..56.

    Does not decode the fitted state or the native program. Column declaration
    is a separate call and does not inherit this digest result.
    """
    from research.transfer.bundle import BundleError, BundleLimits
    from research.transfer.bundle_io import preflight_dpk

    limits = limits or BundleLimits()
    if type(artifact) is not bytes or len(artifact) < RAW_HEADER or len(artifact) > RAW_MAX_BYTES:
        _fail("raw_artifact_invalid")
    magic = artifact[:8]
    if magic != RAW_MAGIC and magic != RESIDUAL_MAGIC:
        _fail("raw_artifact_invalid")
    state_len = int.from_bytes(artifact[8:16], "little")
    payload_len = int.from_bytes(artifact[16:24], "little")
    if state_len == 0 or state_len > len(artifact) or payload_len > len(artifact):
        _fail("raw_artifact_invalid" if state_len == 0 else "raw_artifact_integrity_failed")
    if RAW_HEADER + state_len + payload_len != len(artifact):
        _fail("raw_artifact_integrity_failed")
    if blake3_hash(artifact[RAW_HEADER:]) != artifact[24:56]:
        _fail("raw_artifact_integrity_failed")
    payload = artifact[RAW_HEADER + state_len:]
    try:
        preflight_dpk(payload, limits)
    except BundleError:
        _fail("raw_payload_limits_invalid")
    _require_residual_agreement(magic, payload)
    charge = {
        "byte_profile_id": RESIDUAL_BYTE_PROFILE if magic == RESIDUAL_MAGIC else RAW_HANDLER,
        "framing_bytes": RAW_HEADER,
        "transform_schema_bytes": state_len,
        "native_payload_bytes": payload_len,
        "complete_sampling_bytes": len(artifact),
        "transport_archive_bytes": len(artifact),
        "generic_runtime_bytes": None,
        "cold_sampling_closure_bytes": None,
        "mapped_runtime_cold_closure": "separate_unmeasured",
    }
    charge.update(profile_limits())
    if magic == RESIDUAL_MAGIC:
        charge["version"] = 2
        charge["profile"] = "raw-numeric-residual-v1"
        charge["capability"] = "fully_observed_numeric_features_continuous_regression"
        charge["mathematical_parameter_count"] = None
    # The label is the algorithm that matched. It is not a kernel decode.
    charge["content_hash"] = "blake3_state_plus_dpk"
    charge["declared_columns_checked"] = False
    return MappingProxyType(charge)


def _number(value):
    return type(value) in (int, float)


def _mixed_profile(state_bytes):
    """Return the profile name. Fitted values stay inside the byte charge."""
    from research.transfer.evidence import EvidenceError, parse_json

    try:
        state = parse_json(state_bytes)
    except (EvidenceError, ValueError, UnicodeError, RecursionError):
        _fail("mixed_state_invalid")
    if (
        type(state) is not dict
        or set(state) != _MIXED_STATE_KEYS
        or state["format"] != "dope-mixed-state"
        or type(state["version"]) is not int
        or state["version"] != 1
        or type(state["profile"]) is not str
        or state["profile"] not in _MIXED_PROFILES
        or type(state["constant_target"]) is not bool
        or type(state["affine"]) is not dict
        or type(state["features"]) is not list
        or not state["features"]
    ):
        _fail("mixed_state_invalid")
    for feature in state["features"]:
        split = feature["split"] if type(feature) is dict else None
        if (
            type(feature) is not dict
            or set(feature) != _MIXED_FEATURE_KEYS
            or type(feature["id"]) is not str
            or type(feature["start"]) is not int
            or type(feature["categories"]) is not list
            or any(type(item) is not str for item in feature["categories"])
            or not _number(feature["completion"])
            or type(feature["probabilities"]) is not list
            or any(not _number(item) for item in feature["probabilities"])
        ):
            _fail("mixed_state_invalid")
        if split is None:
            continue
        if (
            type(split) is not dict
            or set(split) != _MIXED_SPLIT_KEYS
            or type(split["anchor"]) is not int
            or not _number(split["threshold"])
            or type(split["left"]) is not list
            or type(split["right"]) is not list
            or any(not _number(item) for item in split["left"])
            or any(not _number(item) for item in split["right"])
        ):
            _fail("mixed_state_invalid")
    return state["profile"]


def account_mixed_artifact(artifact, limits=None):
    """Charge one DOPMIX01 carrier after BLAKE3(state||DPK) matches bytes 24..56.

    Private affine, dictionaries, completions, probabilities, and splits count
    toward the cap and are not returned. The kernel is not decoded. DOPERAW1
    and DOPERAW2 stay on account_raw_artifact. A passed charge is not
    confirmation, a right, an official asset, a release, formal DP, or HIPAA
    de-identification.
    """
    from research.transfer.bundle import BundleError, BundleLimits
    from research.transfer.bundle_io import preflight_dpk

    limits = limits or BundleLimits()
    if type(artifact) is not bytes or len(artifact) < MIXED_HEADER or len(artifact) > MIXED_MAX_BYTES:
        _fail("mixed_artifact_invalid")
    if artifact[:8] != MIXED_MAGIC:
        _fail("mixed_artifact_invalid")
    state_len = int.from_bytes(artifact[8:16], "little")
    payload_len = int.from_bytes(artifact[16:24], "little")
    if (
        state_len > MIXED_MAX_STATE
        or MIXED_HEADER + state_len + payload_len != len(artifact)
        or blake3_hash(artifact[MIXED_HEADER:]) != artifact[24:56]
    ):
        _fail("mixed_artifact_integrity_failed")
    profile = _mixed_profile(artifact[MIXED_HEADER:MIXED_HEADER + state_len])
    try:
        preflight_dpk(artifact[MIXED_HEADER + state_len:], limits)
    except BundleError:
        _fail("mixed_payload_limits_invalid")
    charge = {
        "format": "dope-mixed-sampling-artifact",
        "version": 1,
        "profile": profile,
        "byte_profile_id": MIXED_BYTE_PROFILE,
        "capability": "bounded_numeric_nominal_features_five_masks_continuous_regression",
        "support": "fit_range_numeric_one_hot_nominal_fit_only_empirical_classes_optional_one_numeric_split",
        "privacy": "unmeasured_private_affine_dictionary_completion_probabilities_retained",
        "certification": "unqualified_development_artifact",
        "framing_bytes": MIXED_HEADER,
        "transform_schema_bytes": state_len,
        "native_payload_bytes": payload_len,
        "complete_sampling_bytes": len(artifact),
        "transport_archive_bytes": len(artifact),
        "generic_runtime_bytes": None,
        "cold_sampling_closure_bytes": None,
        "mapped_runtime_cold_closure": "separate_unmeasured",
        "mathematical_parameter_count": None,
        "content_hash": "blake3_state_plus_dpk",
        "declared_columns_checked": False,
        "confirmation": "unqualified",
        "rights": "unqualified",
        "official_assets": "unqualified",
        "formal_dp": False,
        "hipaa_deidentification": False,
        "release_authorized": False,
        "tail_fidelity": "unqualified",
        "learner_transfer": "unqualified",
    }
    return MappingProxyType(charge)


def declare_raw_columns(columns):
    """Refuse mixed, nominal, integer, boolean, and mask states. Ignore dropped non-model roles.

    Source names and cell values are not accepted. Feature support stays inside the
    historical fit range. Successor target sampling is unclipped and not tail-qualified.
    """
    if type(columns) is not list or not columns:
        _fail("raw_profile_requires_continuous_columns")
    features = 0
    targets = 0
    for column in columns:
        if type(column) is not dict or set(column) - _PUBLIC_KEYS:
            _fail("schema_mismatch")
        role = column.get("role")
        dtype = column.get("dtype")
        identity = column.get("id")
        if type(role) is not str or type(dtype) is not str or type(identity) is not str:
            _fail("schema_mismatch")
        if not 1 <= len(identity) <= 64 or role not in _KNOWN_ROLES:
            _fail("schema_mismatch")
        for optional in ("label", "units"):
            if optional in column and (type(column[optional]) is not str or len(column[optional]) > 64):
                _fail("schema_mismatch")
        if role not in _FEATURE_ROLES:
            continue
        if dtype != "continuous":
            _fail("raw_profile_requires_continuous_columns")
        mask_states = column.get("mask_states", [])
        if type(mask_states) is not list or any(type(item) is not str for item in mask_states):
            _fail("schema_mismatch")
        if mask_states:
            _fail("raw_profile_requires_observed_cells")
        if role == "target":
            targets += 1
        else:
            features += 1
    if features < 1 or targets != 1:
        _fail("raw_sampler_closure_mismatch")
    declared = profile_limits()
    declared["feature_count"] = features
    declared["target_count"] = 1
    declared["declared_columns_checked"] = True
    return MappingProxyType(declared)
