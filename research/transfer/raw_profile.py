"""Charge the native raw-complete profile. Rust owns the digest, reload, and sampling.

The length identity is 56 + fitted-state bytes + native payload bytes. The
32-byte content digest is not checked here. Generic runtime and cold closure
stay None until a separate inventory measures them. A passed charge is not
confirmation, a right, an official asset, a release, formal DP, or HIPAA
de-identification. Historical DPK tier totals are a different account.
"""

from types import MappingProxyType

RAW_MAGIC = b"DOPERAW1"
RAW_HEADER = 56
RAW_MAX_BYTES = 32 * 1024 * 1024
RAW_HANDLER = "native-raw-complete-v1"
RAW_KIND = "doperaw1"
RAW_CAPABILITIES = ["byte-inventory", "native-raw-sample"]
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


def account_raw_artifact(artifact, limits=None):
    """Split one self-contained artifact. Does not decode the fitted state or payload."""
    from research.transfer.bundle import BundleError, BundleLimits
    from research.transfer.bundle_io import preflight_dpk

    limits = limits or BundleLimits()
    if type(artifact) is not bytes or len(artifact) < RAW_HEADER or len(artifact) > RAW_MAX_BYTES:
        _fail("raw_artifact_invalid")
    if artifact[:8] != RAW_MAGIC:
        _fail("raw_artifact_invalid")
    state_len = int.from_bytes(artifact[8:16], "little")
    payload_len = int.from_bytes(artifact[16:24], "little")
    if state_len == 0 or state_len > len(artifact) or payload_len > len(artifact):
        _fail("raw_artifact_invalid" if state_len == 0 else "raw_artifact_integrity_failed")
    if RAW_HEADER + state_len + payload_len != len(artifact):
        _fail("raw_artifact_integrity_failed")
    payload = artifact[RAW_HEADER + state_len:]
    try:
        preflight_dpk(payload, limits)
    except BundleError:
        _fail("raw_payload_limits_invalid")
    charge = {
        "byte_profile_id": RAW_HANDLER,
        "framing_bytes": RAW_HEADER,
        "transform_schema_bytes": state_len,
        "native_payload_bytes": payload_len,
        "complete_sampling_bytes": len(artifact),
        "transport_archive_bytes": len(artifact),
        "generic_runtime_bytes": None,
        "cold_sampling_closure_bytes": None,
    }
    charge.update(profile_limits())
    # Length identity does not reload columns. Mixed and mask states stay unsupported.
    charge["declared_columns_checked"] = False
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
