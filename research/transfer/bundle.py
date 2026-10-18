"""Offline inventory for source-free sampling bundles.

Rust owns decoding and sampling. This module checks manifest identity, the
declared dependency closure, expansion bounds, and measured byte accounting.
It does not decode DPK payloads, draw samples, open a network, or spawn a
process. A verified inventory is not a utility, privacy, or release claim.
Unmeasured transport framing stays labeled instead of being counted as zero.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from types import MappingProxyType

from research.transfer.evidence import EvidenceError, canonical_json, parse_json

FORMAT = "dope.generator-bundle/v1"
VERSION = 1
MANIFEST_NAME = "manifest.json"
JSON_INT_MAX = 2**53 - 1
U64_MAX = 2**64 - 1
HEADER_BYTES = 56
NEURAL_CAP = 16 * 1024 * 1024
DPK3 = b"DPK3"
DPK2 = b"DPK2"
QUANTIZATION = frozenset({4, 6, 8, 10, 12, 16})
HANDLERS = frozenset({"inventory-only", "native-inline-dpk", "native-raw-complete-v1"})
ROLES = frozenset({
    "dictionary",
    "evidence_authorization",
    "materialized_rows",
    "runtime",
    "sampling_payload",
    "shared_learned_state",
    "transform",
})
KINDS = frozenset({"dpk2", "dpk3", "doperaw1", "opaque"})
CAPABILITIES = frozenset({"byte-inventory", "native-dpk-sample", "native-raw-sample"})
TRANSPORT = frozenset({"included", "unmeasured"})
EXECUTABLE_SUFFIXES = frozenset({
    "dll", "dylib", "exe", "jar", "node", "pkl", "py", "pyc", "sh", "so", "wasm",
})
SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
IDENTIFIER = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED = object()
MANIFEST_KEYS = frozenset({
    "amortization_count",
    "bundle_id",
    "capabilities",
    "components",
    "format",
    "sampling_handler",
    "transport_framing",
    "version",
})
COMPONENT_KEYS = frozenset({
    "bytes",
    "consumed_by_sampler",
    "depends_on",
    "executable",
    "handler",
    "id",
    "kind",
    "path",
    "required",
    "role",
    "sha256",
})


class BundleError(ValueError):
    """Stable code. Display text never includes paths, values, or headers."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class BundleLimits:
    def __init__(
        self,
        max_manifest_bytes=1024 * 1024,
        max_component_bytes=16 * 1024 * 1024,
        max_components=32,
        max_expanded_section_bytes=16 * 1024 * 1024,
        max_features=4096,
        max_sample_rows=100_000,
        max_sample_bytes=32 * 1024 * 1024,
        max_total_bytes=32 * 1024 * 1024,
    ):
        self.max_manifest_bytes = max_manifest_bytes
        self.max_component_bytes = max_component_bytes
        self.max_components = max_components
        self.max_expanded_section_bytes = max_expanded_section_bytes
        self.max_features = max_features
        self.max_sample_rows = max_sample_rows
        self.max_sample_bytes = max_sample_bytes
        self.max_total_bytes = max_total_bytes


class VerifiedBundle:
    """Issued only by verify_bundle. Accounting is a private snapshot."""

    def __init__(self, accounting, payload, features, token=None):
        if token is not _VERIFIED:
            _fail("verification_required")
        self._accounting = MappingProxyType(dict(accounting))
        self._payload = None if payload is None else bytes(payload)
        self._features = features

    @property
    def accounting(self):
        return self._accounting

    @property
    def features(self):
        return self._features

    @property
    def retained_payload(self):
        return None if self._payload is None else bytes(self._payload)


def _fail(code):
    raise BundleError(code)


def checked_sum(values):
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            _fail("integer_overflow")
        total += value
        if total > U64_MAX:
            _fail("integer_overflow")
    return total


def within_budget(charged, cap):
    if type(charged) is not int or type(cap) is not int or charged < 0 or cap < 0:
        _fail("invalid_budget")
    return charged <= cap


def charged_sampling_bytes(bundle):
    if type(bundle) is not VerifiedBundle:
        _fail("verification_required")
    return bundle.accounting["charged_sampling_bytes"]


def _u64(value, code="schema_mismatch"):
    if type(value) is not int or isinstance(value, bool) or value < 0 or value > JSON_INT_MAX:
        _fail(code)
    return value


def _object(value):
    if type(value) is not dict:
        _fail("schema_mismatch")
    return value


def _exact_keys(value, keys):
    found = set(_object(value))
    if found - keys:
        _fail("unknown_field")
    if keys - found:
        _fail("schema_mismatch")


def _identifier(value):
    if type(value) is not str or IDENTIFIER.fullmatch(value) is None:
        _fail("schema_mismatch")
    return value


def _sha256(value):
    if type(value) is not str or SHA256_HEX.fullmatch(value) is None:
        _fail("schema_mismatch")
    return value


def _path(value):
    if type(value) is not str or len(value) > 512 or "\\" in value or "\x00" in value or ":" in value:
        _fail("unsafe_path")
    pieces = value.split("/")
    if not 1 <= len(pieces) <= 8 or any(SEGMENT.fullmatch(part) is None for part in pieces):
        _fail("unsafe_path")
    suffix = pieces[-1].rsplit(".", 1)[-1].lower() if "." in pieces[-1] else ""
    if suffix in EXECUTABLE_SUFFIXES:
        _fail("executable_rejected")
    return value


def _sorted_unique(values, code):
    if type(values) is not list or any(type(item) is not str for item in values):
        _fail("schema_mismatch")
    if values != sorted(set(values)):
        _fail(code)
    return values


_IO_EXPORTS = frozenset({"_inventory", "_read_bytes", "_scan_open_directory", "preflight_dpk", "recipient_accounting"})


def __getattr__(name):
    # bundle_io imports this module, so these helpers cannot load while it is initializing.
    if name not in _IO_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from research.transfer import bundle_io
    value = getattr(bundle_io, name)
    globals()[name] = value
    return value


def _role_total(components, role):
    return checked_sum(component["measured_bytes"] for component in components if component["role"] == role)


def account_inventory(components, manifest_bytes, amortization_count, transport_framing):
    if type(amortization_count) is not int or isinstance(amortization_count, bool):
        _fail("invalid_amortization")
    if amortization_count < 1 or amortization_count > 1_000_000:
        _fail("invalid_amortization")
    if transport_framing not in TRANSPORT:
        _fail("schema_mismatch")
    payload = _role_total(components, "sampling_payload")
    shared = _role_total(components, "shared_learned_state")
    transform = _role_total(components, "transform")
    dictionary = _role_total(components, "dictionary")
    runtime = _role_total(components, "runtime")
    evidence = _role_total(components, "evidence_authorization")
    rows = _role_total(components, "materialized_rows")
    learned = checked_sum([
        payload,
        shared,
        transform,
        dictionary,
        manifest_bytes,
    ])
    shared_share = (shared + amortization_count - 1) // amortization_count
    amortized = checked_sum([payload, transform, dictionary, manifest_bytes, shared_share])
    edges = {component["id"]: list(component["depends_on"]) for component in components}
    for component in components:
        for dependency in component["depends_on"]:
            edges.setdefault(dependency, [])
    reachable = set()
    for component in components:
        if component["role"] == "sampling_payload":
            reachable.update(_closure(component["id"], edges))
    conditional = checked_sum(
        component["measured_bytes"]
        for component in components
        if component["role"] in {"evidence_authorization", "materialized_rows"}
        and (component["required"] or component["id"] in reachable)
    )
    charged = checked_sum([learned, runtime, conditional])
    physical = checked_sum(component["measured_bytes"] for component in components)
    accounting = {
        "amortization_count": amortization_count,
        "amortized_learned_state_bytes": amortized,
        "charged_sampling_bytes": charged,
        "dictionary_bytes": dictionary,
        "evidence_authorization_bytes": evidence,
        "manifest_bytes": manifest_bytes,
        "materialized_rows_bytes": rows,
        "runtime_package_bytes": runtime,
        "sampling_payload_bytes": payload,
        "shared_learned_state_bytes": shared,
        "standalone_learned_state_bytes": learned,
        "transform_bytes": transform,
        "transport_framing": transport_framing,
    }
    if transport_framing == "included":
        accounting["complete_delivery_bytes"] = checked_sum([manifest_bytes, physical])
    return accounting


def _closure(start, edges):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(edges[node])
    return seen


def _graph(components, handler):
    by_id = {}
    lowered = {}
    edges = {}
    for component in components:
        identity = component["id"]
        if identity in by_id:
            _fail("duplicate_component")
        by_id[identity] = component
        path_key = component["path"].lower()
        if path_key in lowered:
            _fail("case_collision" if lowered[path_key] != component["path"] else "duplicate_component")
        lowered[path_key] = component["path"]
        edges[identity] = []
    for component in components:
        for dependency in component["depends_on"]:
            if dependency not in by_id:
                _fail("unbound_dependency")
            if dependency == component["id"]:
                _fail("cyclic_dependency")
            edges[component["id"]].append(dependency)
    state = {identity: 0 for identity in by_id}

    def visit(identity):
        state[identity] = 1
        for child in edges[identity]:
            if state[child] == 1:
                _fail("cyclic_dependency")
            if state[child] == 0:
                visit(child)
        state[identity] = 2

    for identity in by_id:
        if state[identity] == 0:
            visit(identity)
    payloads = [component for component in components if component["role"] == "sampling_payload"]
    payload_kinds = {"native-inline-dpk": {"dpk2", "dpk3"}, "native-raw-complete-v1": {"doperaw1"}}.get(handler)
    if payload_kinds is not None:
        if len(payloads) != 1 or not payloads[0]["required"] or not payloads[0]["consumed_by_sampler"] or payloads[0]["kind"] not in payload_kinds:
            _fail("sampler_closure_mismatch")
        if any(component["consumed_by_sampler"] and component is not payloads[0] for component in components):
            _fail("sampler_closure_mismatch")
        reachable = _closure(payloads[0]["id"], edges)
        for component in components:
            if component["required"] and component["id"] not in reachable:
                _fail("unresolved_dependency")
    else:
        if any(component["consumed_by_sampler"] or component["kind"] != "opaque" for component in components):
            _fail("sampler_closure_mismatch")
        if payloads:
            reachable = set()
            for payload in payloads:
                reachable |= _closure(payload["id"], edges)
            for component in components:
                if component["required"] and component["id"] not in reachable:
                    _fail("unresolved_dependency")


def _component(value):
    _exact_keys(value, COMPONENT_KEYS)
    identity = _identifier(value["id"])
    role = value["role"]
    if type(role) is not str or role not in ROLES:
        _fail("unknown_role")
    path = _path(value["path"])
    if path == MANIFEST_NAME:
        _fail("unsafe_path")
    kind = value["kind"]
    if type(kind) is not str or kind not in KINDS:
        _fail("unknown_kind")
    if value["executable"] is not False:
        _fail("executable_rejected")
    if value["handler"] != "none":
        _fail("unsupported_external_handler")
    if type(value["required"]) is not bool or type(value["consumed_by_sampler"]) is not bool:
        _fail("schema_mismatch")
    depends = _sorted_unique(value["depends_on"], "schema_mismatch")
    for item in depends:
        _identifier(item)
    return {
        "id": identity,
        "role": role,
        "path": path,
        "sha256": _sha256(value["sha256"]),
        "bytes": _u64(value["bytes"]),
        "kind": kind,
        "required": value["required"],
        "consumed_by_sampler": value["consumed_by_sampler"],
        "executable": False,
        "handler": "none",
        "depends_on": depends,
    }


def _manifest(value, limits):
    _exact_keys(value, MANIFEST_KEYS)
    if value["format"] != FORMAT:
        _fail("unknown_version")
    # Floats are invalid manifest scalars. Booleans and other non-integers are
    # the wrong version type, matching the native parser's two codes.
    version = value["version"]
    if type(version) is float:
        _fail("invalid_manifest")
    version = _u64(version, "unknown_version")
    if version != VERSION:
        _fail("unknown_version")
    _identifier(value["bundle_id"])
    handler = value["sampling_handler"]
    if type(handler) is not str or handler not in HANDLERS:
        _fail("unknown_handler")
    capabilities = value["capabilities"]
    if type(capabilities) is not list or any(type(item) is not str for item in capabilities):
        _fail("schema_mismatch")
    if any(item not in CAPABILITIES for item in capabilities):
        _fail("unknown_capability")
    expected = {
        "native-inline-dpk": ["byte-inventory", "native-dpk-sample"],
        "native-raw-complete-v1": ["byte-inventory", "native-raw-sample"],
    }.get(handler, ["byte-inventory"])
    if capabilities != expected:
        _fail("sampler_closure_mismatch")
    count = _u64(value["amortization_count"], "invalid_amortization")
    if count < 1 or count > 1_000_000:
        _fail("invalid_amortization")
    framing = value["transport_framing"]
    if framing not in TRANSPORT:
        _fail("schema_mismatch")
    components = value["components"]
    if type(components) is not list or len(components) > limits.max_components:
        _fail("too_many_components" if type(components) is list else "schema_mismatch")
    parsed = [_component(component) for component in components]
    identities = [component["id"] for component in parsed]
    if identities != sorted(identities):
        _fail("schema_mismatch")
    _graph(parsed, handler)
    return handler, count, framing, parsed


def verify_bundle(root, expected_manifest_sha256, limits=None):
    from research.transfer.bundle_io import _inventory, _read_bytes, preflight_dpk

    limits = limits or BundleLimits()
    root = Path(root)
    if type(expected_manifest_sha256) is not str or SHA256_HEX.fullmatch(expected_manifest_sha256) is None:
        _fail("manifest_digest_mismatch")
    try:
        digest, size, data = _read_bytes(root, MANIFEST_NAME, limits.max_manifest_bytes)
    except BundleError as exc:
        if exc.code == "dependency_too_large":
            _fail("manifest_too_large")
        raise
    if size > limits.max_manifest_bytes:
        _fail("manifest_too_large")
    try:
        manifest = parse_json(data)
    except EvidenceError as exc:
        code = str(exc)
        if code in {"duplicate_json_key", "integer_out_of_range", "invalid_json_encoding", "nonfinite_json_number"}:
            _fail(code)
        _fail("invalid_manifest")
    canonical = canonical_json(manifest)
    if canonical != data:
        _fail("manifest_not_canonical")
    if digest != expected_manifest_sha256 or hashlib.sha256(canonical).hexdigest() != expected_manifest_sha256:
        _fail("manifest_digest_mismatch")
    handler, count, framing, components = _manifest(manifest, limits)
    declared = {MANIFEST_NAME} | {component["path"] for component in components}
    max_entries = limits.max_components + 1
    _inventory(root, declared, max_entries, max_entries * 64)
    payload = None
    features = None
    raw_profile = None
    total = size
    for component in components:
        file_digest, file_size, file_bytes = _read_bytes(root, component["path"], limits.max_component_bytes)
        if file_size > limits.max_component_bytes:
            _fail("component_cap_exceeded")
        if file_digest != component["sha256"] or file_size != component["bytes"]:
            _fail("dependency_identity_mismatch")
        component["measured_bytes"] = file_size
        total = checked_sum([total, file_size])
        if component["role"] == "sampling_payload" and handler == "native-inline-dpk":
            summary = preflight_dpk(file_bytes, limits)
            expects_v3 = component["kind"] == "dpk3"
            if summary["v3"] != expects_v3:
                _fail("sampler_closure_mismatch")
            payload = file_bytes
            features = summary["features"]
        elif component["role"] == "sampling_payload" and handler == "native-raw-complete-v1":
            from research.transfer.raw_profile import account_raw_artifact
            raw_profile = account_raw_artifact(file_bytes, limits)
            payload = file_bytes
            features = None
    if total > limits.max_total_bytes:
        _fail("total_bytes_exceeded")
    accounting = account_inventory(components, size, count, framing)
    if raw_profile is not None:
        accounting["raw_profile"] = raw_profile
    return VerifiedBundle(accounting, payload, features, _VERIFIED)
