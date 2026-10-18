"""Directory reads and DPK preflight for offline bundle inventory.

Rust owns decoding and sampling. These helpers bound paths and headers.
They do not decode payloads, draw samples, open a network, or spawn a process.
"""

from __future__ import annotations

import hashlib
import os
import stat

from research.transfer.bundle import (
    BundleError,
    BundleLimits,
    DPK2,
    DPK3,
    HEADER_BYTES,
    NEURAL_CAP,
    QUANTIZATION,
    U64_MAX,
    _fail,
)


def _read_bytes(root, path, limit):
    pieces = path.split("/")
    directory = None
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        for part in pieces[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        fd = os.open(
            pieces[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=directory,
        )
        with os.fdopen(fd, "rb") as stream:
            mode = os.fstat(stream.fileno()).st_mode
            if not stat.S_ISREG(mode):
                _fail("dependency_not_regular_file")
            hasher = hashlib.sha256()
            size = 0
            chunks = []
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > limit:
                    _fail("dependency_too_large")
                hasher.update(block)
                chunks.append(block)
            return hasher.hexdigest(), size, b"".join(chunks)
    except BundleError:
        raise
    except OSError as exc:
        if exc.errno == 40:
            _fail("symlink_component")
        _fail("dependency_io_failed")
    finally:
        if directory is not None:
            os.close(directory)


def _directory_names(directory):
    """Yield one name at a time from an already opened directory fd."""
    duplicate = None
    try:
        duplicate = os.dup(directory)
        with os.scandir("/proc/self/fd/%d" % duplicate) as entries:
            for entry in entries:
                yield entry.name
    except BundleError:
        raise
    except OSError:
        _fail("dependency_io_failed")
    finally:
        if duplicate is not None:
            os.close(duplicate)


def _scan_open_directory(directory, declared, max_entries, max_name_bytes):
    prefixes = set()
    for path in declared:
        parts = path.split("/")
        for index in range(1, len(parts)):
            prefixes.add("/".join(parts[:index]))
    found = set()
    _walk_directory(directory, "", declared, prefixes, found, max_entries, max_name_bytes)
    return found


def _walk_directory(directory, relative, declared, prefixes, found, max_entries, max_name_bytes):
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    entry_count = 0
    name_bytes = 0
    for name in _directory_names(directory):
        if name in (".", ".."):
            continue
        raw_name = os.fsencode(name)
        try:
            raw_name.decode("utf-8")
        except UnicodeDecodeError:
            _fail("unsafe_path")
        entry_count += 1
        name_bytes += len(raw_name)
        if entry_count > max_entries or name_bytes > max_name_bytes:
            _fail("too_many_components")
        if name == "" or "/" in name or "\\" in name or "\x00" in name:
            _fail("unsafe_path")
        child = name if relative == "" else relative + "/" + name
        try:
            mode = os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode
        except OSError as exc:
            if exc.errno == 40:
                _fail("symlink_component")
            _fail("dependency_io_failed")
        if stat.S_ISLNK(mode):
            _fail("symlink_component")
        if stat.S_ISDIR(mode):
            if child not in prefixes:
                _fail("undeclared_component")
            try:
                child_directory = os.open(name, directory_flags, dir_fd=directory)
            except OSError as exc:
                if exc.errno == 40:
                    _fail("symlink_component")
                _fail("dependency_io_failed")
            try:
                _walk_directory(
                    child_directory, child, declared, prefixes, found, max_entries, max_name_bytes
                )
            finally:
                os.close(child_directory)
        elif stat.S_ISREG(mode):
            if child not in declared:
                _fail("undeclared_component")
            found.add(child)
        else:
            _fail("dependency_not_regular_file")


def _inventory(root, declared, max_entries, max_name_bytes):
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        root_directory = os.open(root, directory_flags)
    except OSError as exc:
        if exc.errno == 40:
            _fail("symlink_component")
        _fail("dependency_io_failed")
    try:
        found = _scan_open_directory(root_directory, declared, max_entries, max_name_bytes)
    finally:
        os.close(root_directory)
    if found != set(declared):
        _fail("unresolved_dependency")


def _varint(data, offset):
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            _fail("truncated_artifact" if offset >= len(data) else "noncanonical_varint")
        byte = data[offset]
        offset += 1
        piece = byte & 0x7F
        if shift == 63 and piece > 1:
            _fail("noncanonical_varint")
        value |= piece << shift
        if byte & 0x80 == 0:
            if shift and piece == 0:
                _fail("noncanonical_varint")
            if value > U64_MAX:
                _fail("integer_overflow")
            return value, offset
        shift += 7


def preflight_dpk(artifact, limits=None):
    """Bound a DPK header and section lengths without decompressing or decoding."""
    limits = limits or BundleLimits()
    if type(artifact) is not bytes or len(artifact) < HEADER_BYTES:
        _fail("truncated_artifact")
    if len(artifact) > limits.max_component_bytes:
        _fail("component_cap_exceeded")
    magic = artifact[:4]
    if magic == DPK3:
        v3 = True
        if artifact[4] != 3 or artifact[5] > 3:
            _fail("unsupported_dpk_version")
    elif magic == DPK2:
        v3 = False
        if artifact[4] != 2 or artifact[5] > 1:
            _fail("unsupported_dpk_version")
    else:
        _fail("unsupported_dpk_version")
    if artifact[6] != 1:
        _fail("unsupported_dpk_version")
    if artifact[7] not in (0, 1):
        _fail("unsupported_dpk_header")
    if artifact[8] not in QUANTIZATION:
        _fail("unsupported_quantization")
    flags = int.from_bytes(artifact[10:12], "little")
    if flags & ~1:
        _fail("invalid_flags")
    features = int.from_bytes(artifact[12:16], "little")
    if features == 0 or features > limits.max_features:
        _fail("invalid_dimensions")
    payload_length = int.from_bytes(artifact[32:40], "little")
    if payload_length != len(artifact) - HEADER_BYTES:
        _fail("truncated_artifact")
    payload = artifact[HEADER_BYTES:]
    maximum_tag = 6 if v3 and artifact[5] >= 2 else 5
    offset = 0
    prior = 0
    expanded = 0
    saw_neural = False
    while offset < len(payload):
        tag_byte = payload[offset]
        offset += 1
        tag = tag_byte & 0x7F
        if tag == 0 or tag > maximum_tag or tag <= prior:
            _fail("unknown_section")
        prior = tag
        if tag == 6:
            saw_neural = True
            if len(artifact) > NEURAL_CAP:
                _fail("neural_cap_exceeded")
        encoded_length, offset = _varint(payload, offset)
        if tag_byte & 0x80:
            raw_length, offset = _varint(payload, offset)
            if raw_length == 0:
                _fail("invalid_dimensions")
        else:
            raw_length = encoded_length
        if encoded_length > len(payload) - offset:
            _fail("truncated_artifact")
        body = payload[offset:offset + encoded_length]
        if tag == 6 and tag_byte & 0x80 == 0 and (b"NaN" in body or b"Infinity" in body):
            _fail("nonfinite_value")
        if raw_length > limits.max_expanded_section_bytes:
            _fail("expansion_cap_exceeded")
        expanded += raw_length
        if expanded > limits.max_expanded_section_bytes:
            _fail("expansion_cap_exceeded")
        offset += encoded_length
    if saw_neural and not v3:
        _fail("unknown_section")
    return {"features": features, "minor": artifact[5], "v3": v3}


def recipient_accounting(bundle):
    """Measured declared inventory; no sharing credit without independence proof.

    Runtime labels do not prove public source, licensing, or independence. They
    therefore remain fully in R(A,H), including at K > 1. Generated rows are a
    separate delivery scope; required/reachable rows are already in the closure.
    External inventory is not an executable-sampler or release qualification.
    Historical account_inventory estimates remain available for v1 replay only.
    """
    from research.transfer.bundle import VerifiedBundle, checked_sum

    if type(bundle) is not VerifiedBundle:
        _fail("verification_required")
    a = bundle.accounting
    sections = {name: a[name + "_bytes"] for name in (
        "sampling_payload", "shared_learned_state", "runtime_package",
        "evidence_authorization", "materialized_rows", "transform", "dictionary",
    )}
    sections["runtime"] = sections.pop("runtime_package")
    component_bytes = checked_sum(sections.values())
    physical_bytes = checked_sum((component_bytes, a["manifest_bytes"]))
    cold = a["charged_sampling_bytes"]
    if cold > physical_bytes:
        _fail("inconsistent_byte_ledger")
    return {
        "format": "dope.recipient-inventory/v1",
        "status": "inventory_verified",
        "sampling_closure_qualification": "unmeasured",
        "scope": "declared_recipient_bundle",
        "amortization_status": "unverified_full_charge",
        "dataset_specific_bytes": cold,
        "declared_runtime_bytes": a["runtime_package_bytes"],
        "shared_source_independent_decoder_bytes": None,
        "cold_closure_bytes": cold,
        "amortized_k_closure_bytes": cold,
        "amortization_count": a["amortization_count"],
        "generated_samples_bytes": a["materialized_rows_bytes"],
        "component_bytes": component_bytes,
        "manifest_framing_bytes": a["manifest_bytes"],
        "physical_file_bytes": physical_bytes,
        "nonrecipient_delivery_bytes": physical_bytes - cold,
        "complete_delivery_bytes": a.get("complete_delivery_bytes"),
        "section_bytes": sections,
    }
