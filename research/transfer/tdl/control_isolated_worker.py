"""Dedicated fresh child only. All imports precede irreversible boundaries."""
import os


def main():
    # Bootstrap is inside the redacted error boundary, including source imports.
    import hashlib
    import json
    import math
    from pathlib import Path
    import struct
    import sys
    import zlib

    runtime = Path(__file__).absolute().parent
    sys.path.insert(0, str(runtime))
    from control_boundary import (MAX_ARTIFACT, MAX_VALUES, SOURCES, canonical, digest,
                                  load_boundary, regular_bytes, require)
    from literal_worker import _control_document, _control_sample, _literal_payload, _read_control_bytes

    request_raw = regular_bytes(runtime / "request.json", MAX_ARTIFACT)
    request = json.loads(request_raw)
    require(type(request["abi"]) is int and request["abi"] == 1 and set(request["source_pins"]) == set(SOURCES))
    for name in SOURCES:
        require(digest(regular_bytes(runtime / name)) == request["source_pins"][name])
    require(digest(canonical(request["source_pins"])) == request["source_sha256"])
    library = load_boundary(runtime / "boundary.so", request["library_sha256"], request["abi"])
    # Exercise codecs before sealing; no sampling or artifact parsing occurs here.
    json.loads("{}")
    zlib.decompress(zlib.compress(b"warm"))
    hashlib.sha256(b"warm").hexdigest()
    struct.pack("<d", 0.0)
    math.erf(0.0)
    # Loaded file mappings must be part of the externally pinned runtime closure.
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "/" not in line:
            continue
        path = line[line.index("/"):]
        if path == "/memfd:dope-control-runtime (deleted)":
            continue  # immutable sealed DSO already pinned before dlopen
        require(not path.endswith(" (deleted)"))
        expected = request["closure_pins"].get(path)
        require(expected is not None and digest(regular_bytes(path)) == expected)
    require(library.confine_current_bundle_directory() == 0)
    # The only admitted read phase is the private bundle cwd, with no source FDs.
    raw = _read_control_bytes("control.tdl", MAX_ARTIFACT)
    require(len(raw) == request["artifact_bytes"] and digest(raw) == request["artifact_sha256"])
    require(library.seal_sampling_process() == 0)
    document = _control_document(_literal_payload(raw), request["method"])
    rows, seed, columns = request["rows"], request["seed"], document["columns"]
    require(type(rows) is int and 0 < rows and rows * columns <= MAX_VALUES)
    sampled = _control_sample(document, rows, seed)
    require(len(sampled) == rows and all(len(row) == columns for row in sampled))
    encoded = b"".join(struct.pack("<d", value) for row in sampled for value in row)
    require(len(encoded) == rows * columns * 8)
    receipt = {key: request[key] for key in (
        "method", "rows", "seed", "abi", "artifact_sha256", "artifact_bytes", "library_sha256",
        "source_sha256", "runtime_closure_sha256", "runtime_bytes")}
    receipt.update(state="sampled", format="f64le", layout="row_major", columns=columns,
                   column_order="features_then_target", sample_bytes=len(encoded),
                   sample_sha256=digest(encoded), isolation="existing_rust_landlock_seccomp_tsync",
                   invocation_bytes=len(request_raw),
                   sampling_closure_bytes=request["artifact_bytes"] + request["runtime_bytes"] + len(request_raw),
                   pipeline_complete=False, scientific_admission=False, citable=False)
    # Private bytes are accepted only with a complete receipt AND successful exit.
    public = canonical(receipt) + b"\n"
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(public)
    sys.stderr.buffer.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never emit parser, loader, pathname, source-value or partial-sample diagnostics.
        try:
            os.write(2, b"control_boundary_refused\n")
        finally:
            os._exit(1)
