"""Private typed-control persistence; reload always requires current executor authority."""
from dataclasses import asdict
from pathlib import Path

from .campaign_artifacts import create_private, read_regular, private_root
from .pilot_control_materialized_draws import (
    TrustedCompileBinding, TypedControlDraw, control_pair, require, sha,
)
from .pilot_materialized_draws import DrawSpec, _native_json
from .tdl.control_boundary import canonical, MAX_ARTIFACT

FORMAT = "dope-private-typed-control-pair-v1"
SHARED = ("schema_document", "input_document", "compiler_source", "compile_receipt", "admission_manifest")
PER_DRAW = ("raw", "receipt", "request")


def store_pair(directory, primary, sensitivity, *, artifact):
    control_pair(primary, sensitivity)
    require(read_regular(artifact, MAX_ARTIFACT) == primary.artifact == sensitivity.artifact)
    root = private_root(directory) / "materialized-draws"
    root.mkdir(mode=0o700)  # no overwrite or reuse
    pins = {}
    def write(name, raw):
        create_private(root / name, raw, 0o400)
        pins[name] = sha(raw)
    for name in SHARED:
        require(getattr(primary, name) == getattr(sensitivity, name))
        write(name + ".bin", getattr(primary, name))
    rows = []
    for draw in (primary, sensitivity):
        for name in PER_DRAW:
            write(draw.spec.scale + "." + name, getattr(draw, name))
        rows.append({"spec": asdict(draw.spec), "returncode": draw.returncode,
                     "row_ids_sha256": sha(canonical(list(draw.row_ids)))})
    manifest = {"format": FORMAT, "binding": asdict(primary.binding), "draws": rows, "files": pins,
                "runtime_pins": _native_json(primary.request)["closure_pins"],
                "scientific_admission": False, "pipeline_complete": False}
    create_private(root / "manifest.json", canonical(manifest), 0o400)
    return sha(canonical(manifest))


def load_pair(directory, expected_manifest_sha256, *, artifact, runtime_files, trusted_verifier):
    """Fixed paths only; byte hashes do not replace the mandatory live verifier."""
    require(callable(trusted_verifier))
    root = private_root(Path(directory) / "materialized-draws")
    raw = read_regular(root / "manifest.json")
    require(sha(raw) == expected_manifest_sha256)
    value = _native_json(raw)
    require(set(value) == {"format", "binding", "draws", "files", "runtime_pins",
                          "scientific_admission", "pipeline_complete"}
            and value["format"] == FORMAT and value["scientific_admission"] is False
            and value["pipeline_complete"] is False)
    names = {n + ".bin" for n in SHARED} | {
        scale + "." + n for scale in ("primary_n", "sensitivity_4n") for n in PER_DRAW}
    require(type(value["files"]) is dict and set(value["files"]) == names)
    blobs = {n: read_regular(root / n) for n in sorted(names)}
    require(all(sha(blob) == value["files"][n] for n, blob in blobs.items()))
    require({path: sha(blob) for path, blob in runtime_files} == value["runtime_pins"])
    binding_doc = dict(value["binding"])
    require(type(binding_doc["ordered_column_ids"]) is list)
    binding_doc["ordered_column_ids"] = tuple(binding_doc["ordered_column_ids"])
    binding = TrustedCompileBinding(**binding_doc)
    require(type(value["draws"]) is list and len(value["draws"]) == 2)
    artifact_raw, draws = read_regular(artifact, MAX_ARTIFACT), []
    for scale, row in zip(("primary_n", "sensitivity_4n"), value["draws"]):
        require(type(row) is dict and set(row) == {"spec", "returncode", "row_ids_sha256"})
        spec = DrawSpec(**row["spec"])
        require(spec.scale == scale)
        draw = TypedControlDraw(spec, binding, *(blobs[scale + "." + n] for n in PER_DRAW),
            artifact_raw, *(blobs[n + ".bin"] for n in SHARED), runtime_files,
            row["returncode"], trusted_verifier)
        require(sha(canonical(list(draw.row_ids))) == row["row_ids_sha256"])
        draws.append(draw)
    control_pair(*draws)
    return tuple(draws)
