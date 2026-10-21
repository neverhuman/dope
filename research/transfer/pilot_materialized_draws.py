"""Private, bounded n/4n sample handoff; no launcher or execution authority.

The admitted bridge must reserve the existing StageJournal before producing
these bytes. This module validates and retains results; it never draws again,
changes a quota, trains a model, or establishes scientific qualification.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from .campaign_artifacts import MAX_INPUT, create_private, file_hash, private_root, read_regular
from .campaign_runner import CampaignRunnerError, _require
from .evidence import canonical_json, digest

SCALES = ("primary_n", "sensitivity_4n")
METHODS = ("dope", "gaussian_copula", "compressed_real_rows")
SEED_HEX = re.compile(r"0x[0-9a-f]{16}\Z")
PIN = re.compile(r"[0-9a-f]{64}\Z")
FORMAT = "dope-private-materialized-pilot-draws-v2"


def seed_hex(value):
    """Exact native uint64, represented within dope-finite-json-v1 as text."""
    _require(type(value) is int and 0 <= value < 1 << 64, "invalid_native_draw_seed")
    return "0x" + format(value, "016x")


def _pin(value):
    _require(type(value) is str and PIN.fullmatch(value) is not None, "invalid_draw_binding")


@dataclass(frozen=True)
class DrawSpec:
    scale: str
    method: str
    rows: int
    columns: int  # features, then the target; query labels never enter this API
    seed_hex: str
    artifact_sha256: str
    schema_sha256: str
    input_sha256: str
    byte_budget: int

    def __post_init__(self):
        _require(self.scale in SCALES and self.method in METHODS, "invalid_draw_binding")
        _require(type(self.rows) is int and self.rows > 0 and type(self.columns) is int
                 and self.columns >= 2 and self.rows * self.columns * 8 <= MAX_INPUT
                 and self.rows * 67 + 64 <= MAX_INPUT,  # bounded JSON ordered-ID sidecar too
                 "invalid_draw_dimensions")
        _require(type(self.seed_hex) is str and SEED_HEX.fullmatch(self.seed_hex) is not None,
                 "invalid_native_draw_seed")
        for value in (self.artifact_sha256, self.schema_sha256, self.input_sha256):
            _pin(value)
        _require(type(self.byte_budget) is int and 0 < self.byte_budget <= 1 << 30,
                 "invalid_draw_binding")

    @property
    def seed(self):
        return int(self.seed_hex, 16)


def _require_supported(spec):
    _require(type(spec) is DrawSpec, "invalid_draw_binding")
    # The real control producer supplies no authenticated PublicColumn schema.
    # Do not accept a width-only receipt or an invented optional schema field.
    _require(spec.method == "dope", "pilot_v2_control_runtime_unavailable")


def _native_json(raw):
    # Native receipts may contain uint64 seeds. Preserve their original bytes;
    # do not relax or feed those integers into the finite-json-v1 codec.
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_INPUT, "invalid_draw_receipt")

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "invalid_draw_receipt")
            result[key] = value
        return result

    def constant(_):
        raise CampaignRunnerError("invalid_draw_receipt")

    def finite_float(value):
        result = float(value)
        _require(math.isfinite(result), "invalid_draw_receipt")
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant,
                           parse_float=finite_float)
    except (ValueError, UnicodeError, RecursionError):
        raise CampaignRunnerError("invalid_draw_receipt") from None
    _require(type(value) is dict, "invalid_draw_receipt")
    return value


def _schema_projection(raw, spec):
    """Project the pinned private schema using Rust schema::PublicColumn fields.

    Source names and domains never enter the projection. Rust validates the
    complete schema at compile time; this verifies the received column contract.
    """
    _require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == spec.schema_sha256,
             "draw_schema_mismatch")
    try:
        value = _native_json(raw)
        _require(value.get("format") == "dope-table-schema" and type(value.get("version")) is int
                 and value["version"] == 1 and type(value.get("columns")) is list
                 and len(value["columns"]) == spec.columns, "draw_schema_mismatch")
        approved = value.get("metadata_visibility", "opaque_ids") == "approved_semantic_labels"
        states = ("missing", "unknown_category", "structural_absence", "padding", "censored")
        result = []
        for column in value["columns"]:
            _require(type(column) is dict and all(type(column.get(k)) is str for k in ("id", "role", "dtype")),
                     "draw_schema_mismatch")
            masks = {token["state"] for token in column.get("mask_tokens", [])}
            if (column.get("domain") or {}).get("kind") == "categorical":
                masks.add("unknown_category")
            _require(masks <= set(states), "draw_schema_mismatch")
            public = {k: column[k] for k in ("id", "role", "dtype")}
            public["mask_states"] = [s for s in states if s in masks]
            for private_key, public_key in (("semantic_label", "label"), ("units", "units")):
                if approved and column.get(private_key) is not None:
                    public[public_key] = column[private_key]
            result.append(public)
        return result
    except (CampaignRunnerError, ValueError, TypeError, KeyError, AttributeError):
        raise CampaignRunnerError("draw_schema_mismatch") from None


def _require_canonical_schema(raw):
    """New model pools and persisted handoffs require canonical target-last.

    Retained historical draw decoding separately preserves its original schema
    projection, native receipt and physical identities; it grants no model use.
    """
    from .tdl.target_last import TargetLastError, schema_columns, schema_document
    try:
        schema_columns(schema_document(raw))
    except TargetLastError:
        raise CampaignRunnerError("draw_target_must_be_last") from None


@dataclass(frozen=True, repr=False)
class MaterializedDraw:
    spec: DrawSpec
    raw: bytes
    native_receipt: bytes
    schema_document: bytes
    control_request: bytes | None = None

    def __post_init__(self):
        _require(type(self.spec) is DrawSpec and type(self.raw) is bytes
                 and type(self.native_receipt) is bytes and type(self.schema_document) is bytes,
                 "invalid_materialized_draw")
        _require(len(self.raw) == self.spec.rows * self.spec.columns * 8,
                 "invalid_draw_dimensions")
        _require(all(math.isfinite(v) for (v,) in struct.iter_unpack("<d", self.raw)),
                 "nonfinite_draw")
        _validate_draw(self)

    @property
    def sample_sha256(self):
        return hashlib.sha256(self.raw).hexdigest()

    @property
    def row_ids(self):
        # Opaque generated-row identity, with scale/seed/provenance and order.
        # Identical sampled values are allowed; row identity is not a value hash.
        binding = {"spec": asdict(self.spec), "sample_sha256": self.sample_sha256}
        return tuple(digest({**binding, "row": i}) for i in range(self.spec.rows))

    def pool(self, *, forbidden_row_ids=()):
        _require_canonical_schema(self.schema_document)
        from .adapters.base import FeatureBatch, TrainingBatch
        from .pilot_learning_contract import Pool

        _require(not set(self.row_ids) & set(forbidden_row_ids), "draw_row_identity_collision")
        values = tuple(v for (v,) in struct.iter_unpack("<d", self.raw))
        rows = tuple(values[i:i + self.spec.columns]
                     for i in range(0, len(values), self.spec.columns))
        training = TrainingBatch(FeatureBatch(self.row_ids, tuple(r[:-1] for r in rows),
                                              self.spec.schema_sha256),
                                 tuple(r[-1] for r in rows), "synthetic_train")
        return Pool(training, "synthetic_draw", self.spec.input_sha256,
                    self.spec.artifact_sha256, self.spec.rows, self.spec.rows, self.spec.seed_hex)


def decode_draw(raw, native_receipt, spec, *, schema_document, control_request=None):
    """Reverify original retained stream/receipt bytes, without granting model use.

    Historical schema order stays intact. pool() and new store_pair() enforce
    mandatory target-last before any new training pool or persisted handoff.
    """
    _require_supported(spec)
    _require(control_request is None, "draw_receipt_mismatch")
    return MaterializedDraw(spec, raw, native_receipt, schema_document, control_request)


def _validate_draw(result):
    spec, raw, native_receipt = result.spec, result.raw, result.native_receipt
    _require_supported(spec)
    expected_schema = _schema_projection(result.schema_document, spec)
    report = _native_json(native_receipt)
    _require(report.get("state") == "sampled" and type(report.get("rows")) is int
             and report["rows"] == spec.rows and type(report.get("seed")) is int
             and report["seed"] == spec.seed and report.get("sample_sha256") == result.sample_sha256,
             "draw_receipt_mismatch")
    _require(result.control_request is None, "draw_receipt_mismatch")
    expected = {"format": "f64le", "layout": "row_major", "column_order": "features_then_target",
                "rows": spec.rows, "columns": spec.columns, "bytes": len(raw)}
    stream, inspection = report.get("private_sample_stream"), report.get("inspection")
    _require(type(stream) is dict and stream == expected and
             all(type(stream[k]) is int for k in ("rows", "columns", "bytes")),
             "draw_receipt_mismatch")
    _require(report.get("filesystem_network_exec") == "denied"
             and report.get("inherited_source_fds") == "closed"
             and report.get("isolation") == "linux_x86_64_seccomp_tsync_default_deny",
             "draw_boundary_unverified")
    _require(type(inspection) is dict and inspection.get("schema") == expected_schema,
             "draw_schema_mismatch")
    accounting = inspection.get("accounting")
    _require(type(accounting) is dict and type(accounting.get("complete_sampling_bytes")) is int
             and 0 < accounting["complete_sampling_bytes"] <= spec.byte_budget,
             "draw_artifact_over_budget")


def store_pair(directory, primary, sensitivity, *, artifact):
    """Persist two distinct already-executed draws, once; never invoke sampling."""
    _require(type(primary) is MaterializedDraw and type(sensitivity) is MaterializedDraw,
             "invalid_materialized_draw")
    _require_supported(primary.spec)
    _require_supported(sensitivity.spec)
    _require_canonical_schema(primary.schema_document)
    _require_canonical_schema(sensitivity.schema_document)
    root = private_root(directory)
    left, right = primary.spec, sensitivity.spec
    _require((left.scale, right.scale) == SCALES and right.rows == left.rows * 4
             and left.seed_hex != right.seed_hex, "draw_pair_mismatch")
    fields = ("method", "columns", "artifact_sha256", "schema_sha256", "input_sha256", "byte_budget")
    _require(all(getattr(left, k) == getattr(right, k) for k in fields), "draw_pair_mismatch")
    _require(primary.schema_document == sensitivity.schema_document, "draw_schema_mismatch")
    artifact_raw = read_regular(artifact)
    _require(hashlib.sha256(artifact_raw).hexdigest() == left.artifact_sha256,
             "draw_artifact_changed")
    _validate_draw(primary)
    _validate_draw(sensitivity)
    for draw in (primary, sensitivity):
        report = _native_json(draw.native_receipt)
        charged_artifact = report["inspection"]["accounting"]["complete_sampling_bytes"]
        _require(type(charged_artifact) is int and charged_artifact == len(artifact_raw),
                 "draw_artifact_charge_mismatch")
    dest = root / "materialized-draws"
    try:
        dest.mkdir(mode=0o700)
    except OSError:
        raise CampaignRunnerError("draw_replay_refused") from None
    create_private(dest / "schema.json", primary.schema_document, 0o400)
    entries = []
    for draw in (primary, sensitivity):
        name = draw.spec.scale
        ids = canonical_json({"row_ids": list(draw.row_ids)})
        create_private(dest / (name + ".f64le"), draw.raw, 0o400)
        create_private(dest / (name + ".native.json"), draw.native_receipt, 0o400)
        create_private(dest / (name + ".row_ids.json"), ids, 0o400)
        entry = {"spec": asdict(draw.spec), "sample_sha256": draw.sample_sha256,
                 "receipt_sha256": hashlib.sha256(draw.native_receipt).hexdigest(),
                 "row_ids_sha256": hashlib.sha256(ids).hexdigest()}
        entries.append(entry)
    _require(file_hash(artifact) == left.artifact_sha256, "draw_artifact_changed")
    manifest = {"format": FORMAT, "draws": entries, "execution_or_scientific_admission": False}
    create_private(dest / "manifest.json", canonical_json(manifest), 0o400)
    return manifest


def load_pair(directory, expected_manifest_sha256, *, artifact, comparator="dope", control_runtime=None):
    """Reverify retained bytes, exact artifact and pinned manifest; no admission grant."""
    if comparator != "dope":
        from .pilot_control_runtime import PilotControlRuntime
        _require(comparator in ("gaussian_copula", "compressed_real_rows")
                 and type(control_runtime) is PilotControlRuntime,
                 "pilot_v2_control_runtime_unavailable")
        draws = control_runtime.load_pair(directory, expected_manifest_sha256, artifact=artifact)
        _require(all(d.spec.method == comparator for d in draws), "draw_pair_mismatch")
        return draws
    _require(control_runtime is None, "draw_runtime_mismatch")
    dest = private_root(Path(directory) / "materialized-draws")
    _pin(expected_manifest_sha256)
    raw = read_regular(dest / "manifest.json")
    _require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256, "draw_manifest_changed")
    value = _native_json(raw)
    _require(set(value) == {"format", "draws", "execution_or_scientific_admission"}
             and value["format"] == FORMAT and value["execution_or_scientific_admission"] is False
             and type(value["draws"]) is list and len(value["draws"]) == 2, "invalid_draw_manifest")
    result = []
    schema_document = read_regular(dest / "schema.json")
    for expected_scale, entry in zip(SCALES, value["draws"]):
        _require(type(entry) is dict, "invalid_draw_manifest")
        try:
            spec = DrawSpec(**entry["spec"])
        except (TypeError, KeyError):
            raise CampaignRunnerError("invalid_draw_manifest") from None
        _require_supported(spec)
        keys = {"spec", "sample_sha256", "receipt_sha256", "row_ids_sha256"}
        _require(set(entry) == keys and spec.scale == expected_scale, "invalid_draw_manifest")
        _require(file_hash(artifact) == spec.artifact_sha256, "draw_artifact_changed")
        data = read_regular(dest / (expected_scale + ".f64le"))
        receipt = read_regular(dest / (expected_scale + ".native.json"))
        ids = read_regular(dest / (expected_scale + ".row_ids.json"))
        _require(hashlib.sha256(data).hexdigest() == entry.get("sample_sha256")
                 and hashlib.sha256(receipt).hexdigest() == entry.get("receipt_sha256")
                 and hashlib.sha256(ids).hexdigest() == entry.get("row_ids_sha256"), "draw_payload_changed")
        draw = decode_draw(data, receipt, spec, schema_document=schema_document)
        artifact_raw = read_regular(artifact)
        charged_artifact = _native_json(receipt)["inspection"]["accounting"]["complete_sampling_bytes"]
        _require(type(charged_artifact) is int and charged_artifact == len(artifact_raw),
                 "draw_artifact_charge_mismatch")
        _require(_native_json(ids) == {"row_ids": list(draw.row_ids)}, "draw_row_identity_changed")
        result.append(draw)
    left, right = result
    _require(right.spec.rows == left.spec.rows * 4 and left.spec.seed_hex != right.spec.seed_hex
             and all(getattr(left.spec, k) == getattr(right.spec, k) for k in
                     ("method", "columns", "artifact_sha256", "schema_sha256", "input_sha256", "byte_budget")),
             "draw_pair_mismatch")
    _require(file_hash(artifact) == left.spec.artifact_sha256, "draw_artifact_changed")
    return tuple(result)
