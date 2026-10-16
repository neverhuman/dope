"""vNext byte-bound evidence; hashes do not establish truth or release authority.

Only verification issues VerifiedEvaluationReceipt. The policy must come from a
trusted executor/custodian, independently of the candidate receipt. All file
references are local, relative, non-symlink paths. Final caches require the same
confirmation-role policy and must remain inaccessible to development producers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

_ROOT = Path(__file__).resolve().parents[2]
_MAX_INTEGER = 2**53 - 1
_MAX_MANIFEST = 16 * 1024 * 1024
_TOKEN = object()
_IDENTITY_SCHEMA = json.loads((_ROOT / "schemas/transfer-run-v1.schema.json").read_text())["$defs"]["EvaluationIdentity"]


class EvidenceError(ValueError):
    """Stable diagnostics never include file paths or source values."""


def _fail(code):
    raise EvidenceError(code)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def canonical_json(value) -> bytes:
    """Sorted UTF-8 JSON; integers <=2^53-1, floats with 17 fractional digits.

    This is dope-finite-json-v1, not RFC 8785. Float exponents have no leading
    zero or plus sign. Float and integer scalar representations stay distinct.
    """
    def encode(x):
        if x is None:
            return "null"
        if type(x) is bool:
            return "true" if x else "false"
        if type(x) is int:
            if abs(x) > _MAX_INTEGER:
                _fail("integer_out_of_range")
            return str(x)
        if type(x) is float:
            if not math.isfinite(x):
                _fail("nonfinite_json_number")
            mantissa, exponent = format(x, ".17e").split("e")
            return mantissa + "e" + str(int(exponent))
        if type(x) is str:
            return json.dumps(x, ensure_ascii=False)
        if type(x) is list:
            return "[" + ",".join(encode(v) for v in x) + "]"
        if type(x) is dict and all(type(k) is str for k in x):
            return "{" + ",".join(encode(k) + ":" + encode(x[k]) for k in sorted(x)) + "}"
        _fail("invalid_json_scalar")
    try:
        return encode(value).encode("utf-8")
    except (UnicodeError, RecursionError):
        _fail("invalid_json_encoding")


def parse_json(data: bytes):
    try:
        value = json.loads(data, object_pairs_hook=_pairs,
                           parse_constant=lambda _: _fail("nonfinite_json_number"))
        canonical_json(value)
        return value
    except EvidenceError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        _fail("invalid_json_encoding")


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def admit_generator_evidence(state):
    """Refuse probe output that has no verified vendor run.

    vendor_executed False is the recorded non-execution flag. Any other value
    is not a verified vendor receipt and does not qualify a claim.
    """
    if type(state) is not dict or state.get("vendor_executed") is False:
        _fail("vendor_not_executed")
    _fail("vendor_execution_unverified")


def _validate(value, schema):
    """Validate the closed schema subset used by the three pinned v1 schemas."""
    known = {'$schema', '$id', '$defs', 'title', 'type', 'anyOf', 'allOf', 'if', 'then', 'else',
             'const', 'enum', 'required', 'properties', 'additionalProperties', 'items',
             'minItems', 'uniqueItems', 'minLength', 'pattern', 'minimum', 'maximum'}
    if set(schema) - known:
        _fail('unsupported_schema_keyword')
    if "anyOf" in schema:
        for alternative in schema["anyOf"]:
            try:
                _validate(value, alternative)
                break
            except EvidenceError:
                pass
        else:
            _fail("schema_mismatch")
    expected = schema.get("type")
    checks = {"object": type(value) is dict, "array": type(value) is list,
              "string": type(value) is str, "integer": type(value) is int,
              "number": type(value) in (int, float), "boolean": type(value) is bool,
              "null": value is None}
    if expected and not checks[expected]:
        _fail("schema_mismatch")
    if "const" in schema and canonical_json(value) != canonical_json(schema["const"]):
        _fail("schema_mismatch")
    if "enum" in schema and not any(canonical_json(value) == canonical_json(v) for v in schema["enum"]):
        _fail("schema_mismatch")
    if type(value) is dict:
        if any(k not in value for k in schema.get("required", [])):
            _fail("schema_mismatch")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            _fail("schema_mismatch")
        for key in value.keys() & properties.keys():
            _validate(value[key], properties[key])
    if type(value) is list:
        if len(value) < schema.get("minItems", 0):
            _fail("schema_mismatch")
        if schema.get("uniqueItems") and len({canonical_json(v) for v in value}) != len(value):
            _fail("schema_mismatch")
        for item in value:
            _validate(item, schema.get("items", {}))
    if type(value) is str:
        if len(value) < schema.get("minLength", 0) or ("pattern" in schema and not re.fullmatch(schema["pattern"], value)):
            _fail("schema_mismatch")
    if type(value) in (int, float):
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            _fail("schema_mismatch")
    for branch in schema.get("allOf", []):
        _validate(value, branch)
    if "if" in schema:
        try:
            _validate(value, schema["if"])
            matched = True
        except EvidenceError:
            matched = False
        _validate(value, schema.get("then" if matched else "else", {}))


def validate_behavior_contract(value):
    canonical_json(value)
    _validate(value, json.loads((_ROOT / "schemas/behavior-contract-v1.schema.json").read_text()))
    if set(value['learners']['required_families']) & set(value['learners']['held_out_families']):
        _fail('learner_holdout_overlap')


class EvaluationIdentity:
    def __init__(self, value):
        canonical_json(value)
        _validate(value, _IDENTITY_SCHEMA)
        self._bytes = canonical_json(value)

    @property
    def value(self):
        return parse_json(self._bytes)


def scientific_cache_key(identity: EvaluationIdentity) -> str:
    return digest({"format": "dope-scientific-cache-key", "version": 1,
                   "identity": identity.value})


def fit_replicate_key(identity: EvaluationIdentity) -> str:
    """Path/attempt/sample/learner aliases cannot become independent fits."""
    fields = ("data_hash", "split_hash", "generator_hash", "configuration_hash",
              "feature_transform_hash", "target_transform_hash", "generator_fit_seed", "global_seed")
    return digest({"format": "dope-fit-replicate", "version": 1,
                   "identity": {k: identity.value[k] for k in fields}})


class VerificationPolicy:
    def __init__(self, identity: EvaluationIdentity, executor_hash: str,
                 receipt_hash: str, freeze_hash: str | None = None):
        self.identity = EvaluationIdentity(identity.value)
        self.executor_hash = executor_hash
        self.receipt_hash = receipt_hash
        self.freeze_hash = freeze_hash
        for pin in (executor_hash, receipt_hash):
            if type(pin) is not str or not re.fullmatch("[0-9a-f]{64}", pin):
                _fail("invalid_trust_pin")
        if freeze_hash is not None and (type(freeze_hash) is not str or not re.fullmatch("[0-9a-f]{64}", freeze_hash)):
            _fail("invalid_trust_pin")


class UnverifiedReceipt:
    def __init__(self, data: bytes):
        self._bytes = canonical_json(parse_json(data))


class VerifiedEvaluationReceipt:
    def __init__(self, data: bytes, token=None):
        if token is not _TOKEN:
            _fail("verification_required")
        self._bytes = data

    @property
    def value(self):
        return parse_json(self._bytes)

    @property
    def identity(self):
        return EvaluationIdentity(self.value["identity"])

    @property
    def receipt_hash(self):
        return hashlib.sha256(self._bytes).hexdigest()


def _read_file(root: Path, path: str, manifest: bool):
    pieces = path.split("/")
    if any(part in ("", ".", "..") for part in pieces) or "\\" in path or "\x00" in path:
        _fail("unsafe_dependency_path")
    directory = None
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in pieces[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(pieces[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                _fail("dependency_not_regular_file")
            hasher, size, chunks = hashlib.sha256(), 0, []
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                hasher.update(chunk)
                if manifest:
                    if size > _MAX_MANIFEST:
                        _fail("manifest_too_large")
                    chunks.append(chunk)
            return hasher.hexdigest(), size, b"".join(chunks)
    except OSError:
        _fail("dependency_io_failed")
    finally:
        if directory is not None:
            os.close(directory)


def _closed(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        _fail("invalid_dependency_manifest")


def verify_receipt(unverified: UnverifiedReceipt, root: Path,
                   policy: VerificationPolicy) -> VerifiedEvaluationReceipt:
    receipt = parse_json(unverified._bytes)
    _validate(receipt, json.loads((_ROOT / "schemas/transfer-run-v1.schema.json").read_text()))
    if digest(receipt) != policy.receipt_hash or receipt["identity"] != policy.identity.value:
        _fail("receipt_identity_mismatch")
    if receipt["executor_hash"] != policy.executor_hash or receipt["freeze_hash"] != policy.freeze_hash:
        _fail("executor_or_freeze_mismatch")
    if receipt["execution"] != "complete":
        _fail("evaluation_incomplete")
    if receipt["projection"] != "private":
        _fail("private_evidence_required")
    identity = receipt["identity"]
    if not (identity["produced_rows"] == identity["requested_rows"] and
            0 < identity["consumed_rows"] <= identity["produced_rows"] and
            identity["context_rows"] <= identity["consumed_rows"]):
        _fail("row_count_mismatch")
    nodes, paths = {}, set()
    for node in receipt["dependencies"]:
        if node["sha256"] in nodes or node["path"] in paths:
            _fail("dependency_alias")
        nodes[node["sha256"]] = node
        paths.add(node["path"])
    roots = {v for k, v in identity.items() if k.endswith("_hash")} | {receipt["executor_hash"], receipt["paired_evidence_hash"]}
    if receipt["freeze_hash"]:
        roots.add(receipt["freeze_hash"])
    visited, active = set(), set()
    def visit(key):
        if key not in nodes:
            _fail("dependency_missing")
        if key in active:
            _fail("dependency_cycle")
        if key in visited:
            return
        active.add(key)
        for child in nodes[key]["requires"]:
            visit(child)
        active.remove(key)
        visited.add(key)
    for key in roots:
        visit(key)
    if visited != set(nodes):
        _fail("unbound_dependency")
    manifests = {}
    for key, node in nodes.items():
        manifest = node["kind"] in ("protocol", "split", "fit", "artifact_inventory", "paired_evidence", "freeze")
        actual_hash, size, data = _read_file(Path(root), node["path"], manifest)
        if actual_hash != key or size != node["bytes"]:
            _fail("dependency_bytes_mismatch")
        if manifest:
            manifests[key] = parse_json(data)
    for field, key in identity.items():
        if field.endswith("_hash") and nodes[key]["kind"] != field[:-5]:
            _fail("dependency_kind_mismatch")
    if nodes[receipt["executor_hash"]]["kind"] != "executor" or nodes[receipt["paired_evidence_hash"]]["kind"] != "paired_evidence":
        _fail("dependency_kind_mismatch")
    split = manifests[identity['split_hash']]
    split_fields = ('data_hash', 'query_hash', 'partition_role', 'fit_rows', 'query_rows')
    _closed(split, ('format', 'version', *split_fields))
    for field in split_fields:
        _validate(split[field], _IDENTITY_SCHEMA['properties'][field])
    if split['format'] != 'dope-split-manifest' or type(split['version']) is not int or split['version'] != 1 or any(split[k] != identity[k] for k in split_fields):
        _fail('split_identity_mismatch')
    if set(nodes[identity['split_hash']]['requires']) != {identity['data_hash'], identity['query_hash']}:
        _fail('split_dependency_mismatch')
    protocol = manifests[identity["protocol_hash"]]
    validate_behavior_contract(protocol)
    for declared, actual in (('table_schema_hash', 'schema_hash'), ('task_manifest_hash', 'task_hash'), ('population_history_hash', 'source_family_hash')):
        if protocol['workload'][declared] is not None and protocol['workload'][declared] != identity[actual]:
            _fail('protocol_workload_mismatch')
    commitments = {v for v in protocol['freeze_requirements'].values()
                   if v is not None and not (isinstance(v, str) and v.startswith('TBD-'))}
    commitments.update(v for k, v in protocol['workload'].items() if k.endswith('_hash') and v is not None)
    commitments.update(v for k, v in protocol['disclosure'].items() if k.endswith('_hash') and v is not None)
    if not commitments <= set(nodes[identity['protocol_hash']]['requires']):
        _fail('protocol_commitment_unbound')
    inventory = manifests[identity["artifact_inventory_hash"]]
    _closed(inventory, ("format", "version", "files"))
    if inventory["format"] != "dope-artifact-inventory" or type(inventory["version"]) is not int or inventory["version"] != 1:
        _fail("invalid_artifact_inventory")
    artifacts = sorted(key for key, node in nodes.items() if node["kind"] == "artifact")
    if type(inventory["files"]) is not list or any(type(k) is not str for k in inventory['files']) or sorted(inventory["files"]) != artifacts or not artifacts or len(set(inventory["files"])) != len(inventory["files"]):
        _fail("artifact_inventory_mismatch")
    if sorted(nodes[identity["artifact_inventory_hash"]]["requires"]) != artifacts:
        _fail("artifact_inventory_mismatch")
    if sum(nodes[key]["bytes"] for key in artifacts) != receipt["sampling_payload_bytes"]:
        _fail("sampling_byte_sum_mismatch")
    fit = manifests[identity["fit_hash"]]
    fit_fields = ("data_hash", "split_hash", "generator_hash", "configuration_hash", "feature_transform_hash", "target_transform_hash", "generator_fit_seed", "global_seed")
    _closed(fit, ("format", "version", "state_hash", *fit_fields))
    _validate(fit['state_hash'], {'type': 'string', 'pattern': '^[0-9a-f]{64}$'})
    for field in fit_fields:
        _validate(fit[field], _IDENTITY_SCHEMA['properties'][field])
    if fit["format"] != "dope-generator-fit" or type(fit["version"]) is not int or fit["version"] != 1 or any(fit[k] != identity[k] for k in fit_fields):
        _fail("fit_identity_mismatch")
    if fit["state_hash"] not in nodes or nodes[fit["state_hash"]]["kind"] != "fit_state" or set(nodes[identity["fit_hash"]]["requires"]) != {fit["state_hash"], *(fit[k] for k in fit_fields if k.endswith("_hash"))}:
        _fail("fit_state_unbound")
    paired = manifests[receipt["paired_evidence_hash"]]
    _closed(paired, ("format", "version", "query_hash", "metric_hash", "loss", "loss_space", "loss_unit", "real", "synthetic", "null"))
    if paired["format"] != "dope-paired-losses" or type(paired["version"]) is not int or paired["version"] != 1 or any(paired[k] != receipt[k] for k in ("loss", "loss_space", "loss_unit")) or any(paired[k] != identity[k] for k in ("query_hash", "metric_hash")):
        _fail("paired_identity_mismatch")
    if set(nodes[receipt["paired_evidence_hash"]]["requires"]) != {identity["query_hash"], identity["metric_hash"]}:
        _fail("paired_dependency_mismatch")
    for label in ("real", "synthetic", "null"):
        values = paired[label]
        if type(values) is not list or len(values) != identity["query_rows"] or any(type(x) not in (int, float) or x < 0 for x in values):
            _fail("invalid_paired_losses")
        try:
            mean = math.fsum(x / len(values) for x in values)
        except OverflowError:
            _fail('paired_loss_sum_mismatch')
        if not math.isfinite(mean) or not math.isclose(mean, receipt["losses"][label], rel_tol=1e-12, abs_tol=1e-12):
            _fail("paired_loss_sum_mismatch")
    if receipt["freeze_hash"]:
        if nodes[receipt["freeze_hash"]]["kind"] != "freeze":
            _fail("dependency_kind_mismatch")
        freeze = manifests[receipt["freeze_hash"]]
        _validate(freeze, json.loads((_ROOT / "schemas/freeze-v1.schema.json").read_text()))
        if any(freeze[k] != identity[k] for k in ("protocol_hash", "artifact_inventory_hash", "configuration_hash", "split_hash", "evaluation_code_hash")) or freeze["executor_hash"] != policy.executor_hash or freeze["cell_hash"] != scientific_cache_key(policy.identity):
            _fail("frozen_selection_mismatch")
        for field, kind in (("selection_receipt_hash", "selection_receipt"), ("access_ledger_hash", "access_ledger")):
            if freeze[field] not in nodes or nodes[freeze[field]]["kind"] != kind or freeze[field] not in nodes[receipt["freeze_hash"]]["requires"]:
                _fail("freeze_dependency_missing")
        if freeze["parent_freeze_hash"] is not None and (freeze["parent_freeze_hash"] not in nodes or nodes[freeze["parent_freeze_hash"]]["kind"] != "freeze" or freeze["parent_freeze_hash"] not in nodes[receipt["freeze_hash"]]["requires"]):
            _fail("freeze_parent_missing")
        if not manifests[identity["protocol_hash"]]["protocol_frozen"]:
            _fail("protocol_unfrozen")
    if identity["partition_role"] == "confirmation" and not receipt["freeze_hash"]:
        _fail("confirmation_freeze_required")
    return VerifiedEvaluationReceipt(unverified._bytes, _TOKEN)


def commit_completed_receipt(cache_dir: Path, verified: VerifiedEvaluationReceipt) -> Path:
    if type(verified) is not VerifiedEvaluationReceipt:
        _fail("verification_required")
    key = scientific_cache_key(verified.identity)
    envelope = {"format": "dope-scientific-cache", "version": 1, "state": "complete",
                "cache_key": key, "receipt_hash": verified.receipt_hash, "receipt": verified.value}
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = cache_dir / (key + ".json")
    # One fsynced envelope is the completion marker; partial staging files are ignored.
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".staging-", delete=False) as stream:
            stream.write(canonical_json(envelope))
            stream.flush()
            os.fsync(stream.fileno())
            staged = Path(stream.name)
        os.replace(staged, destination)
        directory = os.open(cache_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination
    except OSError:
        _fail("cache_commit_failed")


def load_completed_receipt(cache_dir: Path, root: Path, policy: VerificationPolicy) -> VerifiedEvaluationReceipt:
    key = scientific_cache_key(policy.identity)
    _, _, data = _read_file(Path(cache_dir), key + ".json", True)
    envelope = parse_json(data)
    _closed(envelope, ("format", "version", "state", "cache_key", "receipt_hash", "receipt"))
    if envelope["format"] != "dope-scientific-cache" or type(envelope["version"]) is not int or envelope["version"] != 1 or envelope["state"] != "complete" or envelope["cache_key"] != key or envelope["receipt_hash"] != policy.receipt_hash:
        _fail("cache_incomplete_or_mismatched")
    return verify_receipt(UnverifiedReceipt(canonical_json(envelope["receipt"])), root, policy)
