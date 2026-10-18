"""Fixture-only native worker contract, not an execution/qualification authority.

JSON contains opaque IDs and descriptors; bounded little-endian f64 buffers travel
separately. Only training labels are admitted. The custodian supplies trusted pins
and retains query labels. No pickle, paths, vendor imports or process launch here;
W01/W02/W04 must independently enforce isolation and verify native measurements.
The request digest is the worker cache identity (legacy adapter keys are separate).
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from research.transfer.evidence import _validate, canonical_json, digest, parse_json

MAX_ENVELOPE = 4 * 1024 * 1024
MAX_BUFFER = 64 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
SCHEMA = json.loads((Path(__file__).resolve().parents[3] /
                     "schemas/model-worker-v1.schema.json").read_text())
DATA_METHODS = {"fit_or_prepare_context", "predict", "predict_perturbations"}
PREDICT_METHODS = {"predict", "predict_perturbations"}


class AdapterError(ValueError):
    """Stable codes only: never include source values or backend diagnostics."""


def _require(condition, code):
    if not condition:
        raise AdapterError(code)


def _hash(value):
    _require(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value), "invalid_pin")


def _count(value, minimum=0):
    _require(type(value) is int and minimum <= value <= 2**53 - 1, "invalid_count")


def _finite(value):
    return ((type(value) is int and abs(value) <= 2**53 - 1) or
            (type(value) is float and math.isfinite(value)))


def _cells(rows):
    _require(type(rows) is tuple and bool(rows), "invalid_rows")
    width = len(rows[0]) if type(rows[0]) is tuple else 0
    _require(width > 0, "invalid_rows")
    for row in rows:
        _require(type(row) is tuple and len(row) == width, "invalid_rows")
        _require(all(v is None or type(v) in (str, bool) or _finite(v) for v in row), "invalid_cell")


@dataclass(frozen=True)
class ResourceLimits:
    wall_seconds: float
    memory_bytes: int
    gpu_bytes: int

    def __post_init__(self):
        _require(_finite(self.wall_seconds) and self.wall_seconds > 0, "invalid_budget")
        _count(self.memory_bytes, 1)
        _count(self.gpu_bytes)


def _guard(operation):
    """Suppress parser/codec exception chains as well as their private diagnostics."""
    try:
        return operation()
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, RecursionError,
            UnicodeError, struct.error):
        raise AdapterError("worker_abi_invalid") from None


def _shape(value, kind):
    _require(len(canonical_json(value)) <= MAX_ENVELOPE, "worker_abi_invalid")
    _validate(value, SCHEMA["anyOf"][kind])


def decode_envelope(data):
    def decode():
        _require(type(data) is bytes and len(data) <= MAX_ENVELOPE, "worker_abi_invalid")
        value = parse_json(data)
        _shape(value, 0 if value.get("type") == "request" else 1)
        return value
    return _guard(decode)


def pack_buffer(row_ids, rows):
    """Already-preprocessed numeric cells only; class codes follow pinned class_order."""
    def pack():
        _require(type(row_ids) is tuple and type(rows) is tuple and bool(rows), "worker_abi_invalid")
        columns = len(rows[0])
        _require(0 < len(rows) <= 1048576 and 0 < columns <= 4096 and
                 len(rows) * columns * 8 <= MAX_BUFFER, "worker_abi_invalid")
        _require(len(row_ids) == len(rows) and len(set(row_ids)) == len(row_ids), "worker_abi_invalid")
        for row_id in row_ids:
            _hash(row_id)
        _require(all(type(row) is tuple and len(row) == columns and
                     all(_finite(v) for v in row) for row in rows), "worker_abi_invalid")
        data = b"".join(struct.pack("<d", v) for row in rows for v in row)
        return {"dtype": "f64le", "rows": len(rows), "columns": columns,
                "row_ids": list(row_ids), "sha256": hashlib.sha256(data).hexdigest(),
                "byte_length": len(data)}, data
    return _guard(pack)


def _buffer(descriptor, data):
    _require(type(data) is bytes and len(data) <= MAX_BUFFER and
             descriptor["byte_length"] == len(data) == descriptor["rows"] * descriptor["columns"] * 8,
             "worker_abi_invalid")
    ids = descriptor["row_ids"]
    _require(len(ids) == descriptor["rows"] and len(set(ids)) == len(ids) and
             descriptor["sha256"] == hashlib.sha256(data).hexdigest(), "worker_abi_invalid")
    _require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<d", data)), "worker_abi_invalid")


def _output(output):
    kind, classes = output["kind"], output["class_order"]
    _require(len(set(classes)) == len(classes) and len(classes) <= 4096, "worker_abi_invalid")
    _require((kind == "probability" and bool(classes) and output["units"] == "probability") or
             (not classes and output["units"] == {"point": "target_original",
              "distribution": "distribution_target_original"}.get(kind)), "worker_abi_invalid")


def _probability_sum(values):
    # Specified left-to-right binary64 accumulation, identical in both runtimes.
    total = 0.0
    for value in values:
        total += value
    return total


def _identities(request):
    inputs = request["inputs"]
    request["context_sha256"] = digest({"train_features": inputs["train_features"],
        "train_labels": inputs["train_labels"], "context_row_ids": request["context_row_ids"]})
    request["query_sha256"] = digest(inputs["query_features"])
    request["request_sha256"] = digest({k: v for k, v in request.items() if k != "request_sha256"})


def validate_request(request, buffers, expected_pins):
    """Validate before touching a backend. expected_pins must come from the custodian.

    No runtime or checkpoint self-report establishes a trusted pin. This compares
    identities; native artifact verification belongs to the qualified executor.
    """
    def validate():
        _shape(request, 0)
        _require(request["pins"] == expected_pins and len(request["pins"]["runtime_version"]) <= 128 and
                 re.fullmatch(r"[0-9]+(?:\.[0-9]+){2,}(?:[-+][A-Za-z0-9._-]+)?",
                              request["pins"]["runtime_version"]), "worker_abi_invalid")
        _output(request["output"])
        inputs, counts, policy = request["inputs"], request["counts"], request["policy"]
        _require((policy["subsampling"] == "none" or policy["truncation"] == "none") and
                 (policy["query_policy"] != "rowwise" or policy["query_batch_size"] == 1), "worker_abi_invalid")
        _require(type(buffers) is dict and set(buffers) == {k for k, v in inputs.items() if v is not None}
                 and sum(len(v) for v in buffers.values()) <= MAX_TOTAL, "worker_abi_invalid")
        for name, data in buffers.items():
            _buffer(inputs[name], data)
        active = request["method"] in DATA_METHODS
        predict = request["method"] in PREDICT_METHODS
        _require(all((inputs[k] is not None) == active for k in ("train_features", "train_labels"))
                 and (inputs["query_features"] is not None) == predict, "worker_abi_invalid")
        ids = request["context_row_ids"]
        if active:
            _require(request["feature_schema_sha256"] is not None, "worker_abi_invalid")
            train, labels = inputs["train_features"], inputs["train_labels"]
            n = policy["primary_n"] * policy["sample_multiplier"]
            _require(request["training_role"] is not None and counts["requested_rows"] == n and
                     counts["produced_rows"] == n == counts["provided_rows"] == train["rows"] and
                     labels["rows"] == n and labels["columns"] == 1 and
                     labels["row_ids"] == train["row_ids"], "worker_abi_invalid")
            if request["output"]["kind"] == "probability":
                classes = len(request["output"]["class_order"])
                _require(all(v[0].is_integer() and 0 <= v[0] < classes
                             for v in struct.iter_unpack("<d", buffers["train_labels"])), "worker_abi_invalid")
            _require(0 < counts["consumed_rows"] == len(ids) <= min(n, policy["context_cap"]) and
                     len(set(ids)) == len(ids) and set(ids) <= set(train["row_ids"]), "worker_abi_invalid")
            if policy["subsampling"] == "none":
                expected = min(n, policy["context_cap"]) if policy["truncation"] == "head" else n
                _require(len(ids) == expected and ids == train["row_ids"][:expected], "worker_abi_invalid")
            if predict:
                query = inputs["query_features"]
                _require(query["columns"] == train["columns"] and query["rows"] <= policy["query_batch_size"] and
                         not set(query["row_ids"]) & set(train["row_ids"]) and
                         (policy["query_policy"] != "rowwise" or query["rows"] == 1), "worker_abi_invalid")
        else:
            _require(not ids and request["training_role"] is None and request["feature_schema_sha256"] is None and
                     all(v == 0 for v in counts.values()), "worker_abi_invalid")
        copy = dict(request)
        _identities(copy)
        _require(all(copy[k] == request[k] for k in ("context_sha256", "query_sha256", "request_sha256")),
                 "worker_abi_invalid")
        return request["request_sha256"]
    return _guard(validate)


def build_request(pins, resources, policy, method, seed, *, training=None, query=None,
                  context_row_ids=None, output=None, fit_seed=None):
    """Accept typed training and features only; never a query-label parameter."""
    def build():
        from research.transfer.adapters.base import FeatureBatch, TrainingBatch
        inputs = dict.fromkeys(("train_features", "train_labels", "query_features"))
        buffers = {}
        role, count = None, 0
        if training is not None:
            _require(type(training) is TrainingBatch, "worker_abi_invalid")
            role, count = training.role, len(training.targets)
            for name, rows in (("train_features", training.features.rows),
                               ("train_labels", tuple((y,) for y in training.targets))):
                inputs[name], buffers[name] = pack_buffer(training.features.row_ids, rows)
        if query is not None:
            _require(type(query) is FeatureBatch, "worker_abi_invalid")
            _require(training is not None and query.schema_sha256 == training.features.schema_sha256,
                     "worker_abi_invalid")
            inputs["query_features"], buffers["query_features"] = pack_buffer(query.row_ids, query.rows)
        ids = list(context_row_ids if context_row_ids is not None else
                   (() if training is None else training.features.row_ids))
        request = {"format": "dope-model-worker-v1", "type": "request", "method": method,
            "pins": pins, "resources": resources, "policy": policy, "seed": seed,
            "fit_seed": seed if fit_seed is None else fit_seed,
            "feature_schema_sha256": None if training is None else training.features.schema_sha256,
            "training_role": role, "counts": {"requested_rows": count, "produced_rows": count,
            "provided_rows": count, "consumed_rows": len(ids)}, "inputs": inputs,
            "context_row_ids": ids, "output": output or {"kind": "point", "class_order": [],
                                                         "units": "target_original"}}
        _identities(request)
        # Detach caller-owned dictionaries; mutating policy cannot mutate this request.
        request = parse_json(canonical_json(request))
        validate_request(request, buffers, pins)
        return request, buffers
    return _guard(build)


def validate_response(response, buffers, request, request_buffers, expected_pins):
    """Verify row-aligned finite results and all functional/count/resource bindings."""
    def validate():
        validate_request(request, request_buffers, expected_pins)
        _shape(response, 1)
        _require(type(buffers) is dict, "worker_abi_invalid")
        for key in ("request_sha256", "pins", "context_sha256", "query_sha256", "output"):
            _require(response[key] == request[key], "worker_abi_invalid")
        _require(response["qualified"] is False and response["qualification"] == "fixture_only",
                 "worker_abi_invalid")
        counts = response["counts"]
        for key in ("requested_rows", "produced_rows", "provided_rows"):
            _require(counts[key] == request["counts"][key], "worker_abi_invalid")
        for key, value in response["usage"].items():
            _require(value <= request["resources"][key], "worker_abi_invalid")
        prediction = response["prediction"]
        if response["status"] != "ok":
            _require(response["error"] is not None and prediction is None and not buffers and
                     counts["consumed_rows"] <= request["counts"]["consumed_rows"], "worker_abi_invalid")
        else:
            _require(response["error"] is None and counts == request["counts"], "worker_abi_invalid")
            predict = request["method"] in PREDICT_METHODS
            _require((prediction is not None) == predict and type(buffers) is dict and
                     set(buffers) == ({"prediction"} if predict else set()), "worker_abi_invalid")
            if predict:
                query = request["inputs"]["query_features"]
                _buffer(prediction, buffers["prediction"])
                _require(prediction["row_ids"] == query["row_ids"], "worker_abi_invalid")
                kind = request["output"]["kind"]
                width = prediction["columns"]
                _require((kind != "point" or width == 1) and (kind != "probability" or
                         width == len(request["output"]["class_order"])), "worker_abi_invalid")
                if kind == "probability":
                    values = tuple(v[0] for v in struct.iter_unpack("<d", buffers["prediction"]))
                    _require(all(0 <= p <= 1 for p in values) and
                             all(abs(_probability_sum(values[i:i+width]) - 1) <= 1e-7
                                 for i in range(0, len(values), width)), "worker_abi_invalid")
        return {"request_sha256": request["request_sha256"], "qualified": False,
                "qualification": "fixture_only", "response_sha256": digest(response)}
    return _guard(validate)
