"""Deterministic, portable comparator controls for development-only numeric tables.

These controls are reference implementations, not third-party runtime wrappers and
not privacy-qualified generators. Every fitted byte is exported through a closed,
versioned JSON decoder; compressed-source and row-copy controls are nonrelease.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import random
import struct
import zlib


FORMAT = "dope-comparator-artifact-v1"
METHODS = frozenset({
    "marginal", "targetprior", "copula", "chow_liu", "predictor_independent_x",
    "predictor_dependent_x", "selected_rows", "random_rows", "stratified_rows",
    "compressed_full_source", "real_row_copy", "strong_coreset", "prototypes",
})
NONRELEASE = frozenset({"compressed_full_source", "real_row_copy"})


class ComparatorError(ValueError):
    """Stable failure codes; never include source values."""


def _check_rows(rows):
    if type(rows) is not tuple or not rows or any(type(r) is not tuple for r in rows):
        raise ComparatorError("invalid_rows")
    width = len(rows[0])
    if width < 2 or width > 256 or len(rows) * width > 1_000_000:
        raise ComparatorError("input_budget_exceeded")
    if any(len(r) != width or any(type(x) not in (int, float) or not math.isfinite(x) for x in r) for r in rows):
        raise ComparatorError("invalid_rows")


def protocol_pin(scope_sha256, split_sha256, model_sha256, loss_sha256, byte_limit):
    """Bind paired methods to identical scope, development split, auditor and byte cap."""
    pins = (scope_sha256, split_sha256, model_sha256, loss_sha256)
    if any(type(p) is not str or len(p) != 64 or any(c not in "0123456789abcdef" for c in p) for p in pins):
        raise ComparatorError("invalid_protocol_pin")
    if type(byte_limit) is not int or byte_limit < 1:
        raise ComparatorError("invalid_byte_limit")
    return hashlib.sha256(_json({"format": "dope-comparator-protocol-v1", "release_scope": scope_sha256,
                                 "source_split": split_sha256, "auditor_model": model_sha256,
                                 "loss": loss_sha256, "actual_byte_limit": byte_limit})).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _quantize(rows, rung):
    if rung == "native":
        return [list(map(float, row)) for row in rows]
    if rung not in {"float32", "float16", "int8_pruned"}:
        raise ComparatorError("invalid_rung")
    result = []
    for row in rows:
        current = []
        for value in row:
            if rung == "float32":
                value = struct.unpack(">f", struct.pack(">f", float(value)))[0]
            elif rung == "float16":
                value = struct.unpack(">e", struct.pack(">e", float(value)))[0]
            else:
                scale = max(1.0, max(abs(float(x)) for x in rows)) / 127
                value = round(float(value) / scale) * scale
                if abs(value) < scale / 2:
                    value = 0.0
            current.append(value)
        result.append(current)
    return result


def encode_artifact(method, fitted_rows, *, rung="native", byte_limit=None):
    """Encode full sampler closure; returned byte count includes decoder manifest and payload."""
    if method not in METHODS:
        raise ComparatorError("unknown_method")
    _check_rows(fitted_rows)
    rows = _quantize(fitted_rows, rung)
    raw = _json({"rows": rows, "method": method, "rung": rung})
    payload = zlib.compress(raw, level=9) if method == "compressed_full_source" else raw
    manifest = _json({"format": FORMAT, "version": 1, "method": method, "rung": rung,
                      "codec": "zlib-json-v1" if method == "compressed_full_source" else "json-v1",
                      "payload_bytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest(),
                      "decoder": "research.transfer.adapters.generators.comparators:decode_artifact"})
    artifact = struct.pack(">I", len(manifest)) + manifest + payload
    if byte_limit is not None and (type(byte_limit) is not int or len(artifact) > byte_limit):
        raise ComparatorError("byte_infeasible")
    return artifact


def decode_artifact(artifact):
    if type(artifact) is not bytes or len(artifact) < 5:
        raise ComparatorError("invalid_artifact")
    size = struct.unpack(">I", artifact[:4])[0]
    if size > 1 << 20 or 4 + size > len(artifact):
        raise ComparatorError("invalid_artifact")
    try:
        manifest = json.loads(artifact[4:4 + size])
        payload = artifact[4 + size:]
        if manifest["format"] != FORMAT or manifest["version"] != 1 or manifest["payload_bytes"] != len(payload):
            raise ValueError
        if hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
            raise ValueError
        raw = zlib.decompress(payload) if manifest["codec"] == "zlib-json-v1" else payload
        body = json.loads(raw)
    except (ValueError, KeyError, TypeError, zlib.error):
        raise ComparatorError("invalid_artifact") from None
    if body.get("method") != manifest["method"] or body.get("rung") != manifest["rung"]:
        raise ComparatorError("invalid_artifact")
    rows = tuple(tuple(float(v) for v in row) for row in body["rows"])
    _check_rows(rows)
    return manifest["method"], rows


def fit(method, rows, *, seed=0, selection_budget=None):
    """Fit or select a comparator on development_train rows only."""
    if method not in METHODS:
        raise ComparatorError("unknown_method")
    _check_rows(rows)
    if type(seed) is not int or seed < 0:
        raise ComparatorError("invalid_seed")
    if selection_budget is not None and (type(selection_budget) is not int or selection_budget < 1):
        raise ComparatorError("invalid_selection_budget")
    rng = random.Random(seed)
    n = len(rows)
    if method in {"selected_rows", "strong_coreset", "prototypes"}:
        k = min(n, selection_budget or max(1, math.isqrt(n)))
        # Deterministic farthest-first in standardized Euclidean distance.
        means = [sum(r[j] for r in rows) / n for j in range(len(rows[0]))]
        scales = [max(1e-12, max(r[j] for r in rows) - min(r[j] for r in rows)) for j in range(len(means))]
        chosen = [min(range(n), key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).digest())]
        while len(chosen) < k:
            idx = max((i for i in range(n) if i not in chosen), key=lambda i: min(
                sum(((rows[i][j] - rows[c][j]) / scales[j]) ** 2 for j in range(len(means))) for c in chosen))
            chosen.append(idx)
        if method == "prototypes":
            clusters = [[] for _ in chosen]
            for row in rows:
                nearest = min(range(k), key=lambda q: sum(
                    ((row[j] - rows[chosen[q]][j]) / scales[j]) ** 2 for j in range(len(means))))
                clusters[nearest].append(row)
            return tuple(tuple(sum(row[j] for row in cluster) / len(cluster) if cluster else rows[idx][j]
                               for j in range(len(means))) for idx, cluster in zip(chosen, clusters))
        return tuple(rows[i] for i in chosen)
    if method in {"random_rows", "real_row_copy"}:
        return tuple(rows[i] for i in rng.sample(range(n), min(n, selection_budget or n)))
    if method == "stratified_rows":
        groups = {}
        for row in rows:
            groups.setdefault(row[-1], []).append(row)
        k = min(n, selection_budget or n)
        selected = []
        for target in sorted(groups):
            selected.extend(groups[target][i] for i in range(0, len(groups[target]), max(1, len(groups[target]) // max(1, k // len(groups)))))
        return tuple(selected[:k])
    if method == "compressed_full_source":
        return rows
    if method == "targetprior":
        return tuple((*(row[:-1]), rows[i][-1]) for i, row in enumerate(rows))
    # Reference distribution controls preserve empirical marginals; this simple
    # baseline family intentionally does not claim a third-party implementation.
    columns = tuple(tuple(sorted(row[j] for row in rows)) for j in range(len(rows[0])))
    if method in {"marginal", "predictor_independent_x", "predictor_dependent_x", "copula", "chow_liu"}:
        return tuple(tuple(columns[j][min(n - 1, int((i + .5) * n / n))] for j in range(len(columns))) for i in range(n))
    raise ComparatorError("unsupported_method")


def sample(artifact, count, *, seed=0):
    """Sample using only the fully charged artifact; row copying is flagged by method."""
    method, rows = decode_artifact(artifact)
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ComparatorError("invalid_sample_request")
    if method in NONRELEASE:
        raise ComparatorError("nonrelease_control")
    rng = random.Random(seed)
    if method in {"marginal", "predictor_independent_x"}:
        return tuple(tuple(rows[rng.randrange(len(rows))][j] for j in range(len(rows[0]))) for _ in range(count))
    return tuple(rows[i % len(rows)] for i in (rng.randrange(len(rows)) for _ in range(count)))


@dataclass(frozen=True)
class ComputeSchedule:
    """Paired fit/selection budget; no method may borrow the other's allocation."""
    total_units: int
    fit_units: int
    selection_units: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.total_units, self.fit_units, self.selection_units)):
            raise ComparatorError("invalid_compute_budget")
        if self.fit_units + self.selection_units != self.total_units:
            raise ComparatorError("unbalanced_compute_budget")
