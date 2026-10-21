"""Versioned complete receiver state for the optional Python table example.

JSON v2 preserves binary64 parameters and typed target classes. DOPEENC1 is an
unconsumed diagnostic prototype, not a complete reloadable receiver format.
No source table, estimator, pickle, or executable object is loaded here.
"""
from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import asdict
from pathlib import Path

FORMAT = "dope-table-encoder-v2"
MAX_STATE_BYTES = 1_048_576
ERROR = "encoder_state_invalid"
OPTIONS = {"numeric", "categorical", "onehot_max_levels", "clip_quantile",
           "skew_threshold", "smoothing", "max_levels", "min_level_share", "drop"}
FIELDS = {"name", "kind", "lo", "hi", "log", "integer", "levels", "codes", "boolean", "knots"}
ROOT = {"format", "target", "task", "classes", "target_range", "options", "columns", "encoded_columns"}


def _need(ok):
    if not ok:
        raise ValueError(ERROR)


def _number(x):
    return type(x) in (int, float) and math.isfinite(x)


def _name(x):
    return type(x) is str and 0 < len(x) <= 4096 and "\0" not in x


def _label(x):
    # numpy scalar normalization is explicit; unsupported objects are refused.
    if type(x).__module__.startswith("numpy") and hasattr(x, "item"):
        x = x.item()
    _need(type(x) in (str, bool, int, float))
    _need(_name(x) if type(x) is str else type(x) is bool or _number(x))
    return x


def dump_state(encoder):
    """Serialize every field consumed by transform/inverse_transform, without rounding."""
    encoder._require_fit()
    state = {
        "format": FORMAT, "target": encoder.target, "task": encoder.task,
        "classes": [_label(c) for c in encoder.classes],
        "target_range": [encoder.target_lo, encoder.target_hi],
        "options": {key: list(encoder.drop) if key == "drop" else getattr(encoder, key)
                    for key in sorted(OPTIONS)},
        "columns": [asdict(spec) for spec in encoder.columns],
        "encoded_columns": encoder.encoded_columns,
    }
    # Validate producer state with the same contract as the receiver.
    restore_state(state)
    _need(len(json.dumps(state, allow_nan=False).encode()) <= MAX_STATE_BYTES)
    return state


def _column(row):
    from dope_tables import ColumnTransform
    _need(type(row) is dict and set(row) == FIELDS)
    _need(_name(row["name"]) and row["kind"] in ("numeric", "datetime", "binary", "ordered", "onehot"))
    _need(all(type(row[k]) is bool for k in ("log", "integer", "boolean")))
    _need(all(_number(row[k]) for k in ("lo", "hi")) and row["hi"] > row["lo"])
    levels, codes, knots = (row[k] for k in ("levels", "codes", "knots"))
    _need(all(type(v) is list for v in (levels, codes, knots)))
    _need(len(levels) <= 64 and all(_name(v) for v in levels))
    _need(len(codes) <= 64 and all(_number(v) and 0 <= v <= 1 for v in codes))
    _need(len(knots) <= 257 and all(_number(v) for v in knots))
    _need(all(a <= b for a, b in zip(knots, knots[1:])))
    kind = row["kind"]
    if kind in ("numeric", "datetime"):
        _need(not levels and not codes and not row["boolean"])
        _need(not knots or len(knots) == 257)
    else:
        _need(not knots and not row["log"] and not row["integer"])
        _need(len(levels) >= 2)
        _need(kind == "binary" or len(set(levels)) == len(levels))
        if kind == "binary":
            _need(len(levels) == 2 and not codes)
        elif kind == "ordered":
            _need(len(codes) == len(levels) and all(a < b for a, b in zip(codes, codes[1:])))
        else:
            _need(not codes)
    return ColumnTransform(**row)


def restore_state(state):
    """Construct a fresh fitted receiver; errors never echo labels, schema or values."""
    from dope_tables import TableEncoder
    try:
        _need(type(state) is dict and set(state) == ROOT and state["format"] == FORMAT)
        _need(_name(state["target"]) and state["task"] in ("binary", "regression"))
        options = state["options"]
        _need(type(options) is dict and set(options) == OPTIONS)
        _need(options["numeric"] in ("minmax", "robust", "quantile"))
        _need(options["categorical"] in ("auto", "target_ordered", "onehot"))
        _need(type(options["max_levels"]) is int and 2 <= options["max_levels"] <= 64)
        _need(type(options["onehot_max_levels"]) is int and 2 <= options["onehot_max_levels"] <= 64)
        _need(_number(options["clip_quantile"]) and 0 <= options["clip_quantile"] < .5)
        _need(all(_number(options[k]) and options[k] >= 0 for k in ("skew_threshold", "smoothing")))
        _need(_number(options["min_level_share"]) and 0 <= options["min_level_share"] <= 1)
        _need(type(options["drop"]) is list and len(options["drop"]) <= 4096)
        _need(all(_name(v) for v in options["drop"]))
        rows = state["columns"]
        _need(type(rows) is list and 1 <= len(rows) <= 4096)
        columns = [_column(row) for row in rows]
        names = [c.name for c in columns]
        _need(len(set(names)) == len(names) and state["target"] not in names)
        classes, bounds = state["classes"], state["target_range"]
        _need(type(classes) is list and type(bounds) is list and len(bounds) == 2)
        _need(all(_number(v) for v in bounds) and bounds[1] > bounds[0])
        if state["task"] == "binary":
            _need(len(classes) == 2)
            classes = [_label(v) for v in classes]
            _need(classes[0] != classes[1])
        else:
            _need(not classes)
        receiver = TableEncoder(state["target"], task=state["task"], **options)
        receiver.columns, receiver.source_columns = columns, names
        receiver.classes = classes
        receiver.target_lo, receiver.target_hi = bounds
        receiver._fitted = True
        encoded = state["encoded_columns"]
        _need(type(encoded) is list and encoded == receiver.encoded_columns)
        _need(len(set(encoded)) == len(encoded) and encoded[-1] == receiver.target)
        return receiver
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise ValueError(ERROR) from None


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        _need(key not in out)
        out[key] = value
    return out


def load_encoder(path: str | Path):
    """Bounded regular-file-only reload. Never follows a final symlink or waits on a FIFO."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            _need(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_STATE_BYTES)
            raw = stream.read(MAX_STATE_BYTES + 1)
        _need(len(raw) == info.st_size and len(raw) <= MAX_STATE_BYTES)
        state = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda _: _need(False))
        return restore_state(state)
    except (OSError, ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise ValueError(ERROR) from None
