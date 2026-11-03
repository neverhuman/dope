"""Bounded offline NC1 apply oracle. Rust alone fits the product contract.

Real rows/cards remain private. This helper never fits, reads files, dispatches
models, uses network, or claims quality/privacy. Errors contain safe codes only.
"""
import math


class Nc1Error(ValueError):
    """Safe, source-free reference refusal."""


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise Nc1Error("invalid_numeric")
    return float(value)


def _normalize(value, state):
    value = _number(value)
    lo, hi = state["lo"], state["hi"]
    if hi == lo:
        return 0.0
    if value == lo:
        return 0.0
    if value == hi:
        return 1.0
    span, delta = hi - lo, value - lo
    if math.isfinite(span) and math.isfinite(delta):
        out = delta / span
    else:
        scale = max(abs(lo), abs(hi), abs(value))
        out = ((value / scale) - (lo / scale)) / ((hi / scale) - (lo / scale))
    if not math.isfinite(out):
        raise Nc1Error("normalization_overflow")
    return out


def apply_private_card(card, columns, *, with_target=True):
    """Apply Rust-exported owner state to named, raw Python values.

    None is explicitly missing. Category strings are never interpreted as
    missing tokens; booleans are distinct from numbers. No card is refitted.
    """
    if not isinstance(card, dict) or card.get("schema") != "dope.nc1-private/v1" or card.get("recipe") != "nc1-minmax-onehot-v1":
        raise Nc1Error("invalid_private_card")
    if not isinstance(columns, dict) or not columns:
        raise Nc1Error("invalid_input")
    features = card.get("features", [])
    schema = card.get("public", {})
    if not features or len(features) > 256 or schema.get("columns", 0) > 256:
        raise Nc1Error("input_shape_limit")
    try:
        required = [columns[feature["name"]] for feature in features]
        count = len(required[0])
        if count == 0 or count > 1000000 or any(len(v) != count for v in required):
            raise Nc1Error("input_shape_limit")
        target = columns[card["names"][card["declared"]["target"]["column"]]] if with_target else None
        if target is not None and len(target) != count:
            raise Nc1Error("input_shape_limit")
        x, y = [], ([] if with_target else None)
        for index in range(count):
            row = []
            for feature, group, values in zip(features, schema["groups"], required):
                value = values[index]
                missing = value is None
                state = feature["state"]
                kind = state["state"]
                if kind == "numeric":
                    if group["kind"] == "integer" and not missing and _number(value) % 1:
                        raise Nc1Error("invalid_integer")
                    row.append(state["numeric"]["median"] if missing else _normalize(value, state["numeric"]))
                elif kind == "boolean":
                    if not missing and type(value) is not bool:
                        raise Nc1Error("invalid_boolean")
                    row.append(0.0 if missing else float(value))
                elif kind == "ordinal":
                    if missing:
                        row.append(state["median"])
                    elif type(value) is str and value in state["levels"]:
                        row.append(state["levels"].index(value) / (len(state["levels"]) - 1))
                    else:
                        raise Nc1Error("unknown_ordinal")
                elif kind == "nominal":
                    if not missing and type(value) is not str:
                        raise Nc1Error("invalid_nominal")
                    levels = state["levels"]
                    selected = (len(levels) + 1 if feature["has_missing"] else len(levels)) if missing else (
                        levels.index(value) if value in levels else len(levels))
                    row.extend(float(slot == selected) for slot in range(group["width"]))
                else:
                    raise Nc1Error("invalid_private_card")
                if kind != "nominal" and feature["has_missing"]:
                    row.append(float(missing))
            if len(row) != schema["columns"]:
                raise Nc1Error("input_shape_limit")
            x.append(row)
            if target is not None:
                state = card["target"]
                if state["state"] == "regression":
                    y.append(_normalize(target[index], state["numeric"]))
                elif type(target[index]) is str and target[index] in state["labels"]:
                    y.append(float(state["labels"].index(target[index])))
                else:
                    raise Nc1Error("unknown_target_class")
        return {"x": x, "y": y}
    except (KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError):
        raise Nc1Error("invalid_private_card") from None
