"""Bounded offline NC1 apply oracle. Rust alone fits the product contract.

Real rows/cards remain private. This helper never fits, reads files, dispatches
models, uses network, or claims quality/privacy. Errors contain safe codes only.
"""
import math
import re


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


def _validate_card(card):
    """Check structural validity; Rust alone verifies canonical private-card digests.

    Digest strings are checked for shape, not authenticated/recomputed here.
    """
    def require(value):
        if not value:
            raise Nc1Error("invalid_private_card")

    def integer(value):
        return type(value) is int and value >= 0

    def labels(values, minimum=0, maximum=256):
        return (isinstance(values, list) and minimum <= len(values) <= maximum
                and all(type(v) is str and len(v) <= 65536 for v in values)
                and len(set(values)) == len(values))

    def numeric(state):
        require(isinstance(state, dict) and set(state) == {"lo", "hi", "median", "all_missing"})
        for field in ("lo", "hi", "median"):
            require(type(state[field]) in (int, float) and math.isfinite(state[field]))
        require(state["lo"] <= state["hi"] and 0 <= state["median"] <= 1
                and type(state["all_missing"]) is bool)

    try:
        require(set(card) == {"schema", "recipe", "source_sha256", "fit_scope_sha256",
                              "transform_sha256", "options", "declared", "names",
                              "features", "target", "fit_class_counts", "public"})
        require(card["schema"] == "dope.nc1-private/v1" and card["recipe"] == "nc1-minmax-onehot-v1")
        for field in ("source_sha256", "fit_scope_sha256", "transform_sha256"):
            require(type(card[field]) is str and re.fullmatch(r"[0-9a-f]{64}", card[field]) is not None)
        options = card["options"]
        require(isinstance(options, dict) and set(options) == {"min_level_count", "max_levels"}
                and type(options["min_level_count"]) is int and options["min_level_count"] == 1
                and type(options["max_levels"]) is int and 1 <= options["max_levels"] <= 255)
        names, features, declared, public = card["names"], card["features"], card["declared"], card["public"]
        require(labels(names, 1, 4096) and all(names))
        require(isinstance(features, list) and 1 <= len(features) <= 256)
        require(isinstance(declared, dict) and set(declared) == {"features", "target"}
                and isinstance(declared["features"], list) and len(declared["features"]) == len(features))
        require(isinstance(public, dict) and set(public) == {"schema", "recipe", "columns", "groups", "target", "classes"}
                and public["schema"] == "dope.nc1-public/v1" and public["recipe"] == card["recipe"]
                and type(public["columns"]) is int and 1 <= public["columns"] <= 256
                and isinstance(public["groups"], list) and len(public["groups"]) == len(features))
        target_role = declared["target"]
        require(isinstance(target_role, dict) and set(target_role) == {"column", "task"}
                and integer(target_role["column"]) and target_role["column"] < len(names))
        used, cursor = {target_role["column"]}, 0
        for index, (feature, role, group) in enumerate(zip(features, declared["features"], public["groups"])):
            require(isinstance(feature, dict) and set(feature) == {"column", "name", "has_missing", "state"})
            require(isinstance(role, dict) and set(role) == {"column", "kind"})
            column = feature["column"]
            require(integer(column) and column < len(names) and column not in used
                    and column == role["column"] and feature["name"] == names[column]
                    and type(feature["has_missing"]) is bool)
            used.add(column)
            kind, state = role["kind"], feature["state"]
            require(isinstance(kind, dict) and isinstance(state, dict))
            source_kind, state_kind = kind.get("kind"), state.get("state")
            expected = {"continuous": ("numeric", "numeric"), "integer": ("numeric", "integer"),
                        "boolean": ("boolean", "boolean"), "ordinal": ("ordinal", "ordinal"),
                        "nominal": ("nominal", "nominal")}.get(source_kind)
            require(expected is not None and state_kind == expected[0])
            require(set(kind) == ({"kind", "levels"} if source_kind == "ordinal" else {"kind"}))
            require(isinstance(group, dict) and set(group) == {"id", "kind", "start", "width", "has_missing"}
                    and type(group["id"]) is int and group["id"] == index
                    and type(group["start"]) is int and group["start"] == cursor
                    and type(group["has_missing"]) is bool and group["has_missing"] == feature["has_missing"]
                    and group["kind"] == expected[1])
            width = 1 + int(feature["has_missing"])
            if state_kind == "numeric":
                require(set(state) == {"state", "numeric"})
                numeric(state["numeric"])
            elif state_kind == "nominal":
                require(set(state) == {"state", "levels"} and labels(state["levels"], 0, options["max_levels"]))
                width = len(state["levels"]) + 1 + int(feature["has_missing"])
            elif state_kind == "ordinal":
                require(set(state) == {"state", "levels", "median"} and labels(state["levels"], 2, 4096)
                        and state["levels"] == kind["levels"] and type(state["median"]) in (int, float)
                        and math.isfinite(state["median"]) and 0 <= state["median"] <= 1)
            else:
                require(set(state) == {"state"})
            require(type(group["width"]) is int and group["width"] == width)
            cursor += width
        require(cursor == public["columns"])
        target, task, counts = card["target"], target_role["task"], card["fit_class_counts"]
        require(isinstance(target, dict) and isinstance(task, dict) and isinstance(counts, list)
                and all(integer(v) for v in counts) and integer(public["classes"]))
        if task.get("task") == "regression":
            require(set(task) == {"task"} and set(target) == {"state", "numeric"}
                    and target["state"] == "regression" and public["target"] == "regression"
                    and public["classes"] == 0 and not counts)
            numeric(target["numeric"])
            require(target["numeric"]["lo"] < target["numeric"]["hi"])
        else:
            require(set(target) == {"state", "labels"} and target["state"] == "classification"
                    and labels(target["labels"], 2, 256) and len(target["labels"]) == public["classes"]
                    and len(counts) == public["classes"])
            if task.get("task") == "binary":
                require(set(task) == {"task", "positive_label"} and public["target"] == "binary"
                        and public["classes"] == 2 and task["positive_label"] == target["labels"][1])
            else:
                require(set(task) == {"task", "labels"} and task.get("task") == "multiclass"
                        and public["target"] == "multiclass" and task["labels"] == target["labels"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise Nc1Error("invalid_private_card") from None


def apply_private_card(card, columns, *, with_target=True):
    """Apply Rust-exported owner state to named, raw Python values.

    None is explicitly missing. Category strings are never interpreted as
    missing tokens; booleans are distinct from numbers. No card is refitted.
    """
    if not isinstance(card, dict):
        raise Nc1Error("invalid_private_card")
    _validate_card(card)
    if not isinstance(columns, dict) or not columns:
        raise Nc1Error("invalid_input")
    features = card.get("features", [])
    schema = card.get("public", {})
    if not features or len(features) > 256 or schema.get("columns", 0) > 256:
        raise Nc1Error("input_shape_limit")
    try:
        required = [columns[feature["name"]] for feature in features]
        count = len(required[0])
        if (count == 0 or count > 1000000 or count * schema["columns"] > 16000000
                or any(not isinstance(v, (list, tuple)) or len(v) != count for v in required)):
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
