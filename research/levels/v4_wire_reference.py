"""Offline DML4 fixed-draw oracle; Rust owns product decoding and sampling."""
import hashlib
import json
import math
import struct
import zlib

GATE_KEYS = ("linear", "asinh", "periodic", "hinge", "pair", "row_mean")


class Dml4Error(ValueError):
    """Source-free reference failure."""


def _require(ok, code):
    if not ok:
        raise Dml4Error(code)


def _number(v):
    return type(v) in (int, float) and math.isfinite(v)


def _unit(v):
    return _number(v) and 0 <= v <= 1


def _pin(v):
    return type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _object(value, keys, code):
    _require(type(value) is dict and set(value) == set(keys), code)


def _probabilities(p):
    return (type(p) is list and 1 <= len(p) <= 256 and all(_number(v) and v >= 0 for v in p)
            and math.isfinite(sum(p)) and sum(p) > 0)


def _validate_gates(gates):
    _object(gates, GATE_KEYS, "invalid_gate")
    _require(all(_number(v) and 0 <= v <= 4 for v in gates.values()), "invalid_gate")


def validate(packet):
    """Independent validation of the same bounded public graph as native Rust."""
    _object(packet, ("contract", "level", "schema", "schema_sha256", "identity", "gates", "groups", "target"), "invalid_packet_json")
    _require(packet["contract"] == "dope.dml4-reference/v1" and packet["level"] in ("2k", "16k", "64k"), "invalid_packet_identity")
    schema = packet["schema"]
    _object(schema, ("schema", "recipe", "columns", "groups", "target", "classes"), "invalid_public_schema")
    p, k = schema["columns"], schema["classes"]
    _require(schema["schema"] == "dope.nc1-public/v1" and schema["recipe"] == "nc1-minmax-onehot-v1"
             and type(p) is int and 1 <= p <= 256 and type(schema["groups"]) is list
             and 1 <= len(schema["groups"]) <= 256 and type(k) is int, "invalid_public_schema")
    _require((schema["target"] == "regression" and k == 0) or (schema["target"] == "binary" and k == 2)
             or (schema["target"] == "multiclass" and 2 <= k <= 256), "invalid_public_schema")
    ordered_groups, cursor = [], 0
    for i, group in enumerate(schema["groups"]):
        _object(group, ("id", "kind", "start", "width", "has_missing"), "invalid_public_schema")
        width, has = group["width"], group["has_missing"]
        _require(type(group["id"]) is int and group["id"] == i and type(group["start"]) is int and group["start"] == cursor
                 and type(width) is int and 1 <= width <= 256 and type(has) is bool
                 and group["kind"] in ("numeric", "integer", "boolean", "ordinal", "nominal"), "invalid_public_schema")
        _require(width >= 1 + int(has) if group["kind"] == "nominal" else width == 1 + int(has), "invalid_public_schema")
        cursor += width
        ordered_groups.append({key: group[key] for key in ("id", "kind", "start", "width", "has_missing")})
    _require(cursor == p, "invalid_public_schema")
    canonical = {"schema": schema["schema"], "recipe": schema["recipe"], "columns": p,
                 "groups": ordered_groups, "target": schema["target"], "classes": k}
    digest = hashlib.sha256(json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    _require(packet["schema_sha256"] == digest, "invalid_packet_identity")
    _object(packet["identity"], ("weights_sha256", "encoder_recipe_sha256", "conditioning_sha256", "fit_scope_sha256"), "invalid_packet_identity")
    _require(all(_pin(v) for v in packet["identity"].values()), "invalid_packet_identity")
    _validate_gates(packet["gates"])
    _require(type(packet["groups"]) is list and len(packet["groups"]) == len(ordered_groups), "group_model_mismatch")
    for group, model in zip(ordered_groups, packet["groups"]):
        _require(type(model) is dict, "group_model_mismatch")
        kind = model.get("kind")
        if kind == "numeric" and group["kind"] == "numeric":
            _object(model, ("kind", "knots", "missing_probability", "imputation"), "invalid_numeric_model")
            v = model["knots"]
            _require(type(v) is list and 2 <= len(v) <= 65 and all(_unit(x) for x in v)
                     and all(a <= b for a, b in zip(v, v[1:])), "invalid_numeric_model")
        elif kind == "discrete" and group["kind"] in ("integer", "ordinal", "boolean"):
            _object(model, ("kind", "values", "probabilities", "missing_probability", "imputation"), "invalid_discrete_model")
            v = model["values"]
            _require(type(v) is list and 1 <= len(v) <= 256 and all(_unit(x) for x in v)
                     and all(a < b for a, b in zip(v, v[1:])) and _probabilities(model["probabilities"])
                     and len(v) == len(model["probabilities"]), "invalid_discrete_model")
            if group["kind"] == "boolean":
                _require(all(x in (0, 1) for x in v) and model["imputation"] in (0, 1), "invalid_discrete_model")
        elif kind == "categorical" and group["kind"] == "nominal":
            _object(model, ("kind", "probabilities"), "invalid_categorical_model")
            _require(_probabilities(model["probabilities"]) and len(model["probabilities"]) == group["width"], "invalid_categorical_model")
            continue
        else:
            raise Dml4Error("group_model_mismatch")
        _require(_unit(model["imputation"]) and _unit(model["missing_probability"])
                 and (group["has_missing"] or model["missing_probability"] == 0), "invalid_group_missing_model")
    target = packet["target"]
    _require(type(target) is dict, "invalid_target_model")
    parameter = lambda v: _number(v) and abs(v) <= 1e6
    weights = lambda v: type(v) is list and len(v) == 5*p and all(parameter(x) for x in v)
    if target.get("task") == "regression" and schema["target"] == "regression":
        _object(target, ("task", "intercept", "coefficients", "noise_scale"), "invalid_target_model")
        _require(parameter(target["intercept"]) and weights(target["coefficients"])
                 and _number(target["noise_scale"]) and 0 <= target["noise_scale"] <= 1e6, "invalid_target_model")
    elif target.get("task") == "classification" and schema["target"] in ("binary", "multiclass"):
        _object(target, ("task", "intercepts", "coefficients"), "invalid_target_model")
        _require(type(target["intercepts"]) is list and len(target["intercepts"]) == k
                 and all(parameter(x) for x in target["intercepts"]) and type(target["coefficients"]) is list
                 and len(target["coefficients"]) == k and all(weights(row) for row in target["coefficients"]), "invalid_target_model")
    else:
        raise Dml4Error("target_model_mismatch")
    return packet


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        _require(key not in out, "invalid_packet_json")
        out[key] = value
    return out


def decode(data):
    if not isinstance(data, bytes) or len(data) < 13 or len(data) > 65536 or data[:5] != b"DML4\x01":
        raise Dml4Error("invalid_envelope")
    size, crc = struct.unpack("<II", data[5:13])
    if size + 13 != len(data):
        raise Dml4Error("length_mismatch")
    if zlib.crc32(data[13:]) != crc:
        raise Dml4Error("checksum_mismatch")
    try:
        packet = json.loads(data[13:], object_pairs_hook=_unique_object,
                            parse_constant=lambda _: (_ for _ in ()).throw(Dml4Error("invalid_packet_json")))
        validate(packet)
        if len(data) > {"2k": 2048, "16k": 16384, "64k": 65536}[packet["level"]]:
            raise Dml4Error("over_budget")
        return packet
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        raise Dml4Error("invalid_packet_json") from None


def basis(row, gates):
    _validate_gates(gates)
    if type(row) not in (list, tuple) or not row or len(row) > 256 or any(not _number(v) for v in row):
        raise Dml4Error("basis_shape_limit")
    out = [gates["linear"] * v for v in row]
    out += [gates["asinh"] * math.asinh(v) for v in row]
    out += [gates["periodic"] * math.sin(math.tau * v) for v in row]
    out += [gates["hinge"] * max(v - 0.5, 0) for v in row]
    out += [gates["pair"] * a * b for a, b in zip(row, row[1:])]
    out += [gates["row_mean"] * sum(row) / len(row)]
    if any(not math.isfinite(v) for v in out):
        raise Dml4Error("nonfinite_basis_output")
    return out


def _categorical(probabilities, uniform):
    if not probabilities or any(not math.isfinite(v) or v < 0 for v in probabilities) or sum(probabilities) <= 0:
        raise Dml4Error("invalid_probabilities")
    threshold, total = uniform * sum(probabilities), 0.0
    for index, value in enumerate(probabilities):
        total += value
        if threshold < total:
            return index
    return len(probabilities) - 1


def sample_with_draws(packet, draws):
    validate(packet)
    if type(draws) is not list or not draws or len(draws) > 100000:
        raise Dml4Error("sample_shape_limit")
    width = 2 * len(packet["groups"]) + 1
    if len(draws) * packet["schema"]["columns"] > 16000000:
        raise Dml4Error("sample_shape_limit")
    x, y = [], []
    for draw in draws:
        if type(draw) is not list or len(draw) != width or any(not _number(v) or not 0 <= v < 1 for v in draw):
            raise Dml4Error("invalid_draws")
        row = []
        for index, (group, model) in enumerate(zip(packet["schema"]["groups"], packet["groups"])):
            uniform, missing_draw = draw[2 * index: 2 * index + 2]
            if model["kind"] == "categorical":
                selected = _categorical(model["probabilities"], uniform)
                row.extend(float(k == selected) for k in range(len(model["probabilities"])))
                continue
            missing = missing_draw < model["missing_probability"]
            if missing:
                value = model["imputation"]
            elif model["kind"] == "discrete":
                value = model["values"][_categorical(model["probabilities"], uniform)]
            else:
                knots = model["knots"]
                position = uniform * (len(knots) - 1)
                low = min(math.floor(position), len(knots) - 2)
                value = knots[low] + (position - low) * (knots[low + 1] - knots[low])
            row.append(value)
            if group["has_missing"]:
                row.append(float(missing))
        phi, target = basis(row, packet["gates"]), packet["target"]
        if target["task"] == "regression":
            prediction = sum(a * b for a, b in zip(phi, target["coefficients"])) + target["intercept"]
            prediction += target["noise_scale"] * (2 * draw[-1] - 1) * math.sqrt(3)
        else:
            logits = [intercept + sum(a * b for a, b in zip(phi, coefficients))
                      for intercept, coefficients in zip(target["intercepts"], target["coefficients"])]
            largest = max(logits)
            prediction = float(_categorical([math.exp(v - largest) for v in logits], draw[-1]))
        if not math.isfinite(prediction):
            raise Dml4Error("nonfinite_sample_target")
        x.append(row)
        y.append(prediction)
    return {"x": x, "y": y}


def sample(packet, rows, seed):
    """Same SplitMix64 draw schedule as Rust. No model/source/card access."""
    validate(packet)
    if type(rows) is not int or not 1 <= rows <= 100000 or type(seed) is not int or not 0 <= seed < 1 << 64:
        raise Dml4Error("sample_shape_limit")
    if rows * packet["schema"]["columns"] > 16000000:
        raise Dml4Error("sample_shape_limit")
    state, mask, draws = seed, (1 << 64) - 1, []
    for _ in range(rows):
        row = []
        for _ in range(2 * len(packet["groups"]) + 1):
            state = (state + 0x9e3779b97f4a7c15) & mask
            value = ((state ^ (state >> 30)) * 0xbf58476d1ce4e5b9) & mask
            value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
            row.append(((value ^ (value >> 31)) >> 11) / 9007199254740992)
        draws.append(row)
    return sample_with_draws(packet, draws)
