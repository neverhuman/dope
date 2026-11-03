"""Offline DML4 fixed-draw oracle; Rust owns product decoding and sampling."""
import json
import math
import struct
import zlib

GATE_KEYS = ("linear", "asinh", "periodic", "hinge", "pair", "row_mean")


class Dml4Error(ValueError):
    """Source-free reference failure."""


def decode(data):
    if not isinstance(data, bytes) or len(data) < 13 or len(data) > 65536 or data[:5] != b"DML4\x01":
        raise Dml4Error("invalid_envelope")
    size, crc = struct.unpack("<II", data[5:13])
    if size + 13 != len(data):
        raise Dml4Error("length_mismatch")
    if zlib.crc32(data[13:]) != crc:
        raise Dml4Error("checksum_mismatch")
    try:
        packet = json.loads(data[13:])
        if set(packet) != {"contract", "level", "schema", "schema_sha256", "identity", "gates", "groups", "target"}:
            raise Dml4Error("invalid_packet_json")
        if packet["contract"] != "dope.dml4-reference/v1" or packet["level"] not in ("2k", "16k", "64k"):
            raise Dml4Error("invalid_packet_identity")
        if len(data) > {"2k": 2048, "16k": 16384, "64k": 65536}[packet["level"]]:
            raise Dml4Error("over_budget")
        if set(packet["gates"]) != set(GATE_KEYS) or any(not math.isfinite(v) or not 0 <= v <= 4 for v in packet["gates"].values()):
            raise Dml4Error("invalid_gate")
        return packet
    except (KeyError, TypeError, ValueError, UnicodeError):
        raise Dml4Error("invalid_packet_json") from None


def basis(row, gates):
    if not row or len(row) > 256 or any(not math.isfinite(v) for v in row):
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
    if not draws or len(draws) > 100000:
        raise Dml4Error("sample_shape_limit")
    width = 2 * len(packet["groups"]) + 1
    if len(draws) * packet["schema"]["columns"] > 16000000:
        raise Dml4Error("sample_shape_limit")
    x, y = [], []
    for draw in draws:
        if len(draw) != width or any(not math.isfinite(v) or not 0 <= v < 1 for v in draw):
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
