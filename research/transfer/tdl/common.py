"""Strict numeric admission and source-free failure codes for conformance."""
import math


class MetricError(ValueError):
    """Only stable, source-free reason codes may cross the public boundary."""


MAX_SAFE_INTEGER = (1 << 53) - 1


def require(condition, reason="invalid_input"):
    if not condition:
        raise MetricError(reason)


def opt_default(data, key, default):
    value = data.get(key)
    return default if value is None else value


def number(value):
    require(type(value) in (int, float), "nonfinite_or_invalid_number")
    if type(value) is int:
        require(-MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER, "nonfinite_or_invalid_number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise MetricError("nonfinite_or_invalid_number") from None
    require(math.isfinite(result), "nonfinite_or_invalid_number")
    return result


def count(value, minimum=0):
    require(type(value) is int and minimum <= value <= MAX_SAFE_INTEGER, "invalid_count")
    return value


def unit(value):
    value = number(value)
    require(0 <= value <= 1, "invalid_probability")
    return value


def vector(values, nonnegative=False):
    require(type(values) is list and bool(values), "invalid_vector")
    result = [number(v) for v in values]
    require(not nonnegative or all(v >= 0 for v in result), "negative_value")
    return result


def sum_checked(values):
    # Explicit left-to-right binary64 accumulation matching Rust sum_checked.
    total = 0.0
    for value in values:
        total = number(total + number(value))
    return total


def mean(values):
    # Same left-to-right arithmetic as the Rust reference, no overflowing sum.
    require(bool(values), "empty_mean")
    total = 0.0
    for value in values:
        total = number(total + number(value) / len(values))
    return total


def failure(reason, status="failed"):
    return {"status": status, "reason": reason}

