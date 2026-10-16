"""Run strict shared TDL JSON cases; Python success alone is not Rust parity."""
import argparse
import json
import math
from pathlib import Path

from . import aggregate, charge, drivers, frontier, transfer
from .common import MetricError, require

COMPONENTS = {m.__name__.rsplit(".", 1)[1]: m.evaluate for m in (transfer, aggregate, drivers, frontier, charge)}
ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests/golden"


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def strict_load(text):
    def reject(_):
        raise MetricError("nonstandard_json_number")
    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), "nonstandard_json_number")
        return result
    def bounded_int(value):
        result = int(value)
        require(-(1 << 63) <= result <= (1 << 64) - 1, "invalid_json")
        return result
    return json.loads(text, parse_constant=reject, parse_float=finite_float, parse_int=bounded_int, object_pairs_hook=_object)


def close(actual, expected, tolerance=1e-12):
    """Exact shapes, states, integers, booleans and nulls; finite float tolerance."""
    if type(expected) is dict:
        return type(actual) is dict and actual.keys() == expected.keys() and all(close(actual[k], v, tolerance) for k, v in expected.items())
    if type(expected) is list:
        return type(actual) is list and len(actual) == len(expected) and all(close(a, b, tolerance) for a, b in zip(actual, expected))
    if type(expected) is float:
        return type(actual) in (int, float) and math.isfinite(actual) and abs(actual - expected) <= tolerance
    return type(actual) is type(expected) and actual == expected


def run_case(case):
    require(type(case) is dict and type(case.get("schema_version")) is int and
            case["schema_version"] == 1, "invalid_golden_schema")
    require(case.get("citable") is False, "golden_must_be_noncitable")
    require(type(case.get("case_id")) is str and case["case_id"], "invalid_case_id")
    component = case.get("component")
    require(type(component) is str and component in COMPONENTS, "unknown_component")
    return COMPONENTS[component](case.get("input"))


def cases(root=GOLDEN):
    return sorted(p for name in COMPONENTS for p in (root / name).glob("*.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=GOLDEN)
    args = parser.parse_args()
    paths = cases(args.root)
    require(bool(paths), "empty_golden_corpus")
    ids = set()
    for path in paths:
        case = strict_load(path.read_text())
        require(case["case_id"] not in ids, "duplicate_case_id")
        ids.add(case["case_id"])
        actual = run_case(case)
        if args.check:
            require(close(actual, case["expected"]), "golden_mismatch_" + case["case_id"])
        else:
            print(json.dumps({"case_id": case["case_id"], "actual": actual}, sort_keys=True, allow_nan=False))
    if args.check:
        print(f"Python conformance: {len(paths)} golden cases passed (tolerance 1e-12)")


if __name__ == "__main__":
    main()
