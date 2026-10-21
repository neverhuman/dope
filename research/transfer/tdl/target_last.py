"""Mandatory canonical table contract: declared features, then the target.

Only intake normalizes. Consumers validate and reject drift. CSV fields remain
opaque strings here, so normalization never turns query labels into model input.
"""
from dataclasses import dataclass
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat

MAX_BYTES = 16 << 20

class TargetLastError(ValueError):
    """Static reasons only; never print source values or headers."""

def _need(ok, code):
    if not ok:
        raise TargetLastError(code)

def _pairs(items):
    result = {}
    for key, value in items:
        _need(key not in result, "target_last_schema_invalid")
        result[key] = value
    return result

def schema_document(raw):
    try:
        _need(type(raw) is bytes and len(raw) <= MAX_BYTES, "target_last_schema_invalid")
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(TargetLastError("target_last_schema_invalid")))
        _need(type(value) is dict and value.get("format") == "dope-table-schema"
              and type(value.get("version")) is int and value["version"] == 1,
              "target_last_schema_invalid")
        return value
    except TargetLastError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise TargetLastError("target_last_schema_invalid") from None

def schema_columns(value, *, expected_target=None, canonical=True):
    _need(type(value) is dict, "target_last_schema_invalid")
    columns = value.get("columns")
    _need(type(columns) is list and len(columns) >= 2, "target_last_schema_invalid")
    _need(all(type(c) is dict and type(c.get("source_name")) is str
              and bool(c["source_name"].strip()) and c.get("role") in ("feature", "target") for c in columns),
          "target_last_schema_invalid")
    names = [c["source_name"] for c in columns]
    targets = [c["source_name"] for c in columns if c["role"] == "target"]
    _need(len(set(names)) == len(names) and len(targets) == 1
          and any(c["role"] == "feature" for c in columns), "target_last_schema_invalid")
    target = targets[0]
    _need(expected_target is None or (type(expected_target) is str and expected_target == target),
          "target_declaration_mismatch")
    if canonical:
        _need(columns[-1]["role"] == "target" and names[-1] == target, "target_must_be_last")
    return names, target

def _table(raw):
    try:
        _need(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES, "target_last_csv_invalid")
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True))
        _need(len(rows) >= 2 and len(rows[0]) >= 2 and len(set(rows[0])) == len(rows[0]),
              "target_last_csv_invalid")
        _need(all(len(row) == len(rows[0]) for row in rows[1:]), "target_last_csv_invalid")
        return rows
    except TargetLastError:
        raise
    except (ValueError, UnicodeError, csv.Error, OverflowError):
        raise TargetLastError("target_last_csv_invalid") from None

def require_target_last(csv_raw, schema_raw, *, expected_target=None):
    value = schema_document(schema_raw)
    names, target = schema_columns(value, expected_target=expected_target)
    rows = _table(csv_raw)
    _need(rows[0] == names and rows[0][-1] == target, "target_last_header_mismatch")
    return names, target

@dataclass(frozen=True, repr=False)
class NormalizedCsv:
    csv: bytes
    schema: bytes
    receipt: dict

def normalize_csv(csv_raw, schema_raw, *, target):
    """The sole intake conversion; target must come from the dataset declaration.

    Values and row order are unchanged. Feature order follows the explicit
    schema, never an inference from labels, values or column spelling.
    """
    _need(type(target) is str and bool(target), "target_declaration_required")
    value = schema_document(schema_raw)
    names, declared = schema_columns(value, expected_target=target, canonical=False)
    rows = _table(csv_raw)
    _need(len(rows[0]) == len(names) and set(rows[0]) == set(names), "target_last_header_mismatch")
    columns = [c for c in value["columns"] if c["role"] != "target"] + [
        c for c in value["columns"] if c["role"] == "target"]
    order = [c["source_name"] for c in columns]
    permutation = [rows[0].index(name) for name in order]
    if rows[0] == order:
        normalized_csv = csv_raw
    else:
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows([[row[i] for i in permutation] for row in rows])
        normalized_csv = output.getvalue().encode("utf-8")
    if value["columns"] == columns:
        normalized_schema = schema_raw
    else:
        normalized_schema = json.dumps({**value,"columns":columns},sort_keys=True,
                                      separators=(",",":"),allow_nan=False).encode()
    require_target_last(normalized_csv, normalized_schema, expected_target=declared)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    receipt = {"format":"dope-target-last-normalization-v1", "version":1,
        "source_csv_sha256":sha(csv_raw), "source_schema_sha256":sha(schema_raw),
        "normalized_csv_sha256":sha(normalized_csv), "normalized_schema_sha256":sha(normalized_schema),
        "source_column_indices_in_output_order":permutation,"target_output_index":len(order)-1,
        "rows":len(rows)-1,"columns":len(order),"row_order_unchanged":True,
        "field_text_unchanged":True,"numeric_values_or_query_labels_parsed":False,
        "csv_serialization":"source_bytes_unchanged" if rows[0] == order else "utf8_lf_minimal_csv_quoting",
        "schema_roles":"features_then_one_target_only",
        "physical_identity_policy":"preserve source row identities; column permutation is a view, not a new physical unit"}
    return NormalizedCsv(normalized_csv, normalized_schema, receipt)

def _read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd,"rb") as handle:
        info = os.fstat(handle.fileno())
        _need(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_BYTES, "target_last_file_refused")
        raw = handle.read(MAX_BYTES+1)
        _need(len(raw) <= MAX_BYTES,"target_last_file_refused")
        return raw

def normalize_files(csv_path, schema_path, out_dir, *, target):
    result = normalize_csv(_read(csv_path), _read(schema_path), target=target)
    out = Path(out_dir)
    _need(out.is_absolute() and ".." not in out.parts and out.parent.resolve(strict=True) == out.parent,
          "target_last_output_refused")
    out.mkdir(mode=0o700)
    for name, raw in (("table.csv",result.csv),("schema.json",result.schema),
                      ("normalization.json",json.dumps(result.receipt,sort_keys=True,indent=2).encode()+b"\n")):
        fd = os.open(out/name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,"wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(out,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return result

def main():
    parser=argparse.ArgumentParser(description="Normalize one declared table to mandatory target-last layout")
    parser.add_argument("--csv",required=True);parser.add_argument("--schema",required=True)
    parser.add_argument("--target",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args()
    try:
        result=normalize_files(args.csv,args.schema,args.out,target=args.target)
    except (TargetLastError,OSError):
        raise SystemExit("target_last_intake_refused") from None
    print(json.dumps(result.receipt,sort_keys=True))

if __name__ == "__main__": main()
