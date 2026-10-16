"""Standalone literal-control worker; imports only the declared stdlib/runtime."""
import csv
import gzip
import io
import json
from pathlib import Path
import struct
import sys
import zlib

if __package__:
    from .common import XorShift64, number, require
else:
    from common import XorShift64, number, require


def _rows(text, schema):
    table = list(csv.reader(io.StringIO(text)))
    headers = [column["source_name"] for column in schema["columns"]]
    if table and table[0] == headers:
        table = table[1:]
    rows = [[number(float(cell)) for cell in row] for row in table]
    require(rows and rows[0] and all(len(row) == len(rows[0]) for row in rows), "invalid_table")
    return rows


def _literal_payload(artifact):
    """Admit the single canonical literal section, including all framing bytes."""
    name = b"literal_payload"
    header = b"TDL1" + struct.pack("<II", 1, len(name)) + name
    require(artifact.startswith(header) and len(artifact) >= len(header) + 8, "invalid_artifact")
    size = struct.unpack("<Q", artifact[len(header):len(header) + 8])[0]
    content = artifact[len(header) + 8:]
    require(len(content) == size, "invalid_artifact")
    return content


def _sample_literal(artifact, method, rows, seed):
    """Worker entry: only the declared artifact and fixed runtime are readable."""
    content = _literal_payload(Path(artifact).read_bytes())
    raw = content if method == "copied_rows" else gzip.decompress(content)
    document = json.loads(raw)
    require(document["format"] == "tdl-literal-v1" and document["method"] == method, "invalid_artifact")
    source = document["rows"] if method == "copied_rows" else _rows(document["original_csv"], document["schema"])
    require(type(source) is list and bool(source), "invalid_table")
    rng = XorShift64(seed)
    return [list(source[rng.index(len(source))]) for _ in range(rows)]


if __name__ == "__main__":
    try:
        rows = _sample_literal(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        print(json.dumps(rows, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, EOFError, csv.Error, zlib.error):
        sys.exit(1)  # No private parser errors or values cross the worker boundary.
