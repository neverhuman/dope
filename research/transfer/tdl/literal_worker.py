"""Standalone literal-control worker; imports only the declared stdlib/runtime."""
import csv
import gzip
import io
import json
import math
import os
import stat
from pathlib import Path
import struct
import sys
import zlib

if __package__:
    from .common import XorShift64, count, number, require
else:
    from common import XorShift64, count, number, require


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


CONTROL_METHODS = {"gaussian_copula", "compressed_real_rows"}
MAX_CONTROL_ROWS, MAX_CONTROL_COLUMNS = 4096, 64
MAX_CONTROL_JSON = 16 << 20
CONTROL_QUANTILES = (257, 129, 65, 33, 17, 9, 5, 3)


def _control_seed(seed):
    require(type(seed) is int and 0 <= seed < (1 << 64), "invalid_control_seed")
    return seed


class _ControlRng(XorShift64):
    def __init__(self, seed):
        self.state = _control_seed(seed) or 1


def _read_control_bytes(path, limit=MAX_CONTROL_JSON):
    """Bound reads before parsing; never follow a final symlink or block on FIFO."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "invalid_control_file")
        pieces, total = [], 0
        while total <= limit:
            chunk = os.read(fd, min(1 << 20, limit + 1 - total))
            if not chunk:
                break
            pieces.append(chunk)
            total += len(chunk)
        require(total <= limit, "control_artifact_exceeds_cap")
        return b"".join(pieces)
    finally:
        os.close(fd)


def _control_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "invalid_control_artifact")
        result[key] = value
    return result


def _control_binary_document(raw):
    require(type(raw) is bytes and len(raw) >= 40, "invalid_control_artifact")
    magic, method, columns, fit_rows, seed, count, stored = struct.unpack_from("<8sIIQQII", raw)
    require(magic == b"DNCB2\0\0\0" and method in (1, 2) and 0 < columns <= 64
            and 0 < fit_rows <= 4096, "invalid_control_artifact")
    require((method == 1 and 0 < stored <= fit_rows and count == 0)
            or (method == 2 and stored == 0 and count in (257,129,65,33,17,9,5,3)),
            "invalid_control_artifact")
    values = stored * columns if method == 1 else 1 + columns * count + columns * (columns + 1) // 2
    require(len(raw) == 40 + 8 * values, "invalid_control_artifact")
    it = iter(value[0] for value in struct.iter_unpack("<d", raw[40:]))
    result = {"format":"tdl-bounded-control-v1", "method":{1:"compressed_real_rows",2:"gaussian_copula"}[method],
              "columns":columns,"fit_rows":fit_rows,"fit_seed":seed,"column_order":"features_then_target"}
    if method == 1:
        result["rows"] = [[next(it) for _ in range(columns)] for _ in range(stored)]
    else:
        result["shrinkage"] = next(it)
        result["quantile_count"] = count
        result["quantiles"] = [[next(it) for _ in range(count)] for _ in range(columns)]
        result["factor"] = [[next(it) for _ in range(i+1)] for i in range(columns)]
    return result


def _control_document(content, method):
    """Strict bounded decoder; private state is never a source-file locator."""
    require(method in CONTROL_METHODS and len(content) <= MAX_CONTROL_JSON, "invalid_control_artifact")
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress(content, MAX_CONTROL_JSON + 1)
    require(len(raw) <= MAX_CONTROL_JSON and decoder.eof and not decoder.unused_data
            and not decoder.unconsumed_tail, "invalid_control_artifact")
    if raw.startswith(b"DNCB2\0\0\0"):
        document = _control_binary_document(raw)
    else:
        document = json.loads(raw, object_pairs_hook=_control_json_pairs)
    common = {"format", "method", "fit_rows", "fit_seed", "columns", "column_order"}
    extra = {"quantile_count", "quantiles", "factor", "shrinkage"} if method == "gaussian_copula" else {"rows"}
    require(type(document) is dict and set(document) == common | extra, "invalid_control_artifact")
    require(document["format"] == "tdl-bounded-control-v1" and document["method"] == method
            and document["column_order"] == "features_then_target",
            "invalid_control_artifact")
    fit_rows = count(document["fit_rows"], 1)
    require(fit_rows <= MAX_CONTROL_ROWS, "invalid_control_artifact")
    _control_seed(document["fit_seed"])
    columns = count(document["columns"], 1)
    require(columns <= MAX_CONTROL_COLUMNS, "invalid_control_artifact")
    if method == "compressed_real_rows":
        stored = document["rows"]
        require(type(stored) is list and 0 < len(stored) <= document["fit_rows"], "invalid_control_artifact")
        for row in stored:
            require(type(row) is list and len(row) == columns, "invalid_control_artifact")
            for value in row:
                number(value)
    else:
        require(document["fit_rows"] >= 3 and document["shrinkage"] == 1e-6
                and type(document["shrinkage"]) is float, "invalid_control_artifact")
        size = count(document["quantile_count"], 3)
        quantiles, factor = document["quantiles"], document["factor"]
        require(size in CONTROL_QUANTILES and type(quantiles) is list and len(quantiles) == columns
                and type(factor) is list and len(factor) == columns, "invalid_control_artifact")
        for index, (marginal, row) in enumerate(zip(quantiles, factor)):
            require(type(marginal) is list and len(marginal) == size
                    and type(row) is list and len(row) == index + 1, "invalid_control_artifact")
            values = [number(value) for value in marginal]
            require(all(left <= right for left, right in zip(values, values[1:])), "invalid_control_artifact")
            values = [number(value) for value in row]
            require(all(abs(value) <= 1.00000001 for value in values) and values[-1] > 0
                    and abs(math.fsum(value * value for value in values) - 1) < 1e-7,
                    "invalid_control_artifact")
    return document


def _copula_state(rows, seed):
    """Midrank Gaussian dependence and bounded empirical inverse marginals."""
    from statistics import NormalDist
    require(3 <= len(rows) <= MAX_CONTROL_ROWS, "invalid_control_fit")
    columns = list(zip(*rows))
    transformed, sorted_columns = [], []
    normal = NormalDist()
    for column in columns:
        ordered = sorted(column)
        sorted_columns.append(ordered)
        ranks, position = {}, 0
        while position < len(ordered):
            end = position + 1
            while end < len(ordered) and ordered[end] == ordered[position]:
                end += 1
            ranks[ordered[position]] = normal.inv_cdf((position + end) / (2 * len(ordered)))
            position = end
        values = [ranks[value] for value in column]
        average = math.fsum(values) / len(values)
        centered = [value - average for value in values]
        norm = math.sqrt(math.fsum(value * value for value in centered))
        transformed.append([value / norm for value in centered] if norm else [0.] * len(values))
    factor = []
    for i in range(len(columns)):
        row = []
        for j in range(i + 1):
            corr = 1. if i == j else (1 - 1e-6) * math.fsum(
                left * right for left, right in zip(transformed[i], transformed[j]))
            residual = corr - math.fsum(row[k] * (row[k] if i == j else factor[j][k])
                                        for k in range(j))
            if i == j:
                require(residual > 0, "invalid_control_fit")
                row.append(math.sqrt(residual))
            else:
                row.append(residual / factor[j][j])
        factor.append(row)
    return factor, sorted_columns


def _control_sample(document, rows, seed):
    count(rows, 1)
    _control_seed(seed)
    require(rows <= 1_000_000, "invalid_count")
    rng = _ControlRng(seed)
    if document["method"] == "compressed_real_rows":
        return [list(document["rows"][rng.index(len(document["rows"]))]) for _ in range(rows)]
    output, columns = [], document["columns"]
    for _ in range(rows):
        independent = []
        while len(independent) < columns:
            u = (rng.index((1 << 53) - 1) + 1) / (1 << 53)
            angle = 2 * math.pi * (rng.index((1 << 53) - 1) + 1) / (1 << 53)
            radius = math.sqrt(-2 * math.log(u))
            independent.extend((radius * math.cos(angle), radius * math.sin(angle)))
        sample = []
        for index, factor in enumerate(document["factor"]):
            latent = math.fsum(left * right for left, right in zip(factor, independent))
            position = (0.5 + 0.5 * math.erf(latent / math.sqrt(2))) * (document["quantile_count"] - 1)
            lower = min(int(position), document["quantile_count"] - 1)
            upper = min(lower + 1, document["quantile_count"] - 1)
            weight = position - lower
            marginal = document["quantiles"][index]
            sample.append(number((1 - weight) * marginal[lower] + weight * marginal[upper]))
        output.append(sample)
    return output



def _sample_literal(artifact, method, rows, seed):
    """Worker entry: only the declared artifact and fixed runtime are readable."""
    raw_artifact = (_read_control_bytes(artifact) if method in CONTROL_METHODS
                    else Path(artifact).read_bytes())
    content = _literal_payload(raw_artifact)
    if method in CONTROL_METHODS:
        return _control_sample(_control_document(content, method), rows, seed)
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
    except (OSError, ValueError, KeyError, TypeError, EOFError, csv.Error, zlib.error, ArithmeticError, RecursionError):
        sys.exit(1)  # No private parser errors or values cross the worker boundary.
