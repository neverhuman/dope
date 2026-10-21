"""Lossless ABI2 control-state wire encoding; v1 JSON remains the default.

Only fitted numeric state is encoded. Source values, schema names and paths are
not added. All bytes stay inside the charged TDL1 literal/gzip payload.
"""
import struct
from .common import require

MAGIC = b"DNCB2\0\0\0"
HEADER = struct.Struct("<8sIIQQII")
METHODS = {"compressed_real_rows": 1, "gaussian_copula": 2}


def encode(document):
    method = document["method"]
    require(method in METHODS, "invalid_control_artifact")
    columns = document["columns"]
    count = document.get("quantile_count", 0)
    stored = len(document["rows"]) if method == "compressed_real_rows" else 0
    data = bytearray(HEADER.pack(MAGIC, METHODS[method], columns,
                                document["fit_rows"], document["fit_seed"], count, stored))
    matrices = [document["rows"]] if stored else [document["quantiles"], document["factor"]]
    if not stored:
        data.extend(struct.pack("<d", document["shrinkage"]))
    for matrix in matrices:
        for row in matrix:
            for value in row:
                data.extend(struct.pack("<d", value))
    return bytes(data)


def decode(raw):
    require(type(raw) is bytes and len(raw) >= HEADER.size, "invalid_control_artifact")
    magic, method, columns, fit_rows, seed, count, stored = HEADER.unpack_from(raw)
    require(magic == MAGIC and method in (1, 2) and 0 < columns <= 64
            and 0 < fit_rows <= 4096, "invalid_control_artifact")
    require((method == 1 and 0 < stored <= fit_rows and count == 0)
            or (method == 2 and stored == 0 and count in (257,129,65,33,17,9,5,3)),
            "invalid_control_artifact")
    values = stored * columns if method == 1 else 1 + columns * count + columns * (columns + 1) // 2
    require(len(raw) == HEADER.size + 8 * values, "invalid_control_artifact")
    it = iter(value[0] for value in struct.iter_unpack("<d", raw[HEADER.size:]))
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
