"""Standalone generator adapter conformance (T-063); no model qualification."""
from abc import ABC, abstractmethod
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import zlib
from pathlib import Path
import subprocess
import struct
import sys

from .access import AccessTrace
from .charge import canonical_container
from .common import MetricError, count, number, require
from .literal_worker import (CONTROL_METHODS, CONTROL_QUANTILES, MAX_CONTROL_COLUMNS, MAX_CONTROL_JSON,
                             MAX_CONTROL_ROWS, _ControlRng, _control_seed, _read_control_bytes,
                             _control_document, _control_json_pairs, _copula_state, _literal_payload, _rows)
from .runtime import closure_inventory, python_runtime_paths

GENERATOR_REGISTRY = {
    "dope": {"status": "conformance_adapter", "native": True},
    "copied_rows": {"status": "conformance_control", "native": False},
    "gzip_original": {"status": "conformance_control", "native": False},
    "gaussian_copula": {"status": "specified", "native": False},
    "compressed_real_rows": {"status": "implemented_unqualified", "native": False},
    "arf": {"status": "specified", "native": False},
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class StandaloneGenerator(ABC):
    """Fitting receives development data; reload/sample receive only artifacts."""
    @abstractmethod
    def fit(self, csv_path, schema_path, seed):
        """Fit a development table under a frozen schema."""

    @abstractmethod
    def export(self, artifact):
        """Export every dataset-specific byte needed by sample."""

    @classmethod
    @abstractmethod
    def reload(cls, artifact, **runtime):
        """Construct from portable artifact and declared fixed runtime only."""

    @abstractmethod
    def sample(self, rows, seed):
        """Return generated rows and a source-free access receipt."""


class LiteralGenerator(StandaloneGenerator):
    """Exact-byte copied-rows and gzip-of-original controls, numeric headerless CSV."""
    def __init__(self, method="copied_rows"):
        require(method in {"copied_rows", "gzip_original"}, "unsupported_generator")
        self.method = method
        self._payload = None
        self._artifact = None

    def fit(self, csv_path, schema_path, seed):
        count(seed)
        self._payload = None
        self._artifact = None
        try:
            text = Path(csv_path).read_text()
            schema = json.loads(Path(schema_path).read_text())
            rows = _rows(text, schema)
        except (OSError, ValueError, KeyError, TypeError):
            raise MetricError("generator_fit_failed") from None
        document = {"format": "tdl-literal-v1", "method": self.method, "schema": schema}
        if self.method == "copied_rows":
            document["rows"] = rows
        else:
            document["original_csv"] = text
        raw = _json(document)
        self._payload = raw if self.method == "copied_rows" else gzip.compress(raw, mtime=0)
        return {"status": "fitted", "method": self.method, "fit_seed": seed, "citable": False}

    def export(self, artifact):
        require(self._payload is not None, "generator_not_fitted")
        artifact = Path(artifact)
        artifact.write_bytes(canonical_container([{"name": "literal_payload", "hex": self._payload.hex()}]))
        self._artifact = artifact.resolve()
        return self._inventory()

    def _inventory(self):
        raw = self._artifact.read_bytes()
        return {"payload_bytes": len(raw), "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "method": self.method, "citable": False}

    @classmethod
    def reload(cls, artifact, **runtime):
        require(set(runtime) == {"method"}, "invalid_runtime")
        result = cls(runtime["method"])
        result._artifact = Path(artifact).resolve(strict=True)
        return result

    def sample(self, rows, seed):
        count(rows, 1)
        count(seed)
        require(self._artifact is not None, "generator_not_loaded")
        try:
            content = _literal_payload(self._artifact.read_bytes())
            closure, paths = closure_inventory(python_runtime_paths())
            manifest = {"sections": [{"name": "literal_payload", "hex": content.hex()}], "closure_files": closure}
            trace = AccessTrace.from_manifest(self._artifact, manifest, paths)
            worker = str(Path(__file__).with_name("literal_worker.py"))
            stdout, receipt = trace.run([sys.executable, "-S", "-B", worker, str(self._artifact),
                                         self.method, str(rows), str(seed)])
            sampled = json.loads(stdout)
        except (OSError, ValueError, KeyError, TypeError):
            raise MetricError("generator_sample_failed") from None
        return sampled, {"method": self.method, "rows": rows, "seed": seed,
                         **trace.accounting, "access_trace": receipt, "citable": False}


class DopeGenerator(StandaloneGenerator):
    """Native Rust raw-table compiler and isolated artifact-only sampling replay."""
    def __init__(self, binary, workdir, byte_budget=32768):
        self.binary = Path(binary).resolve(strict=True)
        self.workdir = Path(workdir).resolve()
        self.byte_budget = count(byte_budget, 1)
        self._artifact = None

    def _run(self, args):
        try:
            result = subprocess.run([str(self.binary), *args], env={"RAYON_NUM_THREADS": "1"},
                                    capture_output=True, check=False, timeout=90)
            require(result.returncode == 0, "native_generator_failed")
            return json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            raise MetricError("native_generator_failed") from None

    def fit(self, csv_path, schema_path, seed):
        self._artifact = None
        count(seed)
        self.workdir.mkdir(parents=True, exist_ok=True)
        artifact = self.workdir / "fitted.raw"
        self._run(["compile-raw", "--csv", str(Path(csv_path).resolve()), "--schema",
                           str(Path(schema_path).resolve()), "--out", str(artifact),
                           "--seed", str(seed), "--candidate", "chow_liu", "--max-complete-bytes",
                           str(self.byte_budget), "--deadline-seconds", "60"])
        self._artifact = artifact
        return {"status": "fitted", "method": "dope", "fit_seed": seed,
                "payload_bytes": self._artifact.stat().st_size, "citable": False}

    def export(self, artifact):
        require(self._artifact is not None, "generator_not_fitted")
        artifact = Path(artifact).resolve()
        payload = self._artifact.read_bytes()
        if artifact != self._artifact:
            artifact.write_bytes(payload)
        self._artifact = artifact
        return {"method": "dope", "payload_bytes": len(payload),
                "artifact_sha256": hashlib.sha256(payload).hexdigest(), "citable": False}

    @classmethod
    def reload(cls, artifact, **runtime):
        result = cls(runtime["binary"], runtime["workdir"])
        result._artifact = Path(artifact).resolve(strict=True)
        return result

    def sample(self, rows, seed):
        count(rows, 1)
        count(seed)
        require(self._artifact is not None, "generator_not_loaded")
        try:
            result = subprocess.run([str(self.binary), "sample-raw-isolated", "--artifact", str(self._artifact),
                                     "--rows", str(rows), "--seed", str(seed), "--private-samples-f64le"],
                                    env={"RAYON_NUM_THREADS": "1"}, capture_output=True, check=False, timeout=90)
            require(result.returncode == 0, "native_generator_failed")
            report = json.loads(result.stderr)
            stream, raw = report["private_sample_stream"], result.stdout
            require(type(stream) is dict and set(stream) == {
                "format", "layout", "column_order", "rows", "columns", "bytes"}, "native_replay_mismatch")
            columns = count(stream["columns"], 1)
            require(stream["format"] == "f64le" and stream["layout"] == "row_major"
                    and stream["column_order"] == "features_then_target" and count(stream["rows"], 1) == rows
                    and columns == len(report["inspection"]["schema"])
                    and count(stream["bytes"], 1) == len(raw) == rows * columns * 8,
                    "native_replay_mismatch")
            require(report["state"] == "sampled" and count(report["rows"], 1) == rows and count(report["seed"]) == seed
                    and report["filesystem_network_exec"] == "denied" and report["inherited_source_fds"] == "closed"
                    and report["isolation"] == "linux_x86_64_seccomp_tsync_default_deny"
                    and hashlib.sha256(raw).hexdigest() == report["sample_sha256"], "native_replay_mismatch")
            values = [number(value[0]) for value in struct.iter_unpack("<d", raw)]
            sampled = [values[i:i + columns] for i in range(0, len(values), columns)]
            accounting = report["inspection"]["accounting"]
        except MetricError:
            raise
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError, struct.error):
            raise MetricError("native_generator_failed") from None
        return sampled, {"method": "dope", "rows": rows, "seed": seed,
                         "sample_sha256": report["sample_sha256"], "private_sample_stream": stream,
                         "payload_bytes": self._artifact.stat().st_size,
                         "sampling_closure_bytes": accounting["cold_sampling_closure_bytes"],
                         "access_trace": {"status": "passed", "trace_kind": report["isolation"],
                                          "filesystem_network_exec": report["filesystem_network_exec"]},
                         "citable": False}


def _pack_control(document, *, artifact_abi=1):
    require(type(artifact_abi) is int and artifact_abi in (1, 2), "invalid_runtime")
    raw = _json(document)
    if artifact_abi == 2:
        from .control_codec import encode
        # Retain the v1 strict semantic validator before converting numeric state.
        _control_document(gzip.compress(raw, mtime=0), document["method"])
        raw = encode(document)
    content = bytearray(gzip.compress(raw, compresslevel=9, mtime=0))
    content[9] = 255  # Freeze the gzip OS marker across supported Python versions.
    content = bytes(content)
    return content, canonical_container([{"name": "literal_payload", "hex": content.hex()}])


def _read_control(artifact, byte_budget, method):
    raw = _read_control_bytes(artifact, byte_budget)
    return raw, _control_document(_literal_payload(raw), method)


class _BoundedControlGenerator(StandaloneGenerator):
    """Numeric research controls; implementation success is not qualification."""
    def __init__(self, method, byte_budget=32768, *, artifact_abi=1):
        require(type(artifact_abi) is int and artifact_abi in (1, 2), "invalid_runtime")
        self.artifact_abi = artifact_abi
        require(method in CONTROL_METHODS, "unsupported_generator")
        self._method, self.byte_budget = method, count(byte_budget, 1)
        require(self.byte_budget <= MAX_CONTROL_JSON, "invalid_count")
        self._payload = self._artifact = self._artifact_sha256 = None

    @property
    def method(self):
        return self._method

    def fit(self, csv_path, schema_path, seed):
        self._payload = self._artifact = self._artifact_sha256 = None
        _control_seed(seed)
        try:
            raw = _read_control_bytes(csv_path, MAX_CONTROL_JSON)
            schema = json.loads(_read_control_bytes(schema_path, MAX_CONTROL_JSON),
                                object_pairs_hook=_control_json_pairs)
            require(type(schema) is dict and schema.get("format") == "dope-table-schema"
                    and type(schema.get("version")) is int and schema["version"] == 1,
                    "invalid_control_schema")
            columns = schema["columns"]
            require(type(columns) is list and 0 < len(columns) <= MAX_CONTROL_COLUMNS,
                    "invalid_control_fit")
            headers = [column["source_name"] for column in columns]
            require(all(type(name) is str and name for name in headers)
                    and len(set(headers)) == len(headers), "invalid_control_fit")
            require(all(type(column) is dict and column.get("dtype") == "continuous"
                        and column.get("role") in {"feature", "target"} for column in columns)
                    and sum(column["role"] == "target" for column in columns) == 1,
                    "unsupported_control_schema")
            ids = [column["id"] for column in columns]
            require(all(type(value) is str and value for value in ids) and len(set(ids)) == len(ids),
                    "invalid_control_schema")
            from .target_last import TargetLastError, schema_columns
            try:
                canonical_names, canonical_target = schema_columns(schema)
            except TargetLastError:
                raise MetricError("noncanonical_control_schema") from None
            table = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
            require(table and table[0] == canonical_names and table[0][-1] == canonical_target,
                    "invalid_control_header")
            positions = [table[0].index(name) for name in headers]
            require(all(len(row) == len(headers) for row in table[1:]), "invalid_control_fit")
            rows = [[number(float(row[i])) for i in positions] for row in table[1:]]
            require(rows and len(rows) <= MAX_CONTROL_ROWS, "invalid_control_fit")
            # Match the native raw sampler's stable feature order, then target.
            order = sorted(range(len(columns)), key=lambda i: columns[i]["role"] == "target")
            rows = [[row[i] for i in order] for row in rows]
            base = {"format": "tdl-bounded-control-v1", "method": self.method,
                    "fit_rows": len(rows), "fit_seed": seed, "columns": len(columns),
                    "column_order": "features_then_target"}
            selected = None
            if self.method == "gaussian_copula":
                factor, ordered = _copula_state(rows, seed)
                for size in CONTROL_QUANTILES:
                    quantiles = []
                    for column in ordered:
                        marginal = []
                        for index in range(size):
                            position = index * (len(column) - 1) / (size - 1)
                            lower, upper = int(position), min(int(position) + 1, len(column) - 1)
                            weight = position - lower
                            left, right = column[lower], column[upper]
                            # Equal empirical quantiles are exact plateaus. The
                            # weighted expression can round a negative plateau
                            # in opposite directions and make it nonmonotone.
                            value = left if left == right else (1 - weight) * left + weight * right
                            marginal.append(number(min(right, max(left, value))))
                        quantiles.append(marginal)
                    document = {**base, "factor": factor, "quantiles": quantiles,
                                "quantile_count": size, "shrinkage": 1e-6}
                    content, artifact = _pack_control(document, artifact_abi=self.artifact_abi)
                    if len(artifact) <= self.byte_budget:
                        selected = content
                        break
            else:
                order, rng = list(range(len(rows))), _ControlRng(seed)
                for index in range(len(order) - 1, 0, -1):
                    chosen = rng.index(index + 1)
                    order[index], order[chosen] = order[chosen], order[index]
                # Gzip size is not monotone. This frozen halving policy makes
                # no maximal-retention claim; every accepted full frame is measured.
                retained = len(rows)
                while retained:
                    content, artifact = _pack_control({**base, "rows": [rows[i] for i in order[:retained]]}, artifact_abi=self.artifact_abi)
                    if len(artifact) <= self.byte_budget:
                        selected = content
                        break
                    retained //= 2
            require(selected is not None, "control_artifact_exceeds_cap")
            _control_document(selected, self.method)
            self._payload = selected
            return {"status": "fitted", "method": self.method, "fit_seed": seed,
                    "fit_rows": len(rows), "columns": len(columns), "citable": False}
        except MetricError:
            raise
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError, RecursionError, csv.Error, zlib.error):
            raise MetricError("generator_fit_failed") from None

    def export(self, artifact):
        require(self._payload is not None, "generator_not_fitted")
        self._artifact = self._artifact_sha256 = None
        raw = canonical_container([{"name": "literal_payload", "hex": self._payload.hex()}])
        require(len(raw) <= self.byte_budget, "control_artifact_exceeds_cap")
        path = Path(artifact).absolute()
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                position = 0
                while position < len(raw):
                    written = os.write(fd, raw[position:])
                    require(written > 0, "generator_export_uncertain")
                    position += written
                os.fsync(fd)
            finally:
                os.close(fd)
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except FileExistsError:
            raise MetricError("generator_artifact_exists") from None
        except OSError:
            raise MetricError("generator_export_uncertain") from None
        self._artifact, self._artifact_sha256 = path.resolve(strict=True), hashlib.sha256(raw).hexdigest()
        return {"method": self.method, "payload_bytes": len(raw),
                "artifact_sha256": self._artifact_sha256, "budget_scope": "dataset_specific_artifact",
                "citable": False}

    @classmethod
    def reload(cls, artifact, **runtime):
        require(set(runtime) == {"byte_budget"}, "invalid_runtime")
        result = cls(byte_budget=runtime["byte_budget"])
        try:
            raw, _ = _read_control(artifact, result.byte_budget, result.method)
            result._artifact, result._artifact_sha256 = Path(artifact).resolve(strict=True), hashlib.sha256(raw).hexdigest()
        except MetricError:
            raise
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError, RecursionError, csv.Error, zlib.error):
            raise MetricError("generator_reload_failed") from None
        return result

    def sample(self, rows, seed):
        count(rows, 1)
        _control_seed(seed)
        require(rows <= 1_000_000, "invalid_count")
        require(self._artifact is not None, "generator_not_loaded")
        try:
            raw, document = _read_control(self._artifact, self.byte_budget, self.method)
            require(hashlib.sha256(raw).hexdigest() == self._artifact_sha256, "generator_artifact_changed")
            content = _literal_payload(raw)
            closure, paths = closure_inventory(python_runtime_paths())
            manifest = {"sections": [{"name": "literal_payload", "hex": content.hex()}], "closure_files": closure}
            trace = AccessTrace.from_manifest(self._artifact, manifest, paths)
            worker = str(Path(__file__).with_name("literal_worker.py"))
            stdout, receipt = trace.run([sys.executable, "-S", "-B", worker, str(self._artifact),
                                        self.method, str(rows), str(seed)])
            sampled = json.loads(stdout)
            require(type(sampled) is list and len(sampled) == rows
                    and all(type(row) is list and len(row) == document["columns"] for row in sampled),
                    "generator_sample_failed")
            sampled = [[number(value) for value in row] for row in sampled]
            encoded = b"".join(struct.pack("<d", value) for row in sampled for value in row)
        except MetricError:
            raise
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError, RecursionError, csv.Error, zlib.error):
            raise MetricError("generator_sample_failed") from None
        return sampled, {"method": self.method, "rows": rows, "seed": seed,
                         "artifact_sha256": self._artifact_sha256, "sample_sha256": hashlib.sha256(encoded).hexdigest(),
                         "budget_scope": "dataset_specific_artifact", "column_order": document["column_order"],
                         **trace.accounting,
                         "access_trace": receipt, "citable": False}


class GaussianCopulaGenerator(_BoundedControlGenerator):
    def __init__(self, byte_budget=32768, *, artifact_abi=1):
        super().__init__("gaussian_copula", byte_budget, artifact_abi=artifact_abi)


class CompressedRowsGenerator(_BoundedControlGenerator):
    def __init__(self, byte_budget=32768, *, artifact_abi=1):
        super().__init__("compressed_real_rows", byte_budget, artifact_abi=artifact_abi)


class SpecifiedGenerator(StandaloneGenerator):
    def __init__(self, method):
        require(method in {"gaussian_copula", "arf"}, "unsupported_generator")
        self.method = method

    def fit(self, *args, **kwargs):
        raise MetricError("generator_specified_unrun")

    def export(self, artifact):
        raise MetricError("generator_specified_unrun")

    @classmethod
    def reload(cls, artifact, **runtime):
        raise MetricError("generator_specified_unrun")

    def sample(self, rows, seed):
        raise MetricError("generator_specified_unrun")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["copied_rows", "gzip_original", "dope", "gaussian_copula", "compressed_real_rows"])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--byte-budget", type=int, help="Dataset-specific bounded-control cap, including artifact framing")
    args = parser.parse_args()
    try:
        if args.method in CONTROL_METHODS:
            cls = GaussianCopulaGenerator if args.method == "gaussian_copula" else CompressedRowsGenerator
            port = cls.reload(args.artifact, byte_budget=args.byte_budget)
        elif args.method == "dope":
            port = DopeGenerator.reload(args.artifact, binary=args.binary, workdir=args.artifact.parent)
        else:
            port = LiteralGenerator.reload(args.artifact, method=args.method)
        require(args.byte_budget is None or args.method in CONTROL_METHODS, "invalid_runtime")
        sampled, receipt = port.sample(args.rows, args.seed)
        if args.output is not None and sampled is not None:
            with args.output.open("w") as stream:
                csv.writer(stream).writerows(sampled)
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    except (MetricError, OSError, TypeError, ValueError):
        print(json.dumps({"status": "failed", "reason": "generator_replay_failed", "citable": False}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
