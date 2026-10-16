"""Standalone generator adapter conformance (T-063); no model qualification."""
from abc import ABC, abstractmethod
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import struct
import sys

from .access import AccessTrace
from .charge import canonical_container
from .common import MetricError, count, number, require
from .literal_worker import _literal_payload, _rows
from .runtime import closure_inventory, python_runtime_paths

GENERATOR_REGISTRY = {
    "dope": {"status": "conformance_adapter", "native": True},
    "copied_rows": {"status": "conformance_control", "native": False},
    "gzip_original": {"status": "conformance_control", "native": False},
    "gaussian_copula": {"status": "specified", "native": False},
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
        # The native isolated command builds its own declared sampling closure,
        # records access evidence and enforces the existing syscall boundary.
        report = self._run(["sample-raw-isolated", "--artifact", str(self._artifact),
                            "--rows", str(rows), "--seed", str(seed)])
        self.workdir.mkdir(parents=True, exist_ok=True)
        output = self.workdir / "sample.csv"
        self._run(["sample-raw", "--artifact", str(self._artifact), "--rows", str(rows),
                   "--seed", str(seed), "--out", str(output)])
        sampled = [[number(float(cell)) for cell in row] for row in csv.reader(io.StringIO(output.read_text()))]
        encoded = b"".join(struct.pack("<d", value) for row in sampled for value in row)
        require(hashlib.sha256(encoded).hexdigest() == report["sample_sha256"], "native_replay_mismatch")
        accounting = report["inspection"]["accounting"]
        return sampled, {"method": "dope", "rows": rows, "seed": seed,
                         "sample_sha256": report["sample_sha256"],
                         "payload_bytes": self._artifact.stat().st_size,
                         "sampling_closure_bytes": accounting["cold_sampling_closure_bytes"],
                         "access_trace": {"status": "passed", "trace_kind": report["isolation"],
                                          "filesystem_network_exec": report["filesystem_network_exec"]},
                         "citable": False}


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
    parser.add_argument("--method", required=True, choices=["copied_rows", "gzip_original", "dope"])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    try:
        if args.method == "dope":
            port = DopeGenerator.reload(args.artifact, binary=args.binary, workdir=args.artifact.parent)
        else:
            port = LiteralGenerator.reload(args.artifact, method=args.method)
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
