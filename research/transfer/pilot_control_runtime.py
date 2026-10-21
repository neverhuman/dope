"""Actual typed control parent stages under the existing live pilot journal.

An executor must authenticate every operation against retained admission bytes.
This class supplies no grant, scheduler, alternate ledger or unconfined sampler.
Compiler and learner seeds remain distinct signed streams. Parent CPU/RAM/disk
containment is the admitting executor's responsibility; RSS below is not a child
or GPU measurement. All public failures are static and all retained data private.
"""
from dataclasses import asdict
from pathlib import Path
import resource
import sys
import time

from .campaign_runner import StageJournal
from .campaign_admission import seed_uint64
from .campaign_artifacts import create_private, read_regular, sync_artifact
from .pilot_control_materialized_draws import (
    METHODS, TrustedCompileBinding, TypedControlDraw, _compile_schema, require, sha,
)
from .pilot_control_draw_io import load_pair, store_pair
from .pilot_materialized_draws import DrawSpec, _native_json
from .tdl import control_boundary as boundary
from .tdl.generators import GaussianCopulaGenerator, CompressedRowsGenerator
from .tdl.literal_worker import _control_document, _literal_payload

COMPILER_FILES = ("target_last.py", "control_codec.py", "generators.py", "literal_worker.py", "charge.py", "common.py", "access.py", "runtime.py")
GENERATORS = {"gaussian_copula": GaussianCopulaGenerator, "compressed_real_rows": CompressedRowsGenerator}
MODULES = ("pilot_control_runtime.py", "pilot_control_draw_io.py", "pilot_control_materialized_draws.py",
           *("tdl/" + n for n in (*COMPILER_FILES, "control_boundary.py", "control_isolated_worker.py", "control_native.py")))


class PilotControlRuntime:
    def __init__(self, *, trusted_runtime, compiler_source_pins, admission_manifest, verify_operation):
        require(callable(verify_operation) and type(admission_manifest) is bytes and bool(admission_manifest))
        self.runtime = _native_json(boundary.canonical(trusted_runtime))
        self.compiler_pins = dict(compiler_source_pins)
        self.admission_manifest, self.verify_operation = admission_manifest, verify_operation
        self._bound, self._served, self._binding = False, [], None
        self._compiled, self._reloaded, self._manifest = False, False, None
        self.processes = 0
        self._capsule = boundary.canonical(self.capsule())

    def capsule(self):
        rt, source_dir = self.runtime, Path(boundary.__file__).resolve().parent
        if rt.get("scope") == "standalone-native-control-v2":
            from .tdl.control_native import verify_runtime
            verify_runtime(rt)
            require(set(self.compiler_pins) == set(COMPILER_FILES))
            for name, expected in self.compiler_pins.items():
                require(sha(boundary.regular_bytes(source_dir / name)) == expected)
            return {"format": "dope-parent-typed-control-runtime-v2", "runtime": rt,
                    "compiler_source_pins": self.compiler_pins,
                    "admission_manifest_sha256": sha(self.admission_manifest)}
        require(set(rt) == {"scope", "abi", "library", "library_sha256", "python",
                           "source_dir", "source_pins", "closure_pins"}
                and rt["scope"] == boundary.SCOPE and type(rt["abi"]) is int and rt["abi"] == boundary.ABI
                and rt["source_dir"] == str(source_dir)
                and set(rt["source_pins"]) == set(boundary.SOURCES)
                and set(self.compiler_pins) == set(COMPILER_FILES))
        # Match actual imported repository sources, not a foreign same-named compiler.
        for name, expected in self.compiler_pins.items():
            require(sha(boundary.regular_bytes(source_dir / name)) == expected)
        for name, expected in rt["source_pins"].items():
            require(sha(boundary.regular_bytes(source_dir / name)) == expected
                    and rt["closure_pins"].get(str(source_dir / name)) == expected)
        require(rt["closure_pins"].get(rt["library"]) == rt["library_sha256"]
                and rt["python"] in rt["closure_pins"])
        return {"format": "dope-parent-typed-control-runtime-v1", "runtime": rt,
                "compiler_source_pins": self.compiler_pins,
                "admission_manifest_sha256": sha(self.admission_manifest)}

    def _files(self):
        files = tuple((p, boundary.regular_bytes(p)) for p in sorted(self.runtime["closure_pins"]))
        require(all(sha(raw) == self.runtime["closure_pins"][p] for p, raw in files))
        return files

    def _authorize(self, operation, claims):
        require(callable(self.verify_operation))
        # None means the executor actually accepted; no bool/hash-only fallback.
        require(self.verify_operation(operation, claims, self.admission_manifest) is None)

    def bind(self, spec, cell_plan, auth, paths):
        doc = cell_plan.document()
        require(not self._bound and boundary.canonical(self.capsule()) == self._capsule
                and doc["version"] == 2 and doc["comparator"] in METHODS)
        self._cell_sha, self._input_sha, self._doc = cell_plan.sha256, cell_plan.input_sha256, doc
        self._spec_sha = spec.sha256
        self._paths = dict(paths)
        self._schema, self._input = read_regular(paths["schema"]), read_regular(paths["csv"])
        require(sha(self._schema) == doc["schema_sha256"] and sha(self._input) == doc["csv_sha256"])
        self._source = boundary.canonical(self.compiler_pins)
        self._files()  # complete runtime captured and pinned before compiler work
        self._authorize("prepare", {"cell_plan_sha256": cell_plan.sha256,
            "run_spec_sha256": spec.sha256, "authorization": auth, "capsule": self.capsule(),
            "input_sha256": cell_plan.input_sha256, "fit_csv_sha256": sha(self._input),
            "schema_sha256": sha(self._schema), "fit_seed_hex": doc["fit_seed"],
            "stage_wall_ms": doc["stage_wall_ms"], "byte_budget": doc["byte_budget"]})
        self._bound = True

    def _unchanged(self):
        require(self._bound and boundary.canonical(self.capsule()) == self._capsule)
        require(read_regular(self._paths["schema"]) == self._schema
                and read_regular(self._paths["csv"]) == self._input)
        for key in ("selection_csv", "query_csv"):
            require(sha(read_regular(self._paths[key])) == self._doc[key + "_sha256"])
        if self._binding is not None:
            require(sha(read_regular(self._paths["artifact"], boundary.MAX_ARTIFACT)) ==
                    self._binding.artifact_sha256)

    def _compile(self, root):
        doc, paths = self._doc, self._paths
        require(self._binding is None)
        columns = _native_json(self._schema)["columns"]
        ordered = tuple(c["id"] for c in sorted(columns, key=lambda c: c["role"] == "target"))
        # Reserve the whole paid runtime and the larger n/4n invocation before
        # fitting. Placeholder hashes have exactly the final hash width; using
        # the total cap as artifact length bounds its decimal JSON width.
        artifact_cap = min(doc["byte_budget"], boundary.MAX_ARTIFACT)
        reservation = None
        if self.runtime.get("abi") == 2:
            from .tdl.control_native import request, invocation_bytes
            files = self._files()
            overhead = max(sum(len(raw) for _, raw in files) + invocation_bytes(request(
                self.runtime, "0" * 64, artifact_cap, doc["comparator"],
                doc["rows"] * factor, seed_uint64(doc[key]), files))
                for factor, key in ((1, "draw_seed"), (4, "draw_4n_seed")))
            reservation = {"runtime_and_max_invocation_bytes": overhead,
                           "complete_byte_budget": doc["byte_budget"]}
            if doc["byte_budget"] > overhead:
                artifact_cap = min(artifact_cap, doc["byte_budget"] - overhead)
            # Impossible caps retain a real cost negative in sampling before
            # any sampler/learner. No budget is increased.
        generator = GENERATORS[doc["comparator"]](byte_budget=artifact_cap,
            artifact_abi=2 if self.runtime.get("abi") == 2 else 1)
        fitted = generator.fit(paths["csv"], paths["schema"], seed_uint64(doc["fit_seed"]))
        exported = generator.export(paths["artifact"])
        sync_artifact(paths["artifact"])
        artifact = read_regular(paths["artifact"], boundary.MAX_ARTIFACT)
        require(exported["artifact_sha256"] == sha(artifact) and exported["payload_bytes"] == len(artifact)
                and fitted["fit_rows"] == doc["fit_rows"] and fitted["columns"] == len(ordered)
                and fitted["fit_seed"] == seed_uint64(doc["fit_seed"]))
        self._compile_receipt = boundary.canonical({"format": "dope-actual-control-compile-witness-v1",
            "cell_plan_sha256": self._cell_sha, "input_sha256": self._input_sha,
            "fit_csv_sha256": sha(self._input), "schema_sha256": sha(self._schema),
            "ordered_column_ids": list(ordered), "compiler_source_sha256": sha(self._source),
            "fit": fitted, "export": exported, "sampling_reservation": reservation,
            "artifact_ceiling": artifact_cap, "complete_sampling_charge_pending": True})
        self._binding = TrustedCompileBinding(doc["comparator"], sha(artifact), self._input_sha,
            sha(self._input), sha(self._schema), sha(self._source), sha(self._compile_receipt),
            sha(self.admission_manifest), doc["fit_seed"], doc["fit_rows"], ordered, doc["byte_budget"])
        _compile_schema(self._schema, self._input, self._binding)
        self._authorize("compiled", {"cell_plan_sha256": self._cell_sha, "binding": asdict(self._binding),
                                      "compile_receipt_sha256": sha(self._compile_receipt)})
        create_private(root / "control-compile-witness.json", self._compile_receipt, 0o400)
        create_private(root / "control-compile-binding.json", boundary.canonical(asdict(self._binding)), 0o400)
        create_private(root / "control-compiler-source.json", self._source, 0o400)
        create_private(root / "control-admission.bin", self.admission_manifest, 0o400)
        self._compiled = True
        return {"comparator": doc["comparator"], "candidate": doc["candidate"],
                "artifact_sha256": sha(artifact), "artifact_bytes": len(artifact),
                "compile_binding_sha256": self._binding.sha256,
                "accounting_scope": "full_framed_artifact_only", "sampling_charge_pending": True}

    def _reload(self):
        require(self._compiled and not self._reloaded)
        doc, path = self._doc, self._paths["artifact"]
        generator = GENERATORS[doc["comparator"]].reload(
            path, byte_budget=min(doc["byte_budget"], boundary.MAX_ARTIFACT))
        raw = read_regular(path, boundary.MAX_ARTIFACT)
        document = _control_document(_literal_payload(raw), doc["comparator"])
        require(generator.method == self._binding.method and sha(raw) == self._binding.artifact_sha256
                and document["fit_seed"] == seed_uint64(doc["fit_seed"])
                and document["fit_rows"] == doc["fit_rows"]
                and document["columns"] == len(self._binding.ordered_column_ids))
        self._reloaded = True
        return {"comparator": doc["comparator"], "artifact_sha256": sha(raw),
                "accounting": {"artifact_bytes": len(raw), "scope": "full_framed_artifact_only",
                               "sampling_charge_pending": True}}

    def _verify_draw(self, binding, claims, compile_receipt, admission):
        self._unchanged()
        require(binding == self._binding and compile_receipt == self._compile_receipt
                and admission == self.admission_manifest and self._compiled and self._reloaded
                and claims["library_sha256"] == self.runtime.get("executable_sha256", self.runtime.get("library_sha256"))
                and claims["source_sha256"] == sha(boundary.canonical(self.runtime["source_pins"])))
        self._authorize("draw_handoff", {"cell_plan_sha256": self._cell_sha, **claims})
        return True

    def _request(self, scale, factor, key, files):
        doc, rt = self._doc, self.runtime
        if rt.get("scope") == "standalone-native-control-v2":
            from .tdl.control_native import request
            return request(rt, self._binding.artifact_sha256,
                len(read_regular(self._paths["artifact"], boundary.MAX_ARTIFACT)), doc["comparator"],
                doc["rows"] * factor, seed_uint64(doc[key]), files)
        closure = [{"sha256": sha(raw), "bytes": len(raw)} for _, raw in files]
        return {"method": doc["comparator"], "rows": doc["rows"] * factor,
            "seed": seed_uint64(doc[key]), "abi": boundary.ABI,
            "artifact_sha256": self._binding.artifact_sha256,
            "artifact_bytes": len(read_regular(self._paths["artifact"], boundary.MAX_ARTIFACT)),
            "library_sha256": rt["library_sha256"], "source_sha256": sha(boundary.canonical(rt["source_pins"])),
            "source_pins": rt["source_pins"], "runtime_closure_sha256": sha(boundary.canonical(closure)),
            "runtime_bytes": sum(x["bytes"] for x in closure), "closure_pins": rt["closure_pins"]}

    def _sample(self, root, started):
        require(self._compiled and self._reloaded and self._manifest is None)
        doc, files = self._doc, self._files()
        draws, requests = [], []
        slots = (("primary_n", 1, "draw_seed"), ("sensitivity_4n", 4, "draw_4n_seed"))
        for scale, factor, key in slots:
            request = self._request(scale, factor, key, files)
            requests.append(request)
        def invocation(r):
            if r["abi"] == 2:
                from .tdl.control_native import invocation_bytes
                return invocation_bytes(r)
            return len(boundary.canonical(r))
        charges = [r["artifact_bytes"] + r["runtime_bytes"] + invocation(r) for r in requests]
        create_private(root / "control-sampling-charge.json", boundary.canonical({
            "scope": "full_artifact_runtime_invocation", "per_draw_bytes": charges,
            "byte_budget": doc["byte_budget"], "admitted": max(charges) <= doc["byte_budget"]}), 0o400)
        if max(charges) > doc["byte_budget"]:
            return None  # real measured cost negative; no sampler/learner call
        for (scale, factor, key), request in zip(slots, requests):
            self._unchanged()
            remaining = doc["stage_wall_ms"]["sample"] / 1000 - (time.monotonic() - started)
            require(remaining > 0)
            self._authorize("sample_draw", {"cell_plan_sha256": self._cell_sha, "scale": scale,
                "request_sha256": sha(boundary.canonical(request)), "request": request,
                "remaining_wall_seconds": remaining, "byte_budget": doc["byte_budget"]})
            def capture(code, raw, public):
                create_private(root / (scale + ".control.stdout"), raw, 0o400)
                create_private(root / (scale + ".control.stderr"), public, 0o400)
                create_private(root / (scale + ".control.exit.json"), boundary.canonical({"returncode": code}), 0o400)
            # Admission time also consumes the same one reserved sample allowance.
            remaining = doc["stage_wall_ms"]["sample"] / 1000 - (time.monotonic() - started)
            require(remaining > 0)
            def spawned(pid):
                self.processes += 1  # actual Popen success, not preflight/invocation count
                create_private(root / (scale + ".control.started.json"), boundary.canonical({"pid": pid}), 0o400)
            raw, receipt = boundary.sample_control(self._paths["artifact"], self._binding.artifact_sha256,
                doc["comparator"], doc["rows"] * factor, seed_uint64(doc[key]), trusted_runtime=self.runtime,
                workdir=root / ("control-" + scale), wall_seconds=remaining, capture_result=capture, capture_started=spawned,
                **({"byte_budget": doc["byte_budget"]} if self.runtime["abi"] == 2 else {}))
            actual_request = read_regular(root / ("control-" + scale) / "runtime/request.json")
            require(actual_request == boundary.canonical(request))
            stderr = read_regular(root / (scale + ".control.stderr"))
            require(read_regular(root / (scale + ".control.stdout")) == raw)
            spec = DrawSpec(scale, doc["comparator"], doc["rows"] * factor, len(self._binding.ordered_column_ids),
                doc[key], self._binding.artifact_sha256, doc["schema_sha256"], self._input_sha, doc["byte_budget"])
            draws.append(TypedControlDraw(spec, self._binding, raw, stderr, actual_request,
                read_regular(self._paths["artifact"], boundary.MAX_ARTIFACT), self._schema, self._input,
                self._source, self._compile_receipt, self.admission_manifest, files, 0, self._verify_draw))
        self._files()  # detect persistent closure changes before success publication
        self._manifest = store_pair(root, *draws, artifact=self._paths["artifact"])
        return {"sample_sha256": sha(draws[0].raw), "rows": doc["rows"], "seed": doc["draw_seed"],
            "filesystem_network_exec": "denied", "inherited_source_fds": "closed",
            "materialized_draw_manifest_sha256": self._manifest, "complete_sampling_bytes": max(charges)}

    def load_pair(self, directory, expected_manifest_sha256, *, artifact):
        require(self._manifest is not None and expected_manifest_sha256 == self._manifest
                and Path(artifact) == self._paths["artifact"])
        self._unchanged()
        return load_pair(directory, expected_manifest_sha256, artifact=artifact,
                         runtime_files=self._files(), trusted_verifier=self._verify_draw)

    def stage(self, stage, journal, cell_plan, paths, root):
        started, payload, status = time.monotonic(), None, "failed"
        try:
            require(type(journal) is StageJournal and journal.fd is not None
                    and journal.create is True and journal.read_only is False
                    and journal.spec.sha256 == self._spec_sha and journal.directory == root
                    and self._bound and cell_plan.sha256 == self._cell_sha and paths == self._paths
                    and stage in ("compile", "reload", "sample") and stage not in self._served
                    and journal.state.get((self._doc["cell_id"], stage), {}).get("pending") is True)
            self._served.append(stage)
            self._unchanged()
            self._authorize(stage, {"cell_plan_sha256": self._cell_sha,
                "input_sha256": self._input_sha, "compiler_fit_seed": self._doc["fit_seed"],
                "stage_wall_ms": self._doc["stage_wall_ms"][stage], "byte_budget": self._doc["byte_budget"]})
            self._unchanged()  # admission callbacks cannot substitute captured inputs
            payload = (self._compile(root) if stage == "compile" else
                       self._reload() if stage == "reload" else self._sample(root, started))
            self._unchanged()
            status = "over_budget" if payload is None else "completed"
            if payload is not None:
                from .pilot_runner import _verify_stage_payload
                _verify_stage_payload(stage, payload, self._doc, self._binding.artifact_sha256,
                                      paths["artifact"].stat().st_size, self._manifest)
        except (ValueError, TypeError, LookupError, ArithmeticError, RuntimeError, OSError):
            payload, status = None, "failed"
        # Unclassified callback faults and journal persistence faults are never
        # converted into successful settlement. The parent owns journal.finish.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        outcome = {"status": status, "returncode": 0 if status == "completed" else 1,
            "wall_ms": max(1, int((time.monotonic() - started) * 1000 + 1)), "peak_ram_bytes": peak,
            "resource_scope": "parent_lifetime_rss_upper_bound_not_child_or_gpu_peak"}
        return outcome, payload, self._binding.artifact_sha256 if self._binding else None
