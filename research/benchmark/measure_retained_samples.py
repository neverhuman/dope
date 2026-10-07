"""Measure retained common-numeric samples using the frozen validation equations.

This command reads TRAIN-derived validation only. It does not fit generators,
select configurations, load checkpoints, or make certification claims.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import warnings


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_verified(ref, limit=64_000_000):
    path = Path(ref["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("invalid_input_file")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("input_digest_mismatch")
    if "bytes" in ref and len(data) != ref["bytes"]:
        raise ValueError("input_size_mismatch")
    return data


def capacity(meminfo=None):
    text = Path("/proc/meminfo").read_text() if meminfo is None else meminfo
    facts = {line.split(":")[0]: int(line.split()[1]) * 1024
             for line in text.splitlines() if line.startswith(("MemTotal:", "MemAvailable:"))}
    if facts["MemAvailable"] * 100 < facts["MemTotal"] * 15 + 4 * 1024**3 * 100:
        raise ValueError("insufficient_ram_above_fifteen_percent_floor")
    return {**facts, "free_ram_floor_fraction": .15, "reserved_bytes": 4 * 1024**3}


def check_job(job):
    if digest(job["job_identity"]) != job["job_sha256"]:
        raise ValueError("job_identity_mismatch")
    cell = job["evaluator_input"]
    if cell.get("official_tests_opened") is not False:
        raise ValueError("validation_only_required")
    if cell["column_order"] != "projected_inputs_then_target":
        raise ValueError("target_last_required")
    for name in ("train_ref", "validation_ref", "synthetic_ref"):
        if Path(cell[name]["path"]).name in ("test.csv", "test.tsv"):
            raise ValueError("sealed_test_forbidden")
    if cell.get("sample_CSV_to_common_permutation", list(range(len(cell["column_kinds"])))) != list(range(len(cell["column_kinds"]))):
        raise ValueError("unexpected_column_permutation")
    return cell


def load_source(ref, name):
    allowed = {
        "frozen_expanded_metrics": "a952062c0f83805f6442a440a5ae15293a843349d423eaa106a9c56a20d14c0f",
        "frozen_pilot_metrics": "6c03892685df856f7a491ca79acb1c28fe8270ead8bbfea1ba50228cb6f0e7e3",
    }
    if name not in allowed or ref["sha256"] != allowed[name]:
        raise ValueError("unapproved_metric_implementation")
    data = read_verified(ref)
    if Path(importlib.util.cache_from_source(ref["path"])).exists():
        raise ValueError("unexpected_metric_bytecode")
    class CapturedSource(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return self.source_to_code(data, ref["path"])

    spec = importlib.util.spec_from_file_location(name, ref["path"],
                                                 loader=CapturedSource(name, ref["path"]))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utility_outcome(null_loss, trtr, tstr):
    gain = null_loss - trtr
    informative = null_loss > 0 and gain > 0 and gain >= .01 * abs(null_loss)
    return dict(status="ok", trtr_loss=trtr, tstr_loss=tstr, informative=informative,
                retention=(null_loss - tstr) / gain if informative else None,
                low_signal_noninferior=tstr <= trtr + .01 * null_loss if not informative else None)


def utility(real, valid, synth, task, pilot):
    import numpy as np
    from sklearn.metrics import log_loss, mean_squared_error
    if task not in ("regression", "binary"):
        raise ValueError("unsupported_target_task")
    if task == "binary" and any(not set(np.unique(x[:, -1])).issubset({0., 1.}) for x in (real, valid, synth)):
        raise ValueError("invalid_binary_target")
    null_prediction = float(real[:, -1].mean())
    prediction = np.full(len(valid), null_prediction)
    null_loss = (float(log_loss(valid[:, -1], np.clip(prediction, 1e-9, 1 - 1e-9), labels=[0, 1]))
                 if task == "binary" else float(mean_squared_error(valid[:, -1], prediction)))
    result = {}
    for name in ("catboost", "linear", "mlp"):
        try:
            real_model = pilot._model(name, task, 1729)
            synth_model = pilot._model(name, task, 1729)
            if name == "catboost":
                real_model.set_params(allow_writing_files=False, task_type="CPU")
                synth_model.set_params(allow_writing_files=False, task_type="CPU")
            real_model.fit(real[:, :-1], real[:, -1])
            synth_model.fit(synth[:, :-1], synth[:, -1])
            trtr = pilot._loss(real_model, valid[:, :-1], valid[:, -1], task)
            tstr = pilot._loss(synth_model, valid[:, :-1], valid[:, -1], task)
            result[name] = utility_outcome(null_loss, trtr, tstr)
        except (ValueError, RuntimeError) as error:
            result[name] = dict(status="unavailable", reason=type(error).__name__, retention=None)
    return dict(null_loss=null_loss, auditors=result, selection_use=False)


def run(manifest, output, source_refs):
    # Validate identity and input bytes before third-party initializers execute.
    cells = [check_job(job) for job in manifest["jobs"]]
    first = cells[0]
    for key in ("train_ref", "validation_ref", "synthetic_ref", "projection_ref"):
        read_verified(first[key])
    observation = capacity()
    metrics = load_source(source_refs["expanded_validation_metrics"], "frozen_expanded_metrics")
    pilot = load_source(source_refs["pilot_metrics"], "frozen_pilot_metrics")
    import numpy as np
    import scipy
    import sklearn
    import catboost
    versions = dict(numpy=np.__version__, scipy=scipy.__version__,
                    **{"scikit-learn": sklearn.__version__, "catboost": catboost.__version__})
    expected = dict(numpy="1.26.4", scipy="1.16.3", **{"scikit-learn": "1.7.2", "catboost": "1.2.10"})
    if versions != expected:
        raise ValueError("common_evaluator_dependency_version_mismatch")
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    started = time.time()
    for job, cell in zip(manifest["jobs"], cells):
        before = capacity()
        begin = time.perf_counter()
        refs = {key: cell[key] for key in ("train_ref", "validation_ref", "synthetic_ref", "projection_ref")}
        buffers = {key: read_verified(ref) for key, ref in refs.items()}
        import io
        arrays = [np.loadtxt(io.BytesIO(buffers[key]), delimiter=",", ndmin=2,
                             skiprows=cell.get("synthetic_csv_header_rows", 0) if key == "synthetic_ref" else 0)
                  for key in ("train_ref", "validation_ref", "synthetic_ref")]
        real, valid, synth = metrics.checked_arrays(*arrays)
        for key, array in zip(("train_ref", "validation_ref", "synthetic_ref"), arrays):
            rows = refs[key].get("row_count", refs[key].get("rows"))
            if len(array) != rows or array.shape[1] != len(cell["column_kinds"]):
                raise ValueError("input_shape_mismatch")
        projection = json.loads(buffers["projection_ref"])
        with warnings.catch_warnings(record=True) as notices:
            result = metrics.evaluate(real, valid, synth, cell["column_kinds"])
            result["utility"] = utility(real, valid, synth, projection["task"], pilot)
        for ref in refs.values():
            read_verified(ref)
        capacity()
        elapsed = time.perf_counter() - begin
        if elapsed > 600:
            raise ValueError("metric_operation_deadline")
        receipt = dict(format="dope-retained-validation-measurement", version=1,
                       job_sha256=job["job_sha256"], job_identity=job["job_identity"],
                       dataset=cell["dataset"], method=cell.get("method", "TabSyn"),
                       fit_seed=cell["fit_seed"], sample_seed=cell["sample_seed"],
                       row_multiplier=cell["row_multiplier"], config_sha256=cell["config_sha256"],
                       selection_binding=cell["selection_binding"],
                       physical_sampling_backend=cell.get("physical_sampling_backend"),
                       charged_artifact_bytes=cell["charged_artifact_bytes"],
                       input_refs=refs, source_refs=source_refs, dependencies=versions,
                       rows=dict(train=len(real), validation=len(valid), synthetic=len(synth)),
                       metrics=result, status="ok", warning_count=len(notices),
                       metric_seconds=elapsed, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       metric_operation_started_unix_seconds=started + (time.time() - started - elapsed),
                       original_generation_deadline_unix_seconds=cell.get("original_lineage_deadline_unix_seconds"),
                       generation_deadline_extended=False, resource_observation=before,
                       cpu_affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                       official_tests_opened=False, mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None)
        path = output / (job["job_sha256"] + ".json")
        data = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        path.write_bytes(data)
        receipts.append(dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        print(json.dumps(dict(done=len(receipts), total=len(cells), dataset=cell["dataset"],
                              seconds=elapsed, peak_rss_bytes=receipt["peak_rss_bytes"])), flush=True)
    (output / "receipt-lock.json").write_text(json.dumps(dict(format="dope-retained-validation-receipt-lock",
        receipts=receipts, count=len(receipts), manifest_sha256=digest(manifest),
        source_refs=source_refs, resource_observation=observation, actual_complete=True,
        official_tests_opened=False), sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency-site", action="append", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--runtime-inventory", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("cpu_environment_required")
    raw = read_verified(dict(path=str(args.manifest), sha256=args.manifest_sha256))
    manifest = json.loads(raw)
    refs = {name: {**ref, "path": str(args.source_dir / Path(ref["path"]).name)}
            for name, ref in manifest["committed_metric_source_refs"].items()}
    for ref in refs.values():
        read_verified(ref)
    runtime = json.loads(read_verified(dict(path=str(args.runtime_inventory), sha256=args.runtime_sha256)))
    capacity()
    for ref in runtime["files"]:
        read_verified(ref, limit=600_000_000)
    sys.path[:0] = args.dependency_site
    run(manifest, args.output, refs)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status="failed", error_type=type(error).__name__)), file=sys.stderr)
        raise SystemExit(1) from None
