#!/usr/bin/env python3
"""Reduce fit and sample receipts to the committed compute-cost ledger.

When scratch is mounted, refresh the committed receipt bundle from those
files. The cost JSON is always reduced from that bundle, including in CI,
where scratch is absent. A missing bundle is an error. The reducer does not
invent a missing timer, host, or device. Official test files are not opened.
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from public_hardware import banned_hits, redact_bundle, seal_cost_row

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "benchmark" / "results"
GENERATED = REPO / "docs" / "whitepaper" / "generated"
OUT = GENERATED / "compute-cost.json"
RECEIPTS = GENERATED / "compute-cost-receipts.json"
SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")


def _load(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = path.resolve()
    under_scratch = SCRATCH.resolve() in resolved.parents or resolved == SCRATCH.resolve()
    under_results = str(resolved).startswith(str(RESULTS.resolve()))
    if not under_scratch and not under_results:
        raise ValueError(f"refusing path outside receipts and ledgers: {path}")
    return json.loads(path.read_text())


def _median(values):
    if not values:
        return None
    return float(statistics.median(values))


def _gpu_name(fitted, directory, host):
    """One GPU name from the fit receipt, or None when the receipt does not record one.

    Admission snapshots contribute only the snapshot whose host equals the fit
    host. A fit.json ``gpu`` string or ``gpu.name`` is the same receipt. Two
    different names are an error, not a choice.
    """
    found = []
    if directory is not None and host:
        for path in sorted(Path(directory).glob("fit*.admission-*.json")):
            document = _load(path)
            for snap in document.get("owner_snapshots") or []:
                if snap.get("host") != host:
                    continue
                for gpu in (snap.get("inventory") or {}).get("gpus") or []:
                    if isinstance(gpu, dict) and isinstance(gpu.get("name"), str):
                        found.append(gpu["name"])
    gpu = fitted.get("gpu") if isinstance(fitted, dict) else None
    if isinstance(gpu, str):
        found.append(gpu)
    elif isinstance(gpu, dict) and isinstance(gpu.get("name"), str):
        found.append(gpu["name"])
    unique = sorted(set(found))
    if len(unique) > 1:
        raise ValueError(f"conflicting GPU names: {unique}")
    return unique[0] if unique else None


def _assign_gpu(row, observations):
    per_host = defaultdict(list)
    for host, name in observations:
        if isinstance(host, str):
            per_host[host].append(name)
    gpu_by_host = {}
    for host, names in sorted(per_host.items()):
        present = sorted({name for name in names if name})
        missing = sum(1 for name in names if not name)
        if len(present) > 1:
            raise ValueError(f"{row['method']} has conflicting GPU names {present}")
        if missing == 0 and len(present) == 1:
            gpu_by_host[host] = present[0]
        else:
            gpu_by_host[host] = None
            if present:
                row["notes"].append("GPU name is not on every fit for one machine")
    row["hosts"] = sorted(per_host)
    row["gpu_by_host"] = gpu_by_host
    row["host_fits"] = {host: len(names) for host, names in sorted(per_host.items())}


def _blank(method):
    return {
        "method": method,
        "fit_n": 0,
        "fit_median_seconds": None,
        "fit_max_seconds": None,
        "fit_field": None,
        "sample_n": 0,
        "sample_median_seconds": None,
        "sample_max_seconds": None,
        "sample_field": None,
        "attempt_n": 0,
        "attempt_median_seconds": None,
        "attempt_max_seconds": None,
        "attempt_field": None,
        "peak_ram_bytes": None,
        "peak_vram_mib": None,
        "hosts": [],
        "gpu_by_host": {},
        "cpu_model": None,
        "cpu_hours": None,
        "gpu_hours": None,
        "gpu_seconds": None,
        "gpu_hour_scope": None,
        "notes": [],
    }


def _finish(row, fit_walls, sample_walls, attempt_walls, rams, vrams, gpu_seconds, scope):
    row["fit_n"] = len(fit_walls)
    row["fit_median_seconds"] = _median(fit_walls)
    row["fit_max_seconds"] = max(fit_walls) if fit_walls else None
    row["sample_n"] = len(sample_walls)
    row["sample_median_seconds"] = _median(sample_walls)
    row["sample_max_seconds"] = max(sample_walls) if sample_walls else None
    row["attempt_n"] = len(attempt_walls)
    row["attempt_median_seconds"] = _median(attempt_walls)
    row["attempt_max_seconds"] = max(attempt_walls) if attempt_walls else None
    row["peak_ram_bytes"] = max(rams) if rams else None
    row["peak_vram_mib"] = max(vrams) if vrams else None
    if gpu_seconds:
        row["gpu_seconds"] = float(sum(gpu_seconds))
        row["gpu_hours"] = row["gpu_seconds"] / 3600.0
        row["gpu_hour_scope"] = scope
    row["notes"] = sorted(set(row["notes"]))
    return row


def _int_or_none(value):
    return value if isinstance(value, int) else None


def _host(value):
    return value if isinstance(value, str) else None


def _collect_density():
    ledger = _load(RESULTS / "density-matched-population-validation.json")
    wanted = {
        "DOPE": "features12_steps2048",
        "GaussianCopula": "native_selected",
        "Chow-Liu": "native_selected",
        "independent_marginals": "native_selected",
    }
    dope = []
    seen = set()
    attempts = {name: [] for name in wanted if name != "DOPE"}
    seen_attempts = {name: set() for name in attempts}
    for cell in ledger["cells"]:
        config = wanted.get(cell.get("method"))
        if config is None:
            continue
        if cell.get("configuration") != config or cell.get("size_multiplier") != 4:
            continue
        if cell.get("status") != "ok":
            continue
        if cell["method"] == "DOPE":
            receipt_path = cell["fit_receipt"]["path"]
            if receipt_path in seen:
                continue
            seen.add(receipt_path)
            directory = Path(receipt_path).parent
            receipt = _load(receipt_path)
            if receipt.get("official_tests_opened"):
                raise ValueError("DOPE fit receipt opened an official test")
            if (directory / "fit.json").is_file():
                fitted_path = directory / "fit.json"
                admission_dir = directory
            else:
                reuse = _load(directory / "reuse.json")
                if reuse.get("new_gpu_fit_started") is not False:
                    raise ValueError("DOPE reuse receipt is not a prior fit")
                fitted_path = Path(reuse["prior_fit"]["receipt"])
                admission_dir = fitted_path.parent
            fitted = _load(fitted_path)
            if fitted.get("official_tests_opened"):
                raise ValueError("DOPE fit.json opened an official test")
            if not isinstance(fitted.get("elapsed_seconds"), (int, float)):
                continue
            host = _host(fitted.get("host"))
            dope.append({
                "fit_path": str(fitted_path),
                "elapsed_seconds": float(fitted["elapsed_seconds"]),
                "host": host,
                "gpu": _gpu_name(fitted, admission_dir, host),
                "peak_resident_bytes": _int_or_none(fitted.get("peak_resident_bytes")),
                "peak_gpu_used_mib": _int_or_none(fitted.get("peak_gpu_used_mib")),
                "gpu_process_observed": fitted.get("gpu_process_observed") is True,
            })
        else:
            attempt_path = cell.get("fit_attempt_receipt_path")
            name = cell["method"]
            if not isinstance(attempt_path, str) or attempt_path in seen_attempts[name]:
                continue
            seen_attempts[name].add(attempt_path)
            attempt = _load(attempt_path)
            if not isinstance(attempt.get("wall_seconds"), (int, float)):
                continue
            attempts[name].append(float(attempt["wall_seconds"]))
    return {"dope": dope, "attempts": attempts}


def _reduce_dope(records):
    row = _blank("DOPE")
    row["fit_field"] = "fit.json elapsed_seconds"
    row["notes"] = [
        "sample wall is not a separate elapsed_seconds field",
        "gpu hours count each physical fit.json once",
    ]
    _assign_gpu(row, [(item.get("host"), item.get("gpu")) for item in records])
    seen_hours = set()
    gpu_seconds = []
    for item in records:
        if item.get("gpu_process_observed") is True and item["fit_path"] not in seen_hours:
            seen_hours.add(item["fit_path"])
            gpu_seconds.append(float(item["elapsed_seconds"]))
    return _finish(
        row,
        [float(item["elapsed_seconds"]) for item in records],
        [],
        [],
        [item["peak_resident_bytes"] for item in records if isinstance(item.get("peak_resident_bytes"), int)],
        [item["peak_gpu_used_mib"] for item in records if isinstance(item.get("peak_gpu_used_mib"), int)],
        gpu_seconds,
        "unique fit.json elapsed_seconds",
    )


def _reduce_attempt(name, walls):
    row = _blank(name)
    row["attempt_field"] = "attempt.json wall_seconds"
    row["notes"] = ["one attempt wall_seconds is not split into fit and sample"]
    _assign_gpu(row, [])
    return _finish(row, [], [], [float(value) for value in walls], [], [], [], None)


def _monitor_record(monitor):
    return {
        "peak_resident_bytes": _int_or_none(monitor.get("peak_resident_bytes")),
        "peak_gpu_used_mib": _int_or_none(monitor.get("peak_gpu_used_mib")),
        "gpu_process_observed": monitor.get("gpu_process_observed") is True,
    }


def _collect_neural():
    ledger = _load(RESULTS / "sdv-matched-population-validation.json")
    rows = {name: [] for name in ("CTGAN", "TVAE")}
    seen = {name: set() for name in rows}
    for cell in ledger["cells"]:
        name = cell.get("method")
        if name not in rows:
            continue
        if cell.get("configuration") != "native_selected" or cell.get("size_multiplier") != 4:
            continue
        if cell.get("status") != "ok":
            continue
        sample_path = (cell.get("sample_evidence") or {}).get("path")
        if not isinstance(sample_path, str):
            continue
        directory = Path(sample_path).parent
        if str(directory) in seen[name]:
            continue
        seen[name].add(str(directory))
        fit_op = _load(directory / "fit.operation.json")
        sample_op = _load(directory / "sample.operation.json")
        fitted = _load(directory / "fit.json")
        if fit_op.get("official_tests_opened") or sample_op.get("official_tests_opened"):
            raise ValueError(f"{name} receipt opened an official test")
        if not isinstance(fit_op.get("elapsed_seconds"), (int, float)):
            raise ValueError(f"{name} fit operation has no elapsed_seconds")
        if not isinstance(sample_op.get("elapsed_seconds"), (int, float)):
            raise ValueError(f"{name} sample operation has no elapsed_seconds")
        host = _host(fitted.get("host")) or _host(fit_op.get("host"))
        monitors = []
        for monitor_name in ("fit.monitor.json", "sample.monitor.json"):
            monitors.append(_monitor_record(_load(directory / monitor_name)))
        rows[name].append({
            "fit_seconds": float(fit_op["elapsed_seconds"]),
            "sample_seconds": float(sample_op["elapsed_seconds"]),
            "host": host,
            "gpu": _gpu_name(fitted, directory, host),
            "monitors": monitors,
        })
    return rows


def _reduce_neural(name, records):
    row = _blank(name)
    row["fit_field"] = "fit.operation.json elapsed_seconds"
    row["sample_field"] = "sample.operation.json elapsed_seconds"
    _assign_gpu(row, [(item.get("host"), item.get("gpu")) for item in records])
    rams, vrams, gpu_seconds = [], [], []
    for item in records:
        for monitor, role in zip(item["monitors"], ("fit", "sample")):
            if isinstance(monitor.get("peak_resident_bytes"), int):
                rams.append(monitor["peak_resident_bytes"])
            if isinstance(monitor.get("peak_gpu_used_mib"), int):
                vrams.append(monitor["peak_gpu_used_mib"])
            observed = monitor.get("gpu_process_observed") is True or (
                isinstance(monitor.get("peak_gpu_used_mib"), int) and monitor["peak_gpu_used_mib"] > 0
            )
            if observed:
                gpu_seconds.append(item["fit_seconds"] if role == "fit" else item["sample_seconds"])
    return _finish(
        row,
        [item["fit_seconds"] for item in records],
        [item["sample_seconds"] for item in records],
        [],
        rams,
        vrams,
        gpu_seconds,
        "fit.operation.json and sample.operation.json elapsed_seconds",
    )


def _collect_forest():
    ledger = _load(RESULTS / "s3-matched-forest-confirmation-validation.json")
    attempts = []
    for attempt in ledger["native_attempts"]:
        directory = Path(attempt["receipt"]["path"]).parent
        receipt = _load(directory / "receipt.json")
        fitted = _load(directory / "fit.json")
        if receipt.get("official_tests_opened") or fitted.get("official_tests_opened"):
            raise ValueError("Forest-Flow receipt opened an official test")
        fit_op = _load(directory / "fit.operation.json")
        sample_op = _load(directory / "sample.operation.json")
        if not isinstance(fit_op.get("elapsed_seconds"), (int, float)):
            raise ValueError("Forest-Flow fit operation has no elapsed_seconds")
        if not isinstance(sample_op.get("elapsed_seconds"), (int, float)):
            raise ValueError("Forest-Flow sample operation has no elapsed_seconds")
        if not isinstance(fitted.get("fit_seconds"), (int, float)):
            raise ValueError("Forest-Flow fit.json has no fit_seconds")
        host = _host(fitted.get("host")) or _host(fit_op.get("host"))
        operations = []
        for operation in receipt.get("operations") or []:
            if operation.get("new_operation_started") and isinstance(operation.get("elapsed_seconds"), (int, float)):
                operations.append(float(operation["elapsed_seconds"]))
        monitors = [
            _monitor_record(_load(directory / name))
            for name in ("fit.monitor.json", "sample.monitor.json", "native.monitor.json")
        ]
        attempts.append({
            "fit_seconds": float(fitted["fit_seconds"]),
            "fit_op_seconds": float(fit_op["elapsed_seconds"]),
            "sample_op_seconds": float(sample_op["elapsed_seconds"]),
            "host": host,
            "gpu": _gpu_name(fitted, directory, host),
            "operations": operations,
            "monitors": monitors,
        })
    return attempts


def _reduce_forest(attempts, expected_cost):
    row = _blank("Forest-Flow")
    row["fit_field"] = "fit.operation.json elapsed_seconds"
    row["sample_field"] = "sample.operation.json elapsed_seconds"
    row["notes"] = [
        "twelve native-grid fits, not only the six selected artifacts",
        "gpu hours sum receipt operations with new_operation_started",
    ]
    _assign_gpu(row, [(item.get("host"), item.get("gpu")) for item in attempts])
    rams, vrams, gpu_seconds = [], [], []
    for item in attempts:
        gpu_seconds.extend(float(value) for value in item["operations"])
        for monitor in item["monitors"]:
            if isinstance(monitor.get("peak_resident_bytes"), int):
                rams.append(monitor["peak_resident_bytes"])
            if isinstance(monitor.get("peak_gpu_used_mib"), int):
                vrams.append(monitor["peak_gpu_used_mib"])
    finished = _finish(
        row,
        [item["fit_op_seconds"] for item in attempts],
        [item["sample_op_seconds"] for item in attempts],
        [],
        rams,
        vrams,
        gpu_seconds,
        "receipt operations with new_operation_started",
    )
    core_sum = sum(item["fit_seconds"] for item in attempts)
    operation_sum = sum(gpu_seconds)
    if abs(core_sum - float(expected_cost["forest_fit_core_seconds"])) > 1e-6:
        raise ValueError("Forest-Flow fit_seconds sum disagrees with the committed cost")
    if abs(operation_sum - float(expected_cost["forest_fit_native_sample_operation_seconds"])) > 1e-4:
        raise ValueError("Forest-Flow operation sum disagrees with the committed cost")
    if not vrams or max(vrams) != int(expected_cost["forest_peak_gpu_used_mib"]):
        raise ValueError("Forest-Flow peak VRAM disagrees with the committed cost")
    return finished


def _arf():
    """Host and selected bytes from the committed population ledger.

    That ledger stores operation_seconds, not a fit or sample elapsed_seconds.
    Those operation times are not copied into the fit or sample fields.
    Peak resident bytes include the coordinator, so they are not job peak RAM.
    """
    row = _blank("ARF")
    path = RESULTS / "arf-s3-population-native.json"
    if not path.is_file():
        row["notes"] = ["population native ledger is absent"]
        return row
    document = _load(path)
    if document.get("official_tests_opened"):
        raise ValueError("ARF population ledger opened an official test")
    hosts = sorted({
        trial.get("host") for trial in document.get("trials") or []
        if trial.get("status") == "ok" and isinstance(trial.get("host"), str)
    })
    row["hosts"] = hosts
    row["gpu_by_host"] = {host: None for host in hosts}
    row["host_fits"] = {
        host: sum(1 for trial in document["trials"] if trial.get("host") == host and trial.get("status") == "ok")
        for host in hosts
    }
    amounts = [
        cell["selected_charged_bytes"]
        for cell in document.get("cells") or []
        if isinstance(cell.get("selected_charged_bytes"), int)
    ]
    if amounts:
        row["artifact_bytes"] = {
            "n": len(amounts),
            "min": min(amounts),
            "median": float(statistics.median(amounts)),
            "max": max(amounts),
            "field": "selected_charged_bytes",
        }
    row["notes"] = [
        "fit and sample elapsed_seconds are absent",
        "operation_seconds is not a fit or sample elapsed_seconds",
        "peak_resident_bytes_including_coordinator is not job peak RAM",
        "shared validation sampling is pending",
    ]
    return row


def _dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_bundle():
    if not RECEIPTS.is_file():
        raise FileNotFoundError(
            "scratch is not mounted and committed compute-cost receipts are missing"
        )
    bundle = json.loads(RECEIPTS.read_text())
    if bundle.get("format") != "dope-paper-compute-cost-receipts" or bundle.get("version") != 1:
        raise ValueError("compute-cost receipts bundle is not the required format")
    if bundle.get("official_tests_opened"):
        raise ValueError("compute-cost receipts opened an official test")
    return bundle


def _methods_from_bundle(bundle):
    forest_ledger = _load(RESULTS / "s3-matched-forest-confirmation-validation.json")
    density = bundle["density"]
    neural = bundle["neural"]
    return {
        "DOPE": _reduce_dope(density["dope"]),
        "GaussianCopula": _reduce_attempt("GaussianCopula", density["attempts"]["GaussianCopula"]),
        "Chow-Liu": _reduce_attempt("Chow-Liu", density["attempts"]["Chow-Liu"]),
        "independent_marginals": _reduce_attempt(
            "independent_marginals", density["attempts"]["independent_marginals"]
        ),
        "CTGAN": _reduce_neural("CTGAN", neural["CTGAN"]),
        "TVAE": _reduce_neural("TVAE", neural["TVAE"]),
        "Forest-Flow": _reduce_forest(bundle["forest"], forest_ledger["cost"]),
        "ARF": _arf(),
    }


def _fresh_bundle():
    return {
        "format": "dope-paper-compute-cost-receipts",
        "version": 1,
        "official_tests_opened": False,
        "density": _collect_density(),
        "neural": _collect_neural(),
        "forest": _collect_forest(),
    }


def _science(bundle):
    """Redacted JSON text. Host names and fit paths do not affect the compare."""
    redacted = redact_bundle(copy.deepcopy(bundle))
    return json.dumps(redacted, sort_keys=True)


def build():
    committed = json.loads(RECEIPTS.read_text()) if RECEIPTS.is_file() else None
    if SCRATCH.is_dir():
        fresh = _fresh_bundle()
        if committed is not None and _science(fresh) != _science(committed):
            print(
                "scratch receipts differ from the committed bundle; keeping committed numbers",
                file=sys.stderr,
            )
            bundle = committed
        else:
            bundle = fresh
    else:
        bundle = _load_bundle()
        print("scratch absent; reducing committed compute-cost receipts")
    bundle = redact_bundle(bundle)
    rendered = json.dumps(bundle)
    if banned_hits(rendered):
        raise ValueError("compute-cost receipts still contain a private token")
    _dump(RECEIPTS, bundle)
    methods = _methods_from_bundle(bundle)
    for row in methods.values():
        row["cpu_model"] = None
        row["cpu_hours"] = None
        seal_cost_row(row)
    payload = {
        "format": "dope-paper-compute-cost",
        "version": 1,
        "official_tests_opened": False,
        "missing_field": "not recorded",
        "methods": methods,
    }
    rendered_cost = json.dumps(payload)
    if banned_hits(rendered_cost):
        raise ValueError("compute-cost ledger still contains a private token")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
