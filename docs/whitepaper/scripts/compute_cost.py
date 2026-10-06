#!/usr/bin/env python3
"""Reduce existing fit and sample receipts to a committed compute-cost ledger.

Scratch receipts are read only when this script runs. The paper build reads
the committed JSON and does not invent a missing timer, host, or device.
Official test files are not opened.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "benchmark" / "results"
OUT = REPO / "docs" / "whitepaper" / "generated" / "compute-cost.json"
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
        raise ValueError(f"conflicting GPU names on {host}: {unique}")
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
            raise ValueError(f"{row['method']} host {host} has conflicting GPU names {present}")
        if missing == 0 and len(present) == 1:
            gpu_by_host[host] = present[0]
        else:
            gpu_by_host[host] = None
            if present:
                row["notes"].append(f"{host} GPU name is not on every fit")
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


def _density_methods():
    ledger = _load(RESULTS / "density-matched-population-validation.json")
    wanted = {
        "DOPE": "features12_steps2048",
        "GaussianCopula": "native_selected",
        "Chow-Liu": "native_selected",
        "independent_marginals": "native_selected",
    }
    grouped = {name: _blank(name) for name in wanted}
    seen = {name: set() for name in wanted}
    seen_hours = set()
    fit_walls = {name: [] for name in wanted}
    attempt_walls = {name: [] for name in wanted}
    rams = {name: [] for name in wanted}
    vrams = {name: [] for name in wanted}
    gpu_seconds = {name: [] for name in wanted}
    observations = {name: [] for name in wanted}
    for cell in ledger["cells"]:
        config = wanted.get(cell.get("method"))
        if config is None:
            continue
        if cell.get("configuration") != config or cell.get("size_multiplier") != 4:
            continue
        if cell.get("status") != "ok":
            continue
        if cell["method"] == "DOPE":
            path = cell["fit_receipt"]["path"]
            if path in seen["DOPE"]:
                continue
            seen["DOPE"].add(path)
            directory = Path(path).parent
            receipt = _load(path)
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
                grouped["DOPE"]["notes"].append("a displayed fit has no elapsed_seconds")
                continue
            wall = float(fitted["elapsed_seconds"])
            fit_walls["DOPE"].append(wall)
            host = fitted.get("host") if isinstance(fitted.get("host"), str) else None
            observations["DOPE"].append((host, _gpu_name(fitted, admission_dir, host)))
            if isinstance(fitted.get("peak_resident_bytes"), int):
                rams["DOPE"].append(fitted["peak_resident_bytes"])
            if isinstance(fitted.get("peak_gpu_used_mib"), int):
                vrams["DOPE"].append(fitted["peak_gpu_used_mib"])
            if fitted.get("gpu_process_observed") is True and str(fitted_path) not in seen_hours:
                seen_hours.add(str(fitted_path))
                gpu_seconds["DOPE"].append(wall)
        else:
            path = cell.get("fit_attempt_receipt_path")
            if not isinstance(path, str) or path in seen[cell["method"]]:
                continue
            seen[cell["method"]].add(path)
            attempt = _load(path)
            if not isinstance(attempt.get("wall_seconds"), (int, float)):
                continue
            attempt_walls[cell["method"]].append(float(attempt["wall_seconds"]))
    grouped["DOPE"]["fit_field"] = "fit.json elapsed_seconds"
    grouped["DOPE"]["notes"] = [
        "sample wall is not a separate elapsed_seconds field",
        "gpu hours count each physical fit.json once",
    ]
    _assign_gpu(grouped["DOPE"], observations["DOPE"])
    _finish(
        grouped["DOPE"], fit_walls["DOPE"], [], [], rams["DOPE"], vrams["DOPE"],
        gpu_seconds["DOPE"], "unique fit.json elapsed_seconds",
    )
    for name in ("GaussianCopula", "Chow-Liu", "independent_marginals"):
        grouped[name]["attempt_field"] = "attempt.json wall_seconds"
        grouped[name]["notes"] = ["one attempt wall_seconds is not split into fit and sample"]
        _assign_gpu(grouped[name], [])
        _finish(grouped[name], [], [], attempt_walls[name], [], [], [], None)
    return grouped


def _neural():
    ledger = _load(RESULTS / "sdv-matched-population-validation.json")
    rows = {name: _blank(name) for name in ("CTGAN", "TVAE")}
    bags = {
        name: {"fit": [], "sample": [], "ram": [], "vram": [], "gpu_s": [], "obs": []}
        for name in rows
    }
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
        fit_wall = float(fit_op["elapsed_seconds"])
        sample_wall = float(sample_op["elapsed_seconds"])
        bags[name]["fit"].append(fit_wall)
        bags[name]["sample"].append(sample_wall)
        host = fitted.get("host") if isinstance(fitted.get("host"), str) else fit_op.get("host")
        if not isinstance(host, str):
            host = None
        bags[name]["obs"].append((host, _gpu_name(fitted, directory, host)))
        for monitor_name in ("fit.monitor.json", "sample.monitor.json"):
            monitor = _load(directory / monitor_name)
            if isinstance(monitor.get("peak_resident_bytes"), int):
                bags[name]["ram"].append(monitor["peak_resident_bytes"])
            if isinstance(monitor.get("peak_gpu_used_mib"), int):
                bags[name]["vram"].append(monitor["peak_gpu_used_mib"])
            observed = monitor.get("gpu_process_observed") is True or (
                isinstance(monitor.get("peak_gpu_used_mib"), int) and monitor["peak_gpu_used_mib"] > 0
            )
            if observed:
                bags[name]["gpu_s"].append(fit_wall if monitor_name.startswith("fit") else sample_wall)
        rows[name]["fit_field"] = "fit.operation.json elapsed_seconds"
        rows[name]["sample_field"] = "sample.operation.json elapsed_seconds"
    for name, row in rows.items():
        bag = bags[name]
        _assign_gpu(row, bag["obs"])
        _finish(
            row, bag["fit"], bag["sample"], [], bag["ram"], bag["vram"], bag["gpu_s"],
            "fit.operation.json and sample.operation.json elapsed_seconds",
        )
    return rows


def _forest(expected_cost):
    ledger = _load(RESULTS / "s3-matched-forest-confirmation-validation.json")
    row = _blank("Forest-Flow")
    fit_walls, sample_walls, rams, vrams, gpu_seconds = [], [], [], [], []
    observations = []
    fit_core = []
    operation_sum = 0.0
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
        fit_walls.append(float(fit_op["elapsed_seconds"]))
        sample_walls.append(float(sample_op["elapsed_seconds"]))
        fit_core.append(float(fitted["fit_seconds"]))
        for operation in receipt.get("operations") or []:
            if operation.get("new_operation_started") and isinstance(operation.get("elapsed_seconds"), (int, float)):
                operation_sum += float(operation["elapsed_seconds"])
                gpu_seconds.append(float(operation["elapsed_seconds"]))
        host = fitted.get("host") if isinstance(fitted.get("host"), str) else fit_op.get("host")
        if not isinstance(host, str):
            host = None
        observations.append((host, _gpu_name(fitted, directory, host)))
        for monitor_name in ("fit.monitor.json", "sample.monitor.json", "native.monitor.json"):
            monitor = _load(directory / monitor_name)
            if isinstance(monitor.get("peak_resident_bytes"), int):
                rams.append(monitor["peak_resident_bytes"])
            if isinstance(monitor.get("peak_gpu_used_mib"), int):
                vrams.append(monitor["peak_gpu_used_mib"])
    row["fit_field"] = "fit.operation.json elapsed_seconds"
    row["sample_field"] = "sample.operation.json elapsed_seconds"
    row["notes"] = [
        "twelve native-grid fits, not only the six selected artifacts",
        "gpu hours sum receipt operations with new_operation_started",
    ]
    _assign_gpu(row, observations)
    _finish(
        row, fit_walls, sample_walls, [], rams, vrams, gpu_seconds,
        "receipt operations with new_operation_started",
    )
    core_sum = sum(fit_core)
    if abs(core_sum - float(expected_cost["forest_fit_core_seconds"])) > 1e-6:
        raise ValueError("Forest-Flow fit_seconds sum disagrees with the committed cost")
    if abs(operation_sum - float(expected_cost["forest_fit_native_sample_operation_seconds"])) > 1e-4:
        raise ValueError("Forest-Flow operation sum disagrees with the committed cost")
    if not vrams or max(vrams) != int(expected_cost["forest_peak_gpu_used_mib"]):
        raise ValueError("Forest-Flow peak VRAM disagrees with the committed cost")
    return row


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


def build():
    if not SCRATCH.is_dir():
        if OUT.is_file():
            print("scratch absent; committed compute-cost.json kept")
            return 0
        raise FileNotFoundError("compute receipts are not mounted and compute-cost.json is missing")
    density = _density_methods()
    neural = _neural()
    forest_ledger = _load(RESULTS / "s3-matched-forest-confirmation-validation.json")
    forest = _forest(forest_ledger["cost"])
    methods = {
        "DOPE": density["DOPE"],
        "GaussianCopula": density["GaussianCopula"],
        "Chow-Liu": density["Chow-Liu"],
        "independent_marginals": density["independent_marginals"],
        "CTGAN": neural["CTGAN"],
        "TVAE": neural["TVAE"],
        "Forest-Flow": forest,
        "ARF": _arf(),
    }
    for row in methods.values():
        row["cpu_model"] = None
        row["cpu_hours"] = None
    payload = {
        "format": "dope-paper-compute-cost",
        "version": 1,
        "official_tests_opened": False,
        "missing_field": "not recorded",
        "methods": methods,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
