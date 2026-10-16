#!/usr/bin/env python3
"""Reduce the pinned compute-cost receipt bundle to the committed cost ledger.

ops/ci/paper-prepare.py materializes generated/compute-cost-receipts.json from
its locked copy. This reducer reads that bundle and the two committed ledgers it
needs only through their registered lengths and digests (pinned_inputs.py), so
a changed, substituted, or redirected body is refused before it is decoded.
Rebuilding the bundle from scratch receipts is held: those receipts have no
pinned inventory, so a scratch mount is never read. The reducer does not invent
a missing timer, host, or device. Official test files are not opened.
"""

from __future__ import annotations

import copy
import json
import statistics
from collections import defaultdict
from pathlib import Path

from pinned_inputs import Registry, present, read_pinned
from public_hardware import banned_hits, redact_bundle, seal_cost_row

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "benchmark" / "results"
GENERATED = REPO / "docs" / "whitepaper" / "generated"
OUT = GENERATED / "compute-cost.json"
BUNDLE = "compute-cost-receipts.json"
FOREST_LEDGER = "s3-matched-forest-confirmation-validation.json"
ARF_LEDGER = "arf-s3-population-native.json"
# Byte length and sha256 of every input. The bundle digest equals its entry in
# ops/ci/paper-inputs/inputs.lock.json; a test holds the two together.
BUNDLE_PINS = Registry(GENERATED, {
    BUNDLE: (73961, "16cd9f392d96d3cfabefe09777d32efd35aadf443b0f324805595e461d30ece9"),
})
LEDGER_PINS = Registry(RESULTS, {
    FOREST_LEDGER: (1237755, "160b48dad42295131b758a2ef76074a2c645f914aa3c831f4f51924f928ead91"),
    ARF_LEDGER: (1477635, "11394f3aa0edfd35521cd8773a7db89919ea2a554ccef5f0c8e16e4538225f2e"),
})


def _ledger(name, registry=None):
    return json.loads(read_pinned(registry or LEDGER_PINS, name))


def _median(values):
    if not values:
        return None
    return float(statistics.median(values))


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
    if not present(LEDGER_PINS, ARF_LEDGER):
        row["notes"] = ["population native ledger is absent"]
        return row
    document = _ledger(ARF_LEDGER)
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


def _load_bundle(registry=None):
    registry = registry or BUNDLE_PINS
    if not present(registry, BUNDLE):
        raise FileNotFoundError("the pinned compute-cost receipt bundle is missing; run ops/ci/paper-prepare.py")
    bundle = json.loads(read_pinned(registry, BUNDLE))
    if bundle.get("format") != "dope-paper-compute-cost-receipts" or bundle.get("version") != 1:
        raise ValueError("compute-cost receipts bundle is not the required format")
    if bundle.get("official_tests_opened"):
        raise ValueError("compute-cost receipts opened an official test")
    if redact_bundle(copy.deepcopy(bundle)) != bundle or banned_hits(json.dumps(bundle)):
        raise ValueError("compute-cost receipts still contain a private token")
    return bundle


def _methods_from_bundle(bundle):
    forest_ledger = _ledger(FOREST_LEDGER)
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


def build(registry=None, out=OUT):
    methods = _methods_from_bundle(_load_bundle(registry))
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
