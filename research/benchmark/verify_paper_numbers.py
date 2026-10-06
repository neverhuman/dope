#!/usr/bin/env python3
"""Trace manuscript numbers to fit.json files, receipts, and the ARF watch log.

Reads those inputs and does not rewrite them. ``--write-evidence`` stores the
reduction in docs/whitepaper/generated/fit-trace.json so the table generator
can format it. A later run with no flag exits nonzero when a macro, a
hand-typed token, or that evidence file disagrees. Retention medians stay in
the committed validation ledgers; this script checks the Forest-Flow byte
charges in those ledgers against hash-checked fit.json files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "research" / "benchmark" / "results"
PAPER = REPO / "docs" / "whitepaper"
GENERATED = PAPER / "generated"
TEX = PAPER / "dope-mfs.tex"
WATCH_LOG = Path(
    "/home/ubuntu/dope/.agent/worktrees/integration/target/arf-native-closure-watch-v1.log"
)
SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
LOSS_ROOT = SCRATCH / "dope-s3-loss-log-v1" / "replay"
BEYOND_PREPARE = SCRATCH / "beyondarena-prepared-v1" / "prepare-summary.json"
TOKEN = re.compile(r"\d+(?:\{,\}\d{3})+|\d+\.\d+|\d+")
CITE = re.compile(r"\\cite[tp]?\{[^{}]*\}")


def parse_watch(text):
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("ARF watch log is empty")
    counted = [row for row in rows if "closed" in row and "planned" in row and "ok" in row]
    if not counted:
        raise ValueError("ARF watch log has no closed/planned/ok record")
    last = counted[-1]
    terminal = rows[-1]
    return {
        "closed": int(last["closed"]),
        "ok": int(last["ok"]),
        "planned": int(last["planned"]),
        "count_utc": last.get("utc"),
        "terminal_phase": terminal.get("phase"),
        "terminal_exit_code": terminal.get("exit_code"),
        "new_generator_fits_started": int(last.get("new_generator_fits_started") or 0),
        "records": len(rows),
    }


def _paths(document):
    found = []

    def walk(node):
        if isinstance(node, dict):
            path = node.get("path")
            if isinstance(path, str) and path.endswith("fit.json"):
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)
    return found


def forest_fits():
    manifest = json.loads(
        (RESULTS / "s3-matched-forest-confirmation-validation.manifest.json").read_text()
    )
    ledger = json.loads(
        (RESULTS / "s3-matched-forest-confirmation-validation.json").read_text()
    )
    selected = []
    for row in ledger["summary"]:
        if (
            row.get("method") == "ForestDiffusion/Forest-Flow"
            and row.get("configuration") == "native_selected"
            and row.get("size_multiplier") == 4
            and isinstance(row.get("charged_artifact_bytes"), int)
        ):
            selected.append(row["charged_artifact_bytes"])
    fits = []
    for ref in _paths(manifest):
        raw = Path(ref["path"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != ref["sha256"]:
            raise ValueError(f"fit.json hash mismatch: {ref['path']}")
        document = json.loads(raw)
        receipt = json.loads(Path(ref["path"]).with_name("receipt.json").read_text())
        operation_seconds = sum(
            float(item["elapsed_seconds"])
            for item in receipt["operations"]
            if isinstance(item, dict) and "elapsed_seconds" in item
        )
        fits.append(
            {
                "artifact_bytes": int(document["artifact_bytes"]),
                "fit_seconds": float(document["fit_seconds"]),
                "status": document["status"],
                "mfs_v2": document["mfs_v2"],
                "official_tests_opened": document["official_tests_opened"],
                "operation_seconds": operation_seconds,
            }
        )
    amounts = {item["artifact_bytes"] for item in fits}
    missing = [amount for amount in selected if amount not in amounts]
    if missing:
        raise ValueError(f"validation bytes missing from fit.json: {missing}")
    if any(item["status"] != "ok" or item["mfs_v2"] is not None for item in fits):
        raise ValueError("a confirmation fit.json is not ok or has a non-null MFS-v2")
    if any(item["official_tests_opened"] for item in fits):
        raise ValueError("a confirmation fit.json opened an official test")
    return {
        "fit_count": len(fits),
        "fit_seconds_sum": sum(item["fit_seconds"] for item in fits),
        "operation_seconds_sum": sum(item["operation_seconds"] for item in fits),
        "selected_byte_min": min(selected),
        "selected_byte_max": max(selected),
        "selected_n": len(selected),
    }


def beyond_prepare():
    document = json.loads(BEYOND_PREPARE.read_text())
    feature_range = 0
    overlap = 0
    for row in document["results"]:
        if row["status"] == "prepared":
            continue
        detail = str(row.get("detail") or "")
        if "projected feature count" in detail:
            feature_range += 1
        elif "overlapping source row" in detail:
            overlap += 1
        else:
            raise ValueError(f"unclassified prepare failure: {row.get('name')}")
    inventory = json.loads((RESULTS / "beyondarena-s3-inventory.json").read_text())
    revision = str(inventory["revision"])
    return {
        "families": int(document["families"]),
        "prepared": int(document["prepared"]),
        "failed": int(document["failed"]),
        "feature_range": feature_range,
        "overlap": overlap,
        "inventory": int(inventory["dataset_count"]),
        "revision": revision,
        "official_test_opened_by_optimizer": bool(document["official_test_opened_by_optimizer"]),
    }


def loss_medians():
    """Point medians of the first and last logged step. Not a bootstrap."""
    found = {}
    for profile in ("features12_steps2048", "features12_steps8192"):
        trains0, valids0, trains1, valids1 = [], [], [], []
        for path in sorted((LOSS_ROOT / profile).glob("*/loss.tsv")):
            first = last = None
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        parts = line.split()
                        if first is None:
                            first = parts
                        last = parts
            if first is None or last is None:
                continue
            trains0.append(float(first[1]))
            valids0.append(float(first[2]))
            trains1.append(float(last[1]))
            valids1.append(float(last[2]))
        found[profile] = {
            "n": len(trains0),
            "step0_train": statistics.median(trains0),
            "step0_validation": statistics.median(valids0),
            "final_train": statistics.median(trains1),
            "final_validation": statistics.median(valids1),
        }
    return found


def replay_medians():
    found = {}
    for profile in ("features12_steps2048", "features12_steps8192"):
        values = []
        for path in sorted((LOSS_ROOT / profile).glob("*/receipt.json")):
            values.append(float(json.loads(path.read_text())["elapsed_seconds"]))
        found[profile] = statistics.median(values)
    return found


def collect_evidence():
    if not WATCH_LOG.is_file():
        raise FileNotFoundError(WATCH_LOG)
    return {
        "watch": parse_watch(WATCH_LOG.read_text()),
        "forest": forest_fits(),
        "beyond": beyond_prepare(),
        "loss": loss_medians(),
        "replay": replay_medians(),
    }


def _elapsed(value):
    number = float(value)
    if abs(number) >= 100:
        return f"{int(round(number)):,}".replace(",", "{,}")
    if abs(number) >= 10:
        return f"{number:.1f}"
    if abs(number) >= 1:
        return f"{number:.2f}"
    return f"{number:.3g}"


def _sig3(value):
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.3g}"


def _bytes(value):
    return f"{int(round(value)):,}".replace(",", "{,}")


def macro_bodies(text):
    bodies = {}
    needle = "\\newcommand{\\"
    start = 0
    while True:
        at = text.find(needle, start)
        if at < 0:
            break
        name_end = text.find("}", at + len(needle))
        name = text[at + len(needle) : name_end]
        body_start = text.find("{", name_end)
        depth = 0
        for index in range(body_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    bodies[name] = text[body_start + 1 : index]
                    start = index + 1
                    break
        else:
            break
    return bodies


def _require(bodies, name, expected, failures):
    found = bodies.get(name)
    if found != expected:
        failures.append(f"{name}: manuscript {found!r}, evidence {expected!r}")


def check_macros(evidence, numbers_text):
    failures = []
    bodies = macro_bodies(numbers_text)
    forest = evidence["forest"]
    watch = evidence["watch"]
    beyond = evidence["beyond"]
    loss = evidence["loss"]["features12_steps2048"]
    loss_b = evidence["loss"]["features12_steps8192"]
    _require(bodies, "ForestFitCore", _elapsed(forest["fit_seconds_sum"]), failures)
    _require(bodies, "ForestSampleOp", _elapsed(forest["operation_seconds_sum"]), failures)
    _require(bodies, "ForestGpuFits", str(forest["fit_count"]), failures)
    _require(bodies, "ForestByteLo", _bytes(forest["selected_byte_min"]), failures)
    _require(bodies, "ForestByteHi", _bytes(forest["selected_byte_max"]), failures)
    _require(bodies, "ArfWatchClosed", str(watch["closed"]), failures)
    _require(bodies, "ArfWatchOk", str(watch["ok"]), failures)
    _require(bodies, "ArfWatchPlanned", str(watch["planned"]), failures)
    _require(bodies, "BeyondPrepared", str(beyond["prepared"]), failures)
    _require(bodies, "BeyondFailed", str(beyond["failed"]), failures)
    _require(bodies, "BeyondFamilies", str(beyond["families"]), failures)
    _require(bodies, "BeyondFeatureRange", str(beyond["feature_range"]), failures)
    _require(bodies, "BeyondOverlap", str(beyond["overlap"]), failures)
    _require(bodies, "BeyondInventory", str(beyond["inventory"]), failures)
    _require(bodies, "BeyondRevision", beyond["revision"], failures)
    _require(bodies, "ReplayMedA", _elapsed(evidence["replay"]["features12_steps2048"]), failures)
    _require(bodies, "ReplayMedB", _elapsed(evidence["replay"]["features12_steps8192"]), failures)
    _require(bodies, "LossTrainStart", _sig3(loss["step0_train"]), failures)
    _require(bodies, "LossValStart", _sig3(loss["step0_validation"]), failures)
    _require(bodies, "LossTrainEndA", _sig3(loss["final_train"]), failures)
    _require(bodies, "LossValEndA", _sig3(loss["final_validation"]), failures)
    _require(bodies, "LossTrainEndB", _sig3(loss_b["final_train"]), failures)
    _require(bodies, "LossValEndB", _sig3(loss_b["final_validation"]), failures)
    if "ForestEvalSec" in bodies:
        failures.append("ForestEvalSec is not a sum of fit.json fields")
    if beyond["official_test_opened_by_optimizer"]:
        failures.append("BeyondArena prepare summary opened an official test")
    if watch["new_generator_fits_started"] != 0:
        failures.append("ARF watch log started new generator fits")
    return failures


def illustration_value():
    components = (0.90, 0.80, 0.85, 0.70, 0.75, 0.95)
    weights = (0.30, 0.20, 0.20, 0.15, 0.10, 0.05)
    epsilon = 1e-6
    total = sum(weights)
    log_score = sum(weight * math.log(epsilon + component) for weight, component in zip(weights, components))
    return 100.0 * math.exp(log_score / total)


def _must_contain(path, snippet, failures):
    text = path.read_text()
    if snippet not in text:
        failures.append(f"{path.name} does not contain {snippet!r}")


def protocol_tokens():
    """Numbers the manuscript may spell because a source file defines them."""
    failures = []
    root = REPO
    _must_contain(root / "rust/compiler/target_fitting/part_02.rs", '"features12_steps2048" => (12, 16, 2048)', failures)
    _must_contain(root / "rust/compiler/target_fitting/part_02.rs", ".build(&store, 2e-3)", failures)
    _must_contain(root / "rust/compiler/target_fitting/part_02.rs", "backward_step_clip(&loss, 5.0)", failures)
    _must_contain(root / "rust/compiler/target_fitting/part_01.rs", "coefficient.abs() >= 1e-5", failures)
    _must_contain(root / "rust/fitness.rs", "utility_transfer: 0.30", failures)
    _must_contain(root / "rust/fitness.rs", "Some(100.0 * (log_score / total_weight).exp())", failures)
    _must_contain(root / "rust/embedding/part_01.rs", "ACTION_EMBEDDING_DIMENSION: usize = 4_168", failures)
    _must_contain(root / "rust/embedding/part_01.rs", "ACTION_EMBEDDING_CANDIDATES: usize = 24", failures)
    _must_contain(root / "rust/embedding/part_01.rs", "ACTION_EMBEDDING_SKETCH_WIDTH: usize = 856", failures)
    _must_contain(root / "rust/embedding/part_01.rs", "ACTION_EMBEDDING_HIDDEN_WIDTH: usize = 128", failures)
    _must_contain(root / "rust/router/part_01.rs", "ROUTER_OUTPUTS: usize = 10", failures)
    _must_contain(root / "rust/compiler/neural_candidates/part_02.rs", "0.45 * utility + 0.25 * driver_agreement + 0.30 * proxy_joint_fidelity", failures)
    _must_contain(root / "research/benchmark/methods.lock.json", '"dependency_version": "copulas==0.14.1"', failures)
    _must_contain(root / "research/benchmark/tests/test_chow_liu.py", '"bins": 8, "laplace_alpha": 1.0', failures)
    _must_contain(root / "docs/whitepaper/scripts/compute_panel.py", "Q_NEMENYI_K4 = 2.569", failures)
    _must_contain(root / "docs/whitepaper/scripts/compute_panel.py", "DRAWS = 10_000", failures)
    _must_contain(root / "research/benchmark/publish_dope_population_validation.py", "SEEDS = (101,211,307)", failures)
    _must_contain(root / "docs/whitepaper/scripts/render_figures.py", "Y_LIM = (-1.5, 1.6)", failures)
    _must_contain(root / "docs/whitepaper/scripts/render_figures.py", "axis.set_xlim(-1.5, 1.5)", failures)
    _must_contain(root / "docs/whitepaper/scripts/compute_panel.py", "np.quantile(samples, [0.025, 0.975])", failures)
    _must_contain(root / "rust/fitness.rs", "(0.0..=1.0)", failures)
    _must_contain(root / "research/benchmark/results/s3-data.lock.json", "official_test_grouped_training_80_20", failures)
    if failures:
        raise ValueError("; ".join(failures))
    tokens = {
        "1", "2", "3", "4", "5", "6", "8", "10", "12", "16", "24", "80", "100", "128", "856",
        "2048", "4168", "8192", "10000", "10240",
        "0.002", "0.01", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.45",
        "0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.0", "1.5", "1.6", "2.569",
        "82.40", "0.14", "101", "211", "307", "11", "95", "0", "20",
    }
    rounded = f"{illustration_value():.2f}"
    if rounded != "82.40":
        raise ValueError(f"illustration score rounded to {rounded}")
    return tokens


def normalize_token(token):
    return token.replace("{,}", "").replace(",", "")


def untraced_numbers(tex, allowed):
    """Decimal and integer tokens in the manuscript body that no source owns."""
    missing = []
    for line in tex.splitlines():
        if "%" in line:
            line = line.split("%", 1)[0]
        line = CITE.sub(" ", line)
        if "\\input{generated/" in line or "\\setlength" in line or "\\bibliography" in line:
            continue
        for token in TOKEN.findall(line):
            normalized = normalize_token(token)
            if normalized not in allowed:
                missing.append(normalized)
    return missing


def write_evidence(evidence):
    GENERATED.mkdir(parents=True, exist_ok=True)
    payload = {
        "beyond": evidence["beyond"],
        "forest": evidence["forest"],
        "loss_medians": evidence["loss"],
        "replay_median_seconds": evidence["replay"],
        "watch": evidence["watch"],
    }
    (GENERATED / "fit-trace.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-evidence", action="store_true")
    args = parser.parse_args(argv)
    evidence = collect_evidence()
    if args.write_evidence:
        write_evidence(evidence)
        print("wrote generated/fit-trace.json")
        return 0
    trace_path = GENERATED / "fit-trace.json"
    failures = []
    if not trace_path.is_file():
        failures.append("generated/fit-trace.json is missing; run with --write-evidence")
    else:
        stored = json.loads(trace_path.read_text())
        fresh = {
            "beyond": evidence["beyond"],
            "forest": evidence["forest"],
            "loss_medians": evidence["loss"],
            "replay_median_seconds": evidence["replay"],
            "watch": evidence["watch"],
        }
        if stored != fresh:
            failures.append("fit-trace.json disagrees with a fresh read of fit.json, receipts, and the watch log")
    numbers = (GENERATED / "numbers.tex").read_text() if (GENERATED / "numbers.tex").is_file() else ""
    if not numbers:
        failures.append("generated/numbers.tex is missing")
    else:
        failures.extend(check_macros(evidence, numbers))
    allowed = protocol_tokens()
    missing = untraced_numbers(TEX.read_text(), allowed)
    if missing:
        failures.append("untraced numbers in dope-mfs.tex: " + ", ".join(missing[:30]))
    if failures:
        print("\n".join(failures))
        return 1
    print("paper numbers match fit.json, receipts, and the ARF watch log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
