#!/usr/bin/env python3
"""Trace manuscript numbers to committed receipts and ledgers.

Reads repo-relative files only. It does not open the ARF watch log, scratch
fit directories, or any path outside this checkout, including ``.agent/`` and
``target/``. ``--write-evidence`` stores the reduction in
docs/whitepaper/generated/fit-trace.json. A later run with no flag exits
nonzero when a macro, a hand-typed token, or that evidence file disagrees.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "research" / "benchmark" / "results"
PAPER = REPO / "docs" / "whitepaper"
GENERATED = PAPER / "generated"
TEX = PAPER / "dope-mfs.tex"
SUPPLEMENT = PAPER / "supplement.tex"
INVENTORY_TEX = GENERATED / "beyondarena-inventory.tex"
ARCH = PAPER / "figures" / "architecture.tex"
ARCH_LITERALS = {"80", "20", "16", "2048", "0.002"}
WATCH_RECEIPT = RESULTS / "arf-native-closure-watch-v1.receipt.json"
FOREST_LEDGER = RESULTS / "s3-matched-forest-confirmation-validation.json"
LOSS_CURVES = GENERATED / "loss-curves.json"
REPLAY_COST = GENERATED / "replay-cost.json"
PROVENANCE = GENERATED / "provenance.json"
INVENTORY = RESULTS / "beyondarena-s3-inventory.json"
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


def load_watch():
    """Reduced watch facts. The source path is recorded and not opened."""
    receipt = json.loads(WATCH_RECEIPT.read_text())
    digest = str(receipt.get("sha256") or "")
    source = receipt.get("source")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("watch receipt sha256 is not 64 hex characters")
    if not isinstance(source, str) or not source:
        raise ValueError("watch receipt has no source")
    watch = receipt["watch"]
    return {
        "closed": int(watch["closed"]),
        "ok": int(watch["ok"]),
        "planned": int(watch["planned"]),
        "count_utc": watch.get("count_utc"),
        "terminal_phase": watch.get("terminal_phase"),
        "terminal_exit_code": watch.get("terminal_exit_code"),
        "new_generator_fits_started": int(watch.get("new_generator_fits_started") or 0),
        "records": int(watch["records"]),
    }


def forest_fits():
    """Byte bounds and fit totals from the committed confirmation ledger."""
    ledger = json.loads(FOREST_LEDGER.read_text())
    selected = []
    for row in ledger["summary"]:
        if (
            row.get("method") == "ForestDiffusion/Forest-Flow"
            and row.get("configuration") == "native_selected"
            and row.get("size_multiplier") == 4
            and isinstance(row.get("charged_artifact_bytes"), int)
        ):
            selected.append(row["charged_artifact_bytes"])
    if not selected:
        raise ValueError("committed forest ledger has no native-selected size-4 byte charges")
    cost = ledger["cost"]
    return {
        "fit_count": int(cost["forest_new_gpu_fits"]),
        "fit_seconds_sum": float(cost["forest_fit_core_seconds"]),
        "operation_seconds_sum": float(cost["forest_fit_native_sample_operation_seconds"]),
        "selected_byte_min": min(selected),
        "selected_byte_max": max(selected),
        "selected_n": len(selected),
    }


def beyond_prepare():
    """Prepare counts stored in the committed provenance dump and inventory."""
    provenance = json.loads(PROVENANCE.read_text())
    summary = provenance["beyondarena"]["prepare_summary"]
    feature_range = 0
    overlap = 0
    for row in summary["failed_families"]:
        detail = str(row.get("detail") or "")
        if "projected feature count" in detail:
            feature_range += 1
        elif "overlapping source row" in detail:
            overlap += 1
        else:
            raise ValueError(f"unclassified prepare failure: {row.get('name')}")
    inventory = json.loads(INVENTORY.read_text())
    return {
        "families": int(summary["families"]),
        "prepared": int(summary["prepared"]),
        "failed": int(summary["failed"]),
        "feature_range": feature_range,
        "overlap": overlap,
        "inventory": int(inventory["dataset_count"]),
        "revision": str(inventory["revision"]),
        "official_test_opened_by_optimizer": bool(summary["official_test_opened_by_optimizer"]),
    }


def loss_medians():
    """Point medians already reduced in the committed loss-curve ledger."""
    document = json.loads(LOSS_CURVES.read_text())
    found = {}
    for profile in ("features12_steps2048", "features12_steps8192"):
        row = document["profiles"][profile]
        found[profile] = {
            "n": int(row["lineages"]),
            "step0_train": float(row["step0_train"]),
            "step0_validation": float(row["step0_validation"]),
            "final_train": float(row["final_train"]),
            "final_validation": float(row["final_validation"]),
        }
    return found


def replay_medians():
    document = json.loads(REPLAY_COST.read_text())
    return {
        profile: float(document[profile]["median"])
        for profile in ("features12_steps2048", "features12_steps8192")
    }


def collect_evidence():
    return {
        "watch": load_watch(),
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
    failures.extend(check_architecture(bodies))
    if "ForestEvalSec" in bodies:
        failures.append("ForestEvalSec is not a sum of fit.json fields")
    if beyond["official_test_opened_by_optimizer"]:
        failures.append("BeyondArena prepare summary opened an official test")
    if watch["new_generator_fits_started"] != 0:
        failures.append("ARF watch log started new generator fits")
    return failures


def architecture_expected():
    """Displayed pipeline constants from the fitter source and the data lock."""
    rust = (REPO / "rust/compiler/target_fitting/part_02.rs").read_text()
    shape = re.search(r'"features12_steps2048" => \(12, (\d+), (\d+)\)', rust)
    rate = re.search(r"\.build\(&store, ([0-9.eE+-]+)\)", rust)
    lock = (REPO / "research/benchmark/results/s3-data.lock.json").read_text()
    split = re.search(r"official_test_grouped_training_(\d+)_(\d+)", lock)
    if shape is None or rate is None or split is None:
        raise ValueError("architecture source constants are missing")
    steps = f"{int(shape.group(2)):,}".replace(",", "{,}")
    return {
        "ArchSplitMajor": split.group(1),
        "ArchSplitMinor": split.group(2),
        "ArchWidth": shape.group(1),
        "ArchSteps": steps,
        "ArchLr": f"{float(rate.group(1)):.3f}",
    }


def check_architecture(bodies):
    failures = []
    expected = architecture_expected()
    for name, value in expected.items():
        _require(bodies, name, value, failures)
    if not ARCH.is_file():
        failures.append("figures/architecture.tex is missing")
        return failures
    raw = []
    for line in ARCH.read_text().splitlines():
        if "%" in line:
            line = line.split("%", 1)[0]
        for token in TOKEN.findall(line):
            normalized = normalize_token(token)
            if normalized in ARCH_LITERALS:
                raw.append(normalized)
    if raw:
        failures.append("architecture.tex repeats generated literals: " + ", ".join(raw))
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


def supplement_for_scan(text):
    """Drop the hash-width token. It names an algorithm, not a measurement."""
    return text.replace("SHA-256", "SHA")


def check_inventory_caption():
    """The family count is the generated macro, not a second typed copy."""
    failures = []
    if not INVENTORY_TEX.is_file():
        return ["beyondarena-inventory.tex is missing"]
    captions = [line for line in INVENTORY_TEX.read_text().splitlines() if "\\caption" in line]
    if len(captions) != 1 or "\\BeyondInventory" not in captions[0]:
        failures.append("beyondarena inventory caption must use BeyondInventory")
        return failures
    visible = captions[0].split("%", 1)[0].replace("\\BeyondInventory", "")
    if TOKEN.search(visible):
        failures.append("beyondarena inventory caption types a number")
    return failures


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


def check_banned_tokens():
    """Fail when manuscript sources name a private machine or a local path."""
    scripts = PAPER / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from public_hardware import banned_hits

    failures = []
    paths = [TEX, SUPPLEMENT, *sorted(GENERATED.glob("*.tex"))]
    for path in paths:
        if not path.is_file():
            continue
        hits = banned_hits(path.read_text(errors="replace"))
        if hits:
            failures.append(f"{path.name} contains a private token: {hits[0]}")
    return failures


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
            failures.append("fit-trace.json disagrees with the committed receipts and ledgers")
    numbers = (GENERATED / "numbers.tex").read_text() if (GENERATED / "numbers.tex").is_file() else ""
    if not numbers:
        failures.append("generated/numbers.tex is missing")
    else:
        failures.extend(check_macros(evidence, numbers))
    allowed = protocol_tokens()
    missing = untraced_numbers(TEX.read_text(), allowed)
    if missing:
        failures.append("untraced numbers in dope-mfs.tex: " + ", ".join(missing[:30]))
    if SUPPLEMENT.is_file():
        supplement_missing = untraced_numbers(supplement_for_scan(SUPPLEMENT.read_text()), allowed)
        if supplement_missing:
            failures.append("untraced numbers in supplement.tex: " + ", ".join(supplement_missing[:30]))
    else:
        failures.append("supplement.tex is missing")
    failures.extend(check_inventory_caption())
    failures.extend(check_banned_tokens())
    if failures:
        print("\n".join(failures))
        return 1
    print("paper numbers match the committed receipts and ledgers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
