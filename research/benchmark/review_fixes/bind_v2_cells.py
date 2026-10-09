"""Bind predeclare_v2 metric cells into one public scalar ledger.

Inputs are raw cell JSON files from the v2 runners (fit_seeds_v2 metric
records and rescore_ledger v2 cells). Every bound cell must carry the pinned
evaluator digest and library versions from predeclare_v2, a closed official
test flag, a known method, and a unique identity. Paths and private fields are
dropped. The ledger is canonical JSON lines, sorted by identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREDECLARE = HERE / "predeclare_v2.json"
AUDITORS = ("catboost", "linear", "mlp")
METHODS = {
    "DOPE", "ARF", "TabSyn", "TabDDPM", "ForestDiffusion/Forest-Flow", "GaussianCopula", "Chow-Liu",
    "independent_marginals", "CTGAN", "TVAE", "TabDiff", "GReaT", "synthpop_CART", "SMOTE",
    "predictor_only", "real_bootstrap_4n",
}
UTILITY_FIELDS = ("trtr_loss", "tstr_loss", "informative", "retention")


def _contract() -> dict:
    document = json.loads(PREDECLARE.read_text())
    evaluator = document["evaluator"]
    return {"sha256": hashlib.sha256(PREDECLARE.read_bytes()).hexdigest(),
            "implementation_sha256": evaluator["utility_sha256"],
            "dependencies": evaluator["dependencies"]}


def _finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("utility value is not finite")
    return float(value)


def public_cell(raw: dict, contract: dict, source: str) -> dict:
    """Reduce one raw record to its public scalar form, refusing anything off contract."""
    if raw.get("official_tests_opened") is not False:
        raise ValueError("cell official test flag is not closed")
    if raw.get("status") != "ok":
        raise ValueError("only ok cells are bound")
    method = raw.get("method")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    configuration = raw.get("configuration") or raw.get("arm")
    if not isinstance(configuration, str) or not configuration:
        raise ValueError("cell configuration is missing")
    dataset = raw.get("dataset")
    if not isinstance(dataset, str) or len(dataset) != 16 or any(ch not in "0123456789abcdef" for ch in dataset):
        raise ValueError("cell dataset is not a 16-hex id")
    report = raw.get("report")
    if not isinstance(report, dict):
        raise ValueError("cell has no evaluator report")
    if report.get("implementation_sha256") != contract["implementation_sha256"]:
        raise ValueError("cell evaluator digest differs from predeclare_v2")
    if report.get("dependencies") != contract["dependencies"]:
        raise ValueError("cell library versions differ from predeclare_v2")
    if raw.get("size") not in (1, 4) or raw.get("sample_seed") not in (101, 211, 307):
        raise ValueError("cell is outside the declared sample grid")
    fit_seed = raw.get("fit_seed")
    if fit_seed is not None and (isinstance(fit_seed, bool) or not isinstance(fit_seed, int)):
        raise ValueError("fit seed must be an integer or null")
    utility = {}
    for auditor in AUDITORS:
        row = (report.get("utility") or {}).get(auditor) or {}
        if "status" in row:
            utility[auditor] = {"status": "failed"}
            continue
        utility[auditor] = {
            "trtr_loss": _finite_or_none(row.get("trtr_loss")),
            "tstr_loss": _finite_or_none(row.get("tstr_loss")),
            "informative": bool(row.get("informative")),
            "retention": _finite_or_none(row.get("retention")),
        }
    digest = raw.get("synthetic_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("cell synthetic digest is missing")
    return {
        "method": method, "configuration": configuration, "dataset": dataset,
        "fit_seed": fit_seed, "sample_seed": raw["sample_seed"], "size": raw["size"],
        "null_loss": _finite_or_none(report.get("null_loss")), "utility": utility,
        "marginal_ks_mean": _finite_or_none(report.get("marginal_ks_mean")),
        "pair_correlation_fidelity": _finite_or_none(report.get("pair_correlation_fidelity")),
        "c2st_auc_logistic": _finite_or_none(report.get("c2st_auc")),
        "synthetic_sha256": digest, "implementation_sha256": report["implementation_sha256"],
        "binary_sha256": raw.get("binary_sha256"), "source": source,
        "official_tests_opened": False,
    }


def identity(cell: dict) -> tuple:
    return (cell["method"], cell["configuration"], cell["dataset"],
            -1 if cell["fit_seed"] is None else cell["fit_seed"], cell["sample_seed"], cell["size"])


def collect(roots: list[tuple[Path, str, str | None]], contract: dict) -> list[dict]:
    """Bind every ok cell under each root; an override names the configuration of a root."""
    cells: dict[tuple, dict] = {}
    for root, source, override in roots:
        for path in sorted(root.rglob("*.json")):
            if not (path.name.endswith(".metric.json") or path.name.endswith(".cell.json")):
                continue
            if path.is_symlink() or path.name == "test.csv" or "evaluator" in path.parts:
                raise ValueError(f"refusing {path}")
            if any(item.suffix == ".tmp" for item in path.parent.iterdir()):
                raise ValueError(f"unfinished write beside {path}")
            raw = json.loads(path.read_text())
            if raw.get("status") != "ok":
                continue
            if override is not None:
                raw = {**raw, "configuration": override}
            cell = public_cell(raw, contract, source)
            key = identity(cell)
            if key in cells:
                raise ValueError(f"duplicate cell identity {key}")
            cells[key] = cell
    return [cells[key] for key in sorted(cells)]


def public_fit(raw: dict, model: Path, source: str) -> dict:
    """One DOPE fit record: status and the charged bytes, model file plus projection."""
    if raw.get("official_tests_opened") is not False:
        raise ValueError("fit official test flag is not closed")
    if raw.get("method", "DOPE") != "DOPE":
        raise ValueError("only DOPE fit records are bound")
    artifact = raw.get("artifact_bytes")
    projection = raw.get("projection_bytes")
    charged = None
    if raw.get("status") == "ok":
        if not model.is_file() or model.stat().st_size != artifact:
            raise ValueError(f"model bytes differ from the fit record at {model}")
        if type(projection) is not int or projection <= 0:
            raise ValueError("projection bytes are missing")
        charged = artifact + projection
    return {"method": "DOPE", "configuration": raw.get("configuration") or raw.get("arm"),
            "dataset": raw["dataset"], "fit_seed": raw["fit_seed"], "status": raw.get("status"),
            "reason": raw.get("reason"), "artifact_bytes": artifact, "projection_bytes": projection,
            "charged_bytes": charged, "elapsed_seconds": raw.get("elapsed_seconds"),
            "model_sha256": raw.get("model_sha256"), "binary_sha256": raw.get("binary_sha256"),
            "gpu": raw.get("gpu"), "source": source, "official_tests_opened": False}


def collect_fits(roots: list[tuple[Path, str, str | None]]) -> list[dict]:
    fits: dict[tuple, dict] = {}
    for root, source, override in roots:
        for path in sorted(root.rglob("fit.json")):
            if path.is_symlink():
                raise ValueError(f"refusing {path}")
            raw = json.loads(path.read_text())
            if override is not None:
                raw = {**raw, "configuration": override}
            record = public_fit(raw, path.parent / "model.dpk", source)
            key = (record["configuration"], record["dataset"], record["fit_seed"])
            if key in fits:
                raise ValueError(f"duplicate fit identity {key}")
            fits[key] = record
    return [fits[key] for key in sorted(fits)]


def write_ledger(cells: list[dict], out: Path, contract: dict) -> dict:
    lines = "".join(json.dumps(cell, sort_keys=True, separators=(",", ":")) + "\n" for cell in cells)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(lines)
    meta = {"format": "dope-review-fix-v2-scalar-ledger", "cells": len(cells),
            "sha256": hashlib.sha256(lines.encode()).hexdigest(),
            "predeclare_sha256": contract["sha256"], "official_tests_opened": False}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[], help="PATH=SOURCE[=CONFIGURATION], repeatable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = []
    for item in args.root:
        parts = item.split("=")
        if len(parts) not in (2, 3) or not parts[1]:
            raise SystemExit("--root takes PATH=SOURCE or PATH=SOURCE=CONFIGURATION")
        roots.append((Path(parts[0]), parts[1], parts[2] if len(parts) == 3 else None))
    contract = _contract()
    meta = write_ledger(collect(roots, contract), args.out, contract)
    fits = collect_fits(roots)
    fit_lines = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in fits)
    fits_path = args.out.with_name("fits.jsonl")
    fits_path.write_text(fit_lines)
    meta["fits"] = len(fits)
    meta["fits_sha256"] = hashlib.sha256(fit_lines.encode()).hexdigest()
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
