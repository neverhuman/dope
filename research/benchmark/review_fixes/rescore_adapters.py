"""Ledger adapters for the K0 common-evaluator re-score (predeclare_v2 block K).

Each adapter reads one committed ledger under research/benchmark/results and
returns normalized jobs. A job names one logical metric cell, the stored
synthetic CSV, the sample sha256 the ledger recorded, and the stored retention
per auditor. A ledger cell without a recorded sample becomes a blocked job. It
is recorded, never imputed. Adapters read ledgers and metric receipts only.
They refuse any path named test.csv or under an evaluator directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
AUDITORS = ("catboost", "linear", "mlp")
CONTROLS = Path("/mnt/fast-scratch/dope-benchmark/review-fixes-lane-b/controls")
TABSYN = Path("/home/ubuntu/dope-scratch-x1/dope-rf-tabsyn-sample-v1")
DENSITY_METHODS = ("GaussianCopula", "Chow-Liu", "independent_marginals")
SDV_METHODS = ("CTGAN", "TVAE")
FOREST = "ForestDiffusion/Forest-Flow"
HEADLINE_PROFILE = "features12_steps2048"
# The frozen TabSyn fit registry admits fit seed 11 only (tabsyn_sample_evidence.py).
TABSYN_FIT_SEED = 11
LOCATIONS = (
    ("/mnt/fast-scratch/", "fast-scratch"),
    ("/home/ubuntu/dope-scratch-x1/", "xbabe1"),
    ("/home/ubuntu/dope-scratch-x2/", "xbabe2"),
    ("/home/ubuntu/dope-scratch-x3/", "xbabe3"),
    ("/home/ubuntu/dope/", "worktree"),
)
HOST_HOMES = ("xbabe1", "xbabe2", "xbabe3")
# The auditor seed behind each ledger's stored values. container_validation_worker.py and
# sdv_frozen_sample_validation.py passed the sample seed (101/211/307), so their stored
# values differ from a seed-1729 replay by construction; at the sample seed they replay
# bit for bit on xbabe2. ARF, TabSyn and the controls used 1729.
STORED_AUDITOR_SEED = {"density": "sample_seed", "sdv": "sample_seed", "forest": "sample_seed",
                       "dope-historical": "sample_seed", "arf": 1729, "tabsyn": 1729, "controls": 1729}


class Refused(ValueError):
    """A custody refusal with a stable reason code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class RemoteOnly(SystemExit):
    """A ledger whose resolution needs another host's private disk."""


def refuse_path(path: Path | str) -> None:
    parts = Path(path).parts
    if "test.csv" in parts:
        raise Refused("refused_test_path")
    if "evaluator" in parts:
        raise Refused("refused_evaluator_path")


def refuse_symlink(path: Path) -> None:
    """Refuse a symlink anywhere on the path. Only lstat calls; nothing is opened."""
    if path.is_symlink() or os.path.realpath(path) != os.path.abspath(path):
        raise Refused("refused_symlink")


def location(path: str) -> str:
    for prefix, label in LOCATIONS:
        if path.startswith(prefix):
            return label
    return "other"


def remote_elsewhere(path: str, host: str) -> bool:
    where = location(path)
    return where in HOST_HOMES and where != host


def read_json(path: Path, expected_sha256: str | None = None):
    refuse_path(path)
    refuse_symlink(path)
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise Refused("receipt_hash_mismatch")
    return json.loads(raw)


def read_jsonl(path: Path) -> list[dict]:
    refuse_path(path)
    refuse_symlink(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def retention(utility) -> dict | None:
    if not isinstance(utility, dict):
        return None
    return {name: (utility.get(name) or {}).get("retention") for name in AUDITORS}


def job(ledger: str, method: str, configuration: str, dataset: str, fit_seed, sample_seed, size, *,
        csv=None, sha=None, stored=None, has_header=False, blocked=None, route="ledger",
        split_seed=None, train=None, validation=None, inputs=None) -> dict:
    if len(dataset) != 16 or any(ch not in "0123456789abcdef" for ch in dataset):
        raise ValueError("dataset is not a 16-hex lineage id")
    if int(size) not in (1, 4):
        raise ValueError("size is not 1n or 4n")
    return {"ledger": ledger, "method": method, "configuration": configuration, "dataset": dataset,
            "fit_seed": None if fit_seed is None else int(fit_seed), "sample_seed": int(sample_seed),
            "size": int(size), "split_seed": split_seed, "csv_path": None if csv is None else str(csv),
            "csv_sha256": sha, "stored_retention": stored, "has_header": has_header,
            "target_position": "last", "train_path": train, "validation_path": validation,
            "expected_inputs": inputs, "blocked": blocked, "route": route}


def _ledger_gap(cell: dict) -> list[str]:
    reason = f"ledger_{cell.get('status')}"
    if cell.get("unavailable_reason"):
        reason += f":{cell['unavailable_reason']}"
    return ["sample_absent", reason]


def _beside(metric_path: str) -> Path:
    path = Path(metric_path)
    if not path.name.endswith(".metric.json"):
        raise ValueError("metric receipt name is not <sample>.metric.json")
    return path.with_name(path.name[: -len(".metric.json")] + ".csv")


def _same_cell(evidence: dict, cell: dict) -> None:
    if (evidence.get("sample_seed"), evidence.get("size_multiplier")) != (cell["sample_seed"], cell["size_multiplier"]):
        raise ValueError("sample evidence names a different sample cell")


def density(path: Path, roots: dict) -> list[dict]:
    jobs = []
    for cell in read_json(path)["cells"]:
        if cell["method"] not in DENSITY_METHODS:
            continue  # The DOPE rows repeat the historical population (dope-historical).
        common = ("density", cell["method"], cell["configuration"], cell["dataset"], cell["fit_seed"],
                  cell["sample_seed"], cell["size_multiplier"])
        evidence = cell.get("sample_evidence")
        if cell["status"] != "ok" or not evidence:
            jobs.append(job(*common, blocked=_ledger_gap(cell)))
            continue
        _same_cell(evidence, cell)
        jobs.append(job(*common, csv=_beside(evidence["metric_path"]), sha=evidence["sample_sha256"],
                        stored=retention(cell.get("utility"))))
    return jobs


def sdv(path: Path, roots: dict) -> list[dict]:
    jobs = []
    for cell in read_json(path)["cells"]:
        if cell["method"] not in SDV_METHODS:
            continue
        configuration = cell.get("configuration_name") or cell["configuration"]
        common = ("sdv", cell["method"], configuration, cell["dataset"], cell["fit_seed"],
                  cell["sample_seed"], cell["size_multiplier"])
        evidence = cell.get("sample_evidence")
        if cell["status"] != "ok" or not evidence:
            jobs.append(job(*common, blocked=_ledger_gap(cell)))
            continue
        jobs.append(job(*common, csv=evidence["path"], sha=evidence["sha256"],
                        stored=retention(cell.get("utility"))))
    return jobs


def forest(path: Path, roots: dict) -> list[dict]:
    jobs = []
    for cell in read_json(path)["cells"]:
        if cell["method"] != FOREST:
            continue  # Its DOPE, CTGAN and TVAE rows belong to other ledgers.
        common = ("forest", FOREST, cell["configuration"], cell["dataset"], cell["fit_seed"],
                  cell["sample_seed"], cell["size_multiplier"])
        stored = retention(cell.get("utility"))
        ref = cell.get("receipt")
        if cell["status"] != "ok" or not ref:
            jobs.append(job(*common, blocked=_ledger_gap(cell)))
            continue
        try:
            spec = read_json(Path(ref["path"]), ref["sha256"])["job"]
        except FileNotFoundError:
            jobs.append(job(*common, stored=stored, blocked=["sample_absent", "receipt_absent"]))
            continue
        except Refused as error:
            jobs.append(job(*common, stored=stored, blocked=["sample_hash_mismatch", error.code]))
            continue
        identity = (spec["method"], spec["dataset"], spec["fit_seed"], spec["sample_seed"], spec["size_multiplier"])
        if identity != (FOREST, *common[3:]):
            raise ValueError("forest receipt names a different cell")
        files = spec["worker"]["files"]
        jobs.append(job(*common, csv=spec["sample_path"], sha=spec["sample_sha256"], stored=stored,
                        route="receipt", inputs={"train": files["train.csv"], "validation": files["validation.csv"]}))
    return jobs


def arf(path: Path, roots: dict) -> list[dict]:
    jobs, parents = [], {}
    for row in read_json(path)["rows"]:
        if row["method"] != "ARF":
            continue  # The 95 CPU TabSyn rows are superseded by review-fixes-tabsyn-v1.
        common = ("arf", "ARF", row["selection_binding"], row["dataset"], row["fit_seed"],
                  row["sample_seed"], row["row_multiplier"])
        hashes = row["input_hashes"]
        extra = {"sha": hashes["synthetic"], "stored": {name: row.get(f"{name}_retention") for name in AUDITORS},
                 "inputs": {"train": hashes["train"], "validation": hashes["validation"]}}
        ref = row["metric_receipt_ref"]
        try:
            receipt = read_json(Path(ref["path"]), ref["sha256"])
        except FileNotFoundError:
            if remote_elsewhere(ref["path"], roots["host"]):
                raise RemoteOnly(f"arf receipts on {location(ref['path'])} are not readable here; "
                                 "resolve on that host with --dry-run and pass --plan-dir") from None
            jobs.append(job(*common, **extra, blocked=["sample_absent", "receipt_absent"]))
            continue
        except Refused as error:
            jobs.append(job(*common, **extra, blocked=["sample_hash_mismatch", error.code]))
            continue
        synthetic = receipt["input_refs"]["synthetic_ref"]
        if synthetic["sha256"] != hashes["synthetic"] or synthetic.get("csv_format") != "headerless_numeric":
            jobs.append(job(*common, **extra, blocked=["sample_hash_mismatch", "receipt_synthetic_mismatch"]))
            continue
        parents[common[2:5]] = Path(synthetic["path"]).parent
        jobs.append(job(*common, **extra, csv=synthetic["path"], route="receipt"))
    for item in jobs:
        # A lost receipt keeps its sibling samples; the ledger hash still has to match.
        parent = parents.get((item["configuration"], item["dataset"], item["fit_seed"]))
        if item["blocked"] == ["sample_absent", "receipt_absent"] and parent is not None:
            item.update(csv_path=str(parent / f"n{item['size']}-seed{item['sample_seed']}.csv"),
                        route="sibling", blocked=None)
    return jobs


def tabsyn(path: Path, roots: dict) -> list[dict]:
    configuration = read_json(path.parent / "panel.json")["configuration"]
    jobs = []
    for row in read_jsonl(path):
        common = ("tabsyn", "TabSyn", configuration, row["dataset"], TABSYN_FIT_SEED, row["sample_seed"], row["size"])
        if row["status"] != "ok" or not row.get("synthetic_sha256"):
            jobs.append(job(*common, blocked=_ledger_gap(row)))
            continue
        csv = Path(roots["tabsyn"]) / row["dataset"] / f"sample-{row['sample_seed']}-{row['size']}n.csv"
        jobs.append(job(*common, csv=csv, sha=row["synthetic_sha256"], has_header=True,
                        stored=retention(row.get("utility"))))
    return jobs


def controls(path: Path, roots: dict) -> list[dict]:
    jobs = []
    for row in read_jsonl(path):
        kind, seed, size = row["kind"], row["sample_seed"], row["size"]
        folder = Path(roots["controls"]) / row["dataset"]
        extra = {}
        if kind == "predictor_only":
            stem = f"predictor-fit{row['fit_seed']}-sample{seed}-size{size}"
            method, configuration, fit_seed = "predictor_only", "control", row["fit_seed"]
        elif kind == "split":
            stem = f"split{row['split_seed']}-sample{seed}-size{size}"
            method, configuration, fit_seed = "predictor_only", f"control_split{row['split_seed']}", row["fit_seed"]
            extra = {"split_seed": row["split_seed"], "train": str(folder / f"{stem}.train.csv"),
                     "validation": str(folder / f"{stem}.validation.csv")}
        elif kind == "real_bootstrap_4n":
            stem = f"real-bootstrap-sample{seed}"
            method, configuration, fit_seed = "real_bootstrap_4n", "control", None
        else:
            raise ValueError("unknown control kind")
        common = ("controls", method, configuration, row["dataset"], fit_seed, seed, size)
        if row["status"] != "ok" or not row.get("synthetic_sha256"):
            jobs.append(job(*common, blocked=_ledger_gap(row), split_seed=extra.get("split_seed")))
            continue
        jobs.append(job(*common, csv=folder / f"{stem}.synthetic.csv", sha=row["synthetic_sha256"],
                        stored=retention((row.get("metrics") or {}).get("utility")), **extra))
    return jobs


def dope_historical(path: Path, roots: dict) -> list[dict]:
    jobs = []
    for cell in read_json(path)["cells"]:
        if cell["method"] != "DOPE" or cell["fit_seed"] != 11:
            raise ValueError("historical population row is not DOPE fit seed 11")
        profile = cell["profile"]
        configuration = "historical_seed11" if profile == HEADLINE_PROFILE else f"historical_seed11_{profile}"
        common = ("dope-historical", "DOPE", configuration, cell["dataset"], 11, cell["sample_seed"],
                  cell["size_multiplier"])
        receipt = cell.get("metric_receipt")
        if cell["status"] != "ok" or not receipt:
            jobs.append(job(*common, blocked=_ledger_gap(cell)))
            continue
        _same_cell(receipt, cell)
        jobs.append(job(*common, csv=_beside(receipt["metric_path"]), sha=receipt["sample_sha256"],
                        stored=retention(cell.get("utility"))))
    return jobs


LEDGERS = {
    "density": ("density-matched-population-validation.json", density),
    "sdv": ("sdv-matched-population-validation.json", sdv),
    "arf": ("arf-tabsyn-followon-validation-v1/panel.json", arf),
    "forest": ("s3-matched-forest-confirmation-validation.json", forest),
    "tabsyn": ("review-fixes-tabsyn-v1/scalar-cells.jsonl", tabsyn),
    "controls": ("review-fixes-controls-v1/scalar-cells.jsonl", controls),
    "dope-historical": ("dope-s3-population-validation.json", dope_historical),
}


def ledger_info(name: str, results: Path = RESULTS) -> dict:
    relative = LEDGERS[name][0]
    path = results / relative
    refuse_path(path)
    refuse_symlink(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"name": name, "file": relative, "path": str(path), "sha256": digest}


def resolve(name: str, host: str, results: Path = RESULTS, roots: dict | None = None) -> list[dict]:
    roots = {"controls": CONTROLS, "tabsyn": TABSYN, "host": host, **(roots or {})}
    return LEDGERS[name][1](results / LEDGERS[name][0], roots)
