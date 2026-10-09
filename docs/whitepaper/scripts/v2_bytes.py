"""Charged artifact bytes per v2 arm, as the files a sampler reads.

DOPE charges the encoded model plus projection.json (the v2 fit ledger).
TabSyn charges every saved file except intermediate diffusion epoch
checkpoints, which sampling never reads. Baselines charge the per-lineage
bytes stored in their committed ledgers. Each arm reports the median,
quartiles, and the share of lineages within the 10,240-byte cap.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "benchmark" / "results"
CAP = 10240


def _summary(per_lineage: dict[str, float], fits_ok: int | None = None, fits_within: int | None = None,
             fits_total: int | None = None) -> dict | None:
    values = np.asarray(sorted(per_lineage.values()), dtype=float)
    if values.size == 0:
        return None
    q1, median, q3 = (float(x) for x in np.quantile(values, [0.25, 0.5, 0.75]))
    return {"n": int(values.size), "median": median, "q1": q1, "q3": q3,
            "min": float(values.min()), "max": float(values.max()),
            "within_cap": int(np.sum(values <= CAP)),
            "fits_ok": fits_ok, "fits_within_cap": fits_within, "fits_total": fits_total}


def dope(fits: list[dict], configuration: str) -> dict | None:
    chosen = [item for item in fits if item["configuration"] == configuration]
    grouped = defaultdict(list)
    for item in chosen:
        if item["status"] == "ok" and isinstance(item["charged_bytes"], int):
            grouped[item["dataset"]].append(item["charged_bytes"])
    per_lineage = {key: float(np.median(vals)) for key, vals in grouped.items()}
    ok = [item for item in chosen if item["status"] == "ok"]
    within = sum(1 for item in ok if item["charged_bytes"] <= CAP)
    return _summary(per_lineage, len(ok), within, len(chosen))


def tabsyn(path: Path = RESULTS / "tabsyn-default100-scaled-cuda-fits.json") -> dict | None:
    document = json.loads(path.read_text())
    per_lineage = {}
    for cell in document["cells"]:
        files = cell.get("artifact_files") or {}
        if not files:
            continue
        per_lineage[cell["dataset_id"]] = float(sum(
            meta["bytes"] for name, meta in files.items() if not name.startswith("diffusion-epoch-")))
    return _summary(per_lineage)


def ledger_cells(name: str, method: str, configuration: str) -> dict | None:
    document = json.loads((RESULTS / name).read_text())
    per_lineage = {}
    for cell in document["cells"]:
        if cell["method"] != method or cell["configuration"] != configuration:
            continue
        value = cell.get("charged_artifact_bytes")
        if isinstance(value, int):
            if per_lineage.get(cell["dataset"], value) != value:
                raise ValueError(f"{method} has two byte charges for one lineage")
            per_lineage[cell["dataset"]] = value
    return _summary(per_lineage)


def arf(configuration: str, path: Path = RESULTS / "arf-s3-population-native.json") -> dict | None:
    field = {"author_default": "default_charged_bytes", "native_selected": "selected_charged_bytes"}[configuration]
    document = json.loads(path.read_text())
    per_lineage = {cell["dataset"]: cell[field] for cell in document["cells"] if isinstance(cell.get(field), int)}
    return _summary(per_lineage)


def all_arms(fits: list[dict]) -> dict[str, dict | None]:
    """Keyed by 'method|configuration'. Controls store no artifact and are absent."""
    density = "density-matched-population-validation.json"
    sdv = "sdv-matched-population-validation.json"
    forest = "s3-matched-forest-confirmation-validation.json"
    return {
        "DOPE|headline": dope(fits, "headline"),
        "DOPE|product_default": dope(fits, "product_default"),
        "TabSyn|scaled_200_vae_1000_diffusion": tabsyn(),
        "ARF|author_default": arf("author_default"),
        "ARF|native_selected": arf("native_selected"),
        "GaussianCopula|native_selected": ledger_cells(density, "GaussianCopula", "native_selected"),
        "Chow-Liu|native_selected": ledger_cells(density, "Chow-Liu", "native_selected"),
        "independent_marginals|native_selected": ledger_cells(density, "independent_marginals", "native_selected"),
        "CTGAN|native_selected": ledger_cells(sdv, "CTGAN", "native_selected"),
        "TVAE|native_selected": ledger_cells(sdv, "TVAE", "native_selected"),
        "ForestDiffusion/Forest-Flow|native_selected": ledger_cells(forest, "ForestDiffusion/Forest-Flow",
                                                                     "native_selected"),
    }
