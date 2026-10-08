"""Write the mapped density cohort onto scratch. No foundation-model forward."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from research.benchmark.mfs_v3_panel import (
    apply_map,
    build_nulls,
    cleartext_absent,
    cohort_document,
    column_name_hashes,
    exact_row_matches,
    fit_map,
    load_numeric_table,
    near_copy_ok,
    null_seed,
    public_salt_id,
    refuse_test_path,
    pilot_cells,
    verify_cell_hash,
)

ROOT = Path(__file__).resolve().parents[2]
DENSITY = ROOT / "research/benchmark/results/density-matched-population-validation.json"
DEFAULT_OUT = Path("/mnt/fast-scratch/dope-benchmark/mfs-v3-panel-v1")


def _salt(directory: Path) -> bytes:
    path = directory / "salt.bin"
    if path.is_file():
        return path.read_bytes()
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    salt = os.urandom(32)
    path.write_bytes(salt)
    path.chmod(0o600)
    return salt


def prepare_cell(cell: dict, out_dir: Path, salt: bytes) -> dict:
    sample = Path(cell["sample_csv"])
    worker = Path(cell["worker_dir"])
    train = worker / "train.csv"
    validation = worker / "validation.csv"
    for path in (sample, train, validation):
        refuse_test_path(path)
    if not verify_cell_hash(cell):
        raise ValueError(f"sample hash mismatch for {cell['dataset']} {cell['method']}")
    fit = load_numeric_table(train)
    holdout = load_numeric_table(validation)
    synthetic = load_numeric_table(sample)
    if fit.shape[1] != holdout.shape[1] or fit.shape[1] != synthetic.shape[1]:
        raise ValueError("fit, holdout, and synthetic widths differ")
    fitted = fit_map(fit)
    mapped_fit = apply_map(fitted, fit)
    mapped_holdout = apply_map(fitted, holdout)
    mapped_synthetic = apply_map(fitted, synthetic)
    seeds = tuple(
        null_seed(cell["dataset"], cell["method"], cell["configuration"], cell["sample_seed"], index)
        for index in range(3)
    )
    nulls, losses, null_index, match_index = build_nulls(
        mapped_fit, mapped_holdout, mapped_synthetic.shape[0], seeds
    )
    projection = worker / "projection.json"
    artifact = None
    receipt = cell.get("fit_receipt_path")
    if receipt:
        candidate = Path(receipt).parent / "model.dpk"
        if candidate.is_file():
            artifact = candidate
    names = column_name_hashes(projection, salt)
    scanned = cleartext_absent(artifact, projection) if artifact is not None else None
    stem = out_dir / "mapped" / cell["method"] / cell["dataset"] / f"seed{cell['sample_seed']}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        fit=mapped_fit,
        holdout=mapped_holdout,
        synthetic=mapped_synthetic,
        null0=nulls[0],
        null1=nulls[1],
        null2=nulls[2],
    )
    record = {
        "artifact_bytes": cell["artifact_bytes"],
        "block": cell["block"],
        "cleartext_absent": scanned,
        "column_hashes": names,
        "configuration": cell["configuration"],
        "counts_as_dope_win": False,
        "dataset": cell["dataset"],
        "exact_row_matches": exact_row_matches(mapped_fit, mapped_synthetic),
        "fit_seed": cell["fit_seed"],
        "linear_null_losses": losses,
        "mapped": str(stem.with_suffix(".npz")),
        "match_index": match_index,
        "method": cell["method"],
        "near_copy_ok": near_copy_ok(mapped_fit, mapped_synthetic),
        "null_index": null_index,
        "official_tests_opened": False,
        "salt_id": public_salt_id(salt),
        "sample_seed": cell["sample_seed"],
        "sample_sha256": cell["sample_sha256"],
        "size_multiplier": cell["size_multiplier"],
        "superiority": None,
    }
    stem.with_suffix(".json").write_text(json.dumps(record, sort_keys=True) + "\n")
    return record


def main() -> None:
    out_dir = Path(os.environ.get("MFS_V3_OUT", DEFAULT_OUT))
    limit = int(os.environ.get("MFS_V3_LIMIT", "0"))
    record = json.loads(DENSITY.read_text())
    document = cohort_document(record)
    cells = document["cells"]
    if os.environ.get("MFS_V3_PILOT") == "1":
        cells = pilot_cells(cells, document["lineages"])
    elif limit:
        cells = cells[:limit]
    salt = _salt(out_dir)
    prepared = []
    for index, cell in enumerate(cells, start=1):
        prepared.append(prepare_cell(cell, out_dir, salt))
        if index % 25 == 0:
            print(f"prepared {index}/{len(cells)}", flush=True)
    document["cells"] = prepared
    document["salt_id"] = public_salt_id(salt)
    target = out_dir / "cohort.json"
    target.write_text(json.dumps(document) + "\n")
    print(f"lineages {len(document['lineages'])} cells {len(prepared)} -> {target}", flush=True)


if __name__ == "__main__":
    main()
