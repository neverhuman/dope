"""Pack mapped tables and score them with the Rust holdout function.

The pack stores numeric rows only. Official test files are refused.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

from research.benchmark.mfs_v3_panel import refuse_test_path

MAGIC = b"MFS3H1\n"


def _label(text: str) -> bytes:
    raw = text.encode()
    if not raw or len(raw) > 64:
        raise ValueError("holdout label length is outside the panel bound")
    return struct.pack("<I", len(raw)) + raw


def _matrix(table: np.ndarray) -> bytes:
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] < 2:
        raise ValueError("a mapped table needs rows and a target column")
    if table.shape[0] > 100_000 or table.shape[1] > 256:
        raise ValueError("holdout cell shape is outside the panel bound")
    if not np.isfinite(table).all() or table.min() < 0.0 or table.max() > 1.0:
        raise ValueError("mapped value is outside the unit interval")
    return np.ascontiguousarray(table, dtype="<f4").tobytes()


def append_cell(handle, cell: dict, arrays) -> None:
    fit = arrays["fit"]
    holdout = arrays["holdout"]
    synthetic = arrays["synthetic"]
    if fit.shape[1] != holdout.shape[1] or fit.shape[1] != synthetic.shape[1]:
        raise ValueError("fit, holdout, and synthetic widths differ")
    handle.write(
        struct.pack("<IIII", fit.shape[0], holdout.shape[0], synthetic.shape[0], fit.shape[1])
    )
    handle.write(_label(str(cell["dataset"])))
    handle.write(_label(str(cell["method"])))
    handle.write(_label(str(cell["configuration"])))
    handle.write(struct.pack("<I", int(cell["sample_seed"])))
    for table in (fit, holdout, synthetic):
        handle.write(_matrix(table))


def _scored(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        prior = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    membership = prior.get("membership_auc")
    marginal = prior.get("marginal_fidelity")
    return (
        prior.get("counts_as_dope_win") is False
        and isinstance(membership, (int, float))
        and isinstance(marginal, (int, float))
        and np.isfinite(membership)
        and np.isfinite(marginal)
    )


def score_manifest(manifest: Path, binary: Path, out_dir: Path) -> None:
    refuse_test_path(manifest)
    document = json.loads(manifest.read_text())
    if document.get("counts_as_dope_win") is not False or document.get("official_tests_opened") is not False:
        raise ValueError("cohort claims are not allowed")
    pack = out_dir / "holdout.pack"
    selected = []
    with pack.open("wb") as handle:
        handle.write(MAGIC)
        for cell in document["cells"]:
            mapped = Path(cell["mapped"])
            refuse_test_path(mapped)
            sidecar = mapped.with_name(mapped.stem + ".holdout.json")
            if _scored(sidecar):
                continue
            arrays = np.load(mapped)
            append_cell(handle, cell, arrays)
            selected.append(cell)
            if len(selected) % 100 == 0:
                print(f"packed {len(selected)}", flush=True)
    if not selected:
        print(f"already scored {len(document['cells'])} cells", flush=True)
        return
    print(f"packed {len(selected)}", flush=True)
    jsonl = out_dir / "holdout.jsonl"
    with jsonl.open("w") as sink:
        completed = subprocess.run(
            [str(binary), "score", str(pack)],
            check=False,
            stdout=sink,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"holdout score exited {completed.returncode}")
    lines = [line for line in jsonl.read_text().splitlines() if line]
    if len(lines) != len(selected):
        raise RuntimeError("holdout score did not return one row per packed cell")
    for cell, line in zip(selected, lines, strict=True):
        scored = json.loads(line)
        if (
            scored.get("dataset") != cell["dataset"]
            or scored.get("method") != cell["method"]
            or scored.get("sample_seed") != cell["sample_seed"]
            or scored.get("counts_as_dope_win") is not False
        ):
            raise RuntimeError("holdout row does not match the packed cell")
        mapped = Path(cell["mapped"])
        mapped.with_name(mapped.stem + ".holdout.json").write_text(json.dumps(scored, sort_keys=True) + "\n")
    print(f"holdout wrote {len(selected)}", flush=True)


def main() -> None:
    manifest = Path(os.environ["MFS_V3_MANIFEST"])
    binary = Path(os.environ["MFS_V3_BIN"])
    out_dir = manifest.parent
    score_manifest(manifest, binary, out_dir)


if __name__ == "__main__":
    main()
