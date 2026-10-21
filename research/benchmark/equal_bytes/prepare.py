"""Reproduce the historical PMLB remainder cohort (mixed/unknown source origins).

The name-based historical mask is not measured-source or rights admission.
It retains simulated pollen; new pilots require a separately frozen origin manifest.

Each lineage is shuffled with seed 1729; up to 2,000 rows (and at least 30%)
become untouched test rows and up to 2,000 of the rest become fit rows. Every
column is min-max projected with fit-row statistics, as in the historical
benchmark harness, and written headerless with the target last.

    python3 -B -m research.benchmark.equal_bytes.prepare --sources DIR --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "ops/ci/paper-inputs/pmlb-sources.json"
# Generator families and PMLB lineages that are themselves simulated.
SIMULATED_PREFIXES = ("feynman", "strogatz", "first_principles")
SIMULATED_MARKERS = ("_fri_", "BNG", "2dplanes", "pwLinear", "344_mv", "564_fried")
# Content-identical or near-identical sources kept once (the second id is dropped).
DUPLICATES = {"562_cpu_small": "227_cpu_small", "573_cpu_act": "197_cpu_act",
              "_deprecated_207_autoPrice": "_deprecated_195_auto_price"}
FIT_CAP, TEST_CAP, SPLIT_SEED = 2000, 2000, 1729


def historical_simulated_mask(name: str) -> bool:
    return name.startswith(SIMULATED_PREFIXES) or any(marker in name for marker in SIMULATED_MARKERS)


def historical_remainder_entries(include_simulated: bool = False) -> list[dict]:
    entries = json.loads(MANIFEST.read_text())["entries"]
    keep = []
    for entry in entries:
        name = entry["pmlb_id"]
        if historical_simulated_mask(name) and not include_simulated:
            continue
        if name in DUPLICATES:
            continue
        keep.append(entry)
    return keep


# Historical API aliases only, never source-origin attestations.
is_simulated = historical_simulated_mask
real_world_entries = historical_remainder_entries


def fetch(entry: dict, sources: Path) -> Path:
    path = sources / f"{entry['pmlb_id']}.tsv.gz"
    if not path.exists():
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "dope-equal-bytes"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"digest mismatch for {entry['pmlb_id']}")
        path.write_bytes(payload)
    return path


def split(source: Path, out: Path) -> dict:
    frame = pd.read_csv(source, sep="\t").dropna()
    order = np.random.default_rng(SPLIT_SEED).permutation(len(frame))
    frame = frame.iloc[order].reset_index(drop=True)
    n_test = min(TEST_CAP, max(int(round(0.3 * len(frame))), 30))
    test, fit = frame.iloc[:n_test], frame.iloc[n_test:n_test + FIT_CAP]
    low, high = fit.min(), fit.max()
    spread = (high - low).replace(0, 1)
    out.mkdir(parents=True, exist_ok=True)
    ((fit - low) / spread).clip(0, 1).to_csv(out / "train.csv", header=False, index=False, float_format="%.8f")
    ((test - low) / spread).clip(0, 1).to_csv(out / "test.csv", header=False, index=False, float_format="%.8f")
    meta = {"name": out.name, "n_fit": len(fit), "n_test": len(test), "d": fit.shape[1] - 1,
            "rows_available": len(frame), "columns": [str(c) for c in frame.columns],
            "low": [float(v) for v in low], "high": [float(v) for v in high]}
    (out / "meta.json").write_text(json.dumps(meta, sort_keys=True))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="include the simulated lineages of the panel")
    args = parser.parse_args()
    args.sources.mkdir(parents=True, exist_ok=True)
    for entry in historical_remainder_entries(include_simulated=args.all):
        print(json.dumps(split(fetch(entry, args.sources), args.out / entry["pmlb_id"])))


if __name__ == "__main__":
    main()
