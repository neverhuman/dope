"""Measure in-memory sampling throughput of DOPE programs (rows per second).

    python3 bench/real-world/throughput.py --data target/bench-data --out bench/real-world/results/throughput.json

For a few confirmation tables, compile a program at each budget and time
`dope throughput` (no output written; the fastest of three runs of one
million rows). Run it on an otherwise idle machine; the JSON records the
device description the engine reports (backend and thread count).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

DOPE = os.environ.get("DOPE_BIN", str(Path(__file__).resolve().parents[2] / "engine/target/release/dope"))
TABLES = ["adult", "california", "covertype", "credit"]
BUDGETS = ["4k", "64k"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--panel", default="confirm")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = []
    for name in TABLES:
        d = a.data / a.panel / name
        meta = json.loads((d / "meta.json").read_text())
        for budget in BUDGETS:
            with tempfile.TemporaryDirectory() as tmp:
                model, report = Path(tmp) / "m.dope", Path(tmp) / "r.json"
                subprocess.run([DOPE, "fit", str(d / "train.csv"), "--target", meta["target"], "--budget", budget,
                                "--device", "cpu", "-o", str(model), "--report", str(report), "-q"], check=True)
                res = subprocess.run([DOPE, "throughput", str(model), "-n", str(a.rows), "--repeats", "3",
                                      "--device", a.device], check=True, capture_output=True, text=True)
                line = json.loads(res.stdout.strip().splitlines()[-1])
                rep = json.loads(report.read_text())
                ins = json.loads(subprocess.run([DOPE, "inspect", str(model), "--json"], check=True,
                                                capture_output=True, text=True).stdout)
                b = ins["bytes"]
                out.append({"dataset": name, "budget": budget, "bytes": model.stat().st_size,
                            "header": b["header"], "stored": b["stored"], "metadata": b["metadata"],
                            "inflated": b["metadata"] + sum(b["tensors"].values()),
                            "forest_bytes": b["tensors"].get("forest", 0),
                            "features": rep["features"], "trees": rep["trees"], "regimes": rep["regimes"],
                            **line})
                print(f"{name:<12} {budget:>4} {out[-1]['bytes']:>7} B  {line['rows_per_second']:>12,} rows/s  "
                      f"({line['device']})", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
