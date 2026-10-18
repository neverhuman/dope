#!/usr/bin/env python3
"""Validate and render the Figure 1 protocol until E05 admits measurements.

The committed ledger is typed and intentionally contains no measurements. Fixture
points are accepted only by tests and can never be rendered as paper evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "docs/whitepaper/generated"
LEDGER = GEN / "figure-one-ledger.json"
DEFAULT_LEDGER_TEXT = '''{
  "format": "dope-figure-one-ledger-v1",
  "mode": "protocol_only",
  "contract": {
    "question": "How few complete dataset-specific bytes preserve learning across required models?",
    "x": "actual_complete_recipient_bytes",
    "y": "source_balanced_eligible_task_coverage",
    "thresholds": {"transfer": 0.9, "drivers": 0.8},
    "mandatory_models": [],
    "max_main_comparators": 5,
    "supplement_comparators": "all_preregistered",
    "uncertainty": "source_cluster",
    "privacy_label": "empirical privacy unmeasured; no formal DP claim",
    "denominator": "eligible tasks fixed before outcomes; unsupported, failed and missing remain in denominator disposition",
    "bytes": "complete recipient-read dataset-specific payload; no extrapolated B90",
    "panel": "source-balanced; no independent median-x/median-y point"
  },
  "retained_states": ["unsupported", "failed", "missing", "native_large", "not_reached", "eligible"],
  "compressed_data_controls": "nonrelease_only",
  "points": [],
  "status": "E05 admission absent; engineering renderer only"
}
'''
CAPTION = ("Figure 1. How few complete dataset-specific bytes preserve learning across required models? "
           "Protocol schematic only; measured inputs await E05 admission. The planned x-axis is actual "
           "complete recipient bytes and the y-axis is source-balanced eligible-task C(B), requiring "
           "T90 and D80 across every mandatory model. No empirical privacy claim is made.")
STATES = {"unsupported", "failed", "missing", "native_large", "not_reached", "eligible"}
REQUIRED = ("transfer", "drivers", "privacy", "bytes", "denominator", "panel", "uncertainty")


def validate(data: dict, *, allow_fixture: bool = False) -> None:
    if data.get("format") != "dope-figure-one-ledger-v1":
        raise ValueError("unknown Figure 1 ledger format")
    if data.get("mode") not in ({"protocol_only", "illustrative_fixture"} if allow_fixture else {"protocol_only"}):
        raise ValueError("measured or fixture data cannot be published as Figure 1")
    contract = data.get("contract", {})
    if contract.get("x") != "actual_complete_recipient_bytes" or contract.get("y") != "source_balanced_eligible_task_coverage":
        raise ValueError("Figure 1 axis contract mismatch")
    if contract.get("thresholds") != {"transfer": 0.90, "drivers": 0.80}:
        raise ValueError("Figure 1 thresholds must be T90/D80")
    if contract.get("privacy_label") != "empirical privacy unmeasured; no formal DP claim":
        raise ValueError("privacy label missing or overstated")
    if contract.get("max_main_comparators") != 5 or contract.get("uncertainty") != "source_cluster":
        raise ValueError("panel or uncertainty contract mismatch")
    if "eligible tasks fixed before outcomes" not in contract.get("denominator", ""):
        raise ValueError("denominator must be fixed before outcomes")
    if "complete recipient-read" not in contract.get("bytes", "") or "no extrapolated B90" not in contract.get("bytes", ""):
        raise ValueError("byte definition must use complete bytes and forbid extrapolated B90")
    if "source-balanced" not in contract.get("panel", "") or "no independent median" not in contract.get("panel", ""):
        raise ValueError("panel must preserve paired source-balanced points")
    for state in data.get("retained_states", []):
        if state not in STATES:
            raise ValueError(f"unknown retained state: {state}")
    if not set(STATES).issubset(data.get("retained_states", [])):
        raise ValueError("all evidence states must remain representable")
    if data.get("compressed_data_controls") != "nonrelease_only":
        raise ValueError("compressed-data controls must remain nonrelease")
    if not allow_fixture and data.get("points"):
        raise ValueError("protocol-only ledger cannot contain points")
    for point in data.get("points", []):
        if point.get("state") not in STATES:
            raise ValueError(f"unknown retained state: {point.get('state')}")
        if not point.get("source_cluster"):
            raise ValueError("each point needs a source cluster")
        if point.get("bytes") is not None and point.get("coverage") is not None and point.get("paired") is not True:
            raise ValueError("unpaired x/y pseudo-points are forbidden")


def draw():
    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    axis.set_xscale("log")
    axis.set_xlim(1, 1_000_000)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Actual complete recipient bytes (log scale)")
    axis.set_ylabel("Source-balanced eligible-task C(B), T90/D80 across all mandatory models")
    axis.set_title("How few bytes preserve learning across required models?", pad=16, weight="bold")
    axis.grid(True, which="both", color="#e2e8f0", linewidth=.6)
    axis.text(.5, .57, "PROTOCOL SCHEMATIC", transform=axis.transAxes, ha="center", va="center",
              fontsize=19, weight="bold", color="#334155")
    axis.text(.5, .45, "T90 and D80 required for every mandatory model", transform=axis.transAxes,
              ha="center", va="center", fontsize=13)
    axis.text(.5, .36, "No measured points admitted; E05 confirmation pending", transform=axis.transAxes,
              ha="center", va="center", fontsize=12)
    axis.text(.5, .23, "Illustrative design only", transform=axis.transAxes,
              ha="center", va="center", fontsize=11, color="#64748b")
    figure.text(.5, .015, "Empirical privacy unmeasured · no formal DP claim", ha="center", fontsize=10)
    figure.tight_layout(rect=(0, .05, 1, 1))
    return figure


def render() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text(DEFAULT_LEDGER_TEXT)
    data = json.loads(LEDGER.read_text())
    validate(data)
    figure = draw()
    try:
        for extension in ("svg", "png", "pdf"):
            (GEN / f"figure-one.{extension}").write_bytes(_render_bytes(figure, extension))
    finally:
        plt.close(figure)
    caption = CAPTION + "\n"
    (GEN / "figure-one-caption.txt").write_text(caption)
    (GEN / "figure-one-hashes.json").write_text(json.dumps({
        "format": "dope-figure-one-hashes-v1",
        "ledger_sha256": hashlib.sha256(LEDGER.read_bytes()).hexdigest(),
        "caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
        "files": {ext: hashlib.sha256((GEN / f"figure-one.{ext}").read_bytes()).hexdigest()
                  for ext in ("svg", "png", "pdf")},
        "evidence_admitted": False,
        "publication_mode": "protocol_schematic",
    }, indent=2, sort_keys=True) + "\n")


def _render_bytes(figure, extension: str) -> bytes:
    buffer = io.BytesIO()
    metadata = {"Creator": "figure_one.py"}
    if extension == "pdf":
        metadata.update({"CreationDate": None, "ModDate": None})
    if extension == "svg":
        matplotlib.rcParams["svg.hashsalt"] = "dope-figure-one-v1"
        metadata["Date"] = None
    figure.savefig(buffer, format=extension, metadata=metadata)
    rendered = buffer.getvalue()
    if extension == "svg":
        rendered = b"\n".join(line.rstrip() for line in rendered.splitlines()) + b"\n"
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if args.check:
            validate(json.loads(LEDGER.read_text()))
            figure = draw()
            try:
                manifest = json.loads((GEN / "figure-one-hashes.json").read_text())
                caption = CAPTION + "\n"
                if (GEN / "figure-one-caption.txt").read_text() != caption:
                    raise ValueError("generated caption is stale")
                if manifest.get("ledger_sha256") != hashlib.sha256(LEDGER.read_bytes()).hexdigest():
                    raise ValueError("ledger hash is stale")
                if manifest.get("caption_sha256") != hashlib.sha256(caption.encode()).hexdigest():
                    raise ValueError("caption hash is stale")
                if manifest.get("evidence_admitted") is not False or manifest.get("publication_mode") != "protocol_schematic":
                    raise ValueError("protocol-only publication status is missing")
                for extension in ("svg", "png", "pdf"):
                    expected = GEN / f"figure-one.{extension}"
                    if not expected.is_file():
                        raise ValueError(f"generated protocol schematic missing {extension}")
                    digest = hashlib.sha256(expected.read_bytes()).hexdigest()
                    if manifest.get("files", {}).get(extension) != digest:
                        raise ValueError(f"{extension} hash is stale")
                    regenerated = hashlib.sha256(_render_bytes(figure, extension)).hexdigest()
                    if digest != regenerated:
                        raise ValueError(f"generated {extension} is stale")
            finally:
                plt.close(figure)
            print("Figure 1 protocol schematic and ledger are valid")
        else:
            render()
            print("Figure 1 protocol schematic rendered; no evidence admitted")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
