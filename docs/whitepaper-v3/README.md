# DOPE v3.0.0 White Paper

This directory houses the versioned **DOPE v3.0.0** white paper:

> **D.O.P.E. v3.0.0: Downstream Objective-Preserving Encoding — Calibrated `DSE8` Sub-2 KiB Tabular Synthesis, FIT-Only Evidence Extraction, Stratified 2-Fold Acceptance Gating, and Neural Compiler Audit**

The historical manuscripts ([`docs/whitepaper/dope-mfs.tex`](../whitepaper/dope-mfs.tex) and [`docs/whitepaper-v2/dope-v2.tex`](../whitepaper-v2/dope-v2.tex)) remain untouched so `ops/ci/paper-clean.sh` stays zero-diff.

## Layout

| Path | Role |
| --- | --- |
| `dope-v3.tex` | Main DOPE v3.0.0 LaTeX manuscript (zero hand-typed numbers; uses `\VThree{...}` macros) |
| `dope-v3.pdf` | Compiled DOPE v3.0.0 white paper PDF |
| `references-v3.bib` | Bibliography |
| `scripts/render_v3_paper.py` | Receipt-bound macro & figure renderer (`--check` fails on any drift from committed `scoreboard/v3.0.0/` JSON receipts, contracts, or Rust constants) |
| `generated/v3-macros.tex` | Generated `\VThree{...}` macros bound to `bench/real-world/results/scoreboard/v3.0.0/`, `contracts/evidence-layout-v1.json`, `contracts/dope-kpi.json`, and `engine/dope-core/src/single_encoding/v8/` |
| `figures/v3-frontier-and-tasks.pdf` | Vector 2-panel value frontier and per-task/MLP breakdown rendered from committed `primary.json` |

## Build & Verify

```bash
python3 docs/whitepaper-v3/scripts/render_v3_paper.py --figures
(cd docs/whitepaper-v3 && latexmk -pdf -interaction=nonstopmode dope-v3.tex)
python3 docs/whitepaper-v3/scripts/render_v3_paper.py --check
```
