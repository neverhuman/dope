# DOPE v2.0.0 White Paper

> **Archived.** This manuscript describes an earlier DOPE release and is kept unchanged for reproducibility.
> The current paper is the [DOPE v3 white paper](../whitepaper-v3/dope-v3.pdf).

This directory houses the versioned **DOPE v2.0.0** white paper:

> **D.O.P.E. v2.0.0: Downstream Objective-Preserving Encoding — Fidelity Engine 2.0, Sub-2 KiB Source-Free Single-Encoding Neural Synthesis, and Twin-Referenced Multi-Auditor Evaluation**

The historical T-111 manuscript ([`docs/whitepaper/dope-mfs.tex`](../whitepaper/dope-mfs.tex)) remains **100% frozen and untouched** so `ops/ci/paper-clean.sh` stays zero-diff.

## Layout

| Path | Role |
| --- | --- |
| `dope-v2.tex` | Main DOPE v2.0.0 LaTeX manuscript (zero hand-typed numbers; uses `\VTwo{...}` macros) |
| `dope-v2.pdf` | Compiled DOPE v2.0.0 white paper PDF |
| `references-v2.bib` | Bibliography |
| `scripts/render_v2_paper.py` | Receipt-bound macro generator (`--check` fails on any drift from committed JSON receipts or Rust constants) |
| `generated/v2-macros.tex` | Generated `\VTwo{...}` macros bound to `bench/real-world/results/v2.0.0/`, `models/single-enc-v7/manifest.json`, and `engine/dope-core/src/single_encoding/mod.rs` |
| `figures/v2-loss-and-throughput.pdf` | Vector 4-panel training loss, objective decomposition, CPU fleet throughput, and DEV trajectory figure rendered by `research/levels/archive_and_plot_v2_wave.py` |

## Build & Verify

```bash
python3 docs/whitepaper-v2/scripts/render_v2_paper.py
(cd docs/whitepaper-v2 && latexmk -pdf -interaction=nonstopmode dope-v2.tex)
python3 docs/whitepaper-v2/scripts/render_v2_paper.py --check
```
