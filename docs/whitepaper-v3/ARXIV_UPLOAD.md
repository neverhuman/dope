# Uploading DOPE v3.0.0 to arXiv

This folder provides a turnkey, pre-verified arXiv TeX source bundle built directly from the receipt-bound `docs/whitepaper-v3/` manuscript.

## Ready-to-upload artifact

- **Upload zip:** [`dope-v3-arxiv.zip`](dope-v3-arxiv.zip) (~80 KB)
- **Inspectable staging folder:** [`arxiv/`](arxiv/)

Both are generated deterministically from committed sources and verified in CI by unzipping into an isolated directory outside this repository and compiling with three `pdflatex -no-shell-escape` passes (no BibTeX run required, because `dope-v3.bbl` is bundled).

## What is inside the archive

| Path inside zip | Purpose |
| --- | --- |
| `dope-v3.tex` | Self-contained manuscript (`\pdfoutput=1` on line 2; all figures rendered in TikZ/PGFPlots) |
| `dope-v3.bbl` | Pre-compiled bibliography matching `dope-v3.tex` |
| `generated/v3-macros.tex` | Receipt-derived `\VThree{...}` macros |
| `budget/data.tex` | Self-contained PGFPlots data loader (rewritten with zero `../` parent paths) |
| `data/addendum-k/*.csv`, `data/addendum-h/summary.csv` | The 8 receipt-verified aggregate CSVs read by PGFPlots |
| `examples/{install,toy-run,own-csv,docker}.bash` | Verbatim quick-start listings verified against `readme_install_smoke` |

Excluded intentionally: `dope-v3.pdf`, unused `figures/hero-v3-methods.pdf`, Python scripts, and `.aux`/`.log` build files.

## Step-by-step arXiv submission

1. Go to [arXiv Submit](https://arxiv.org/submit) and choose **Upload Files**.
2. Upload [`docs/whitepaper-v3/dope-v3-arxiv.zip`](dope-v3-arxiv.zip) directly (arXiv automatically unpacks `.zip` archives and runs `pdfLaTeX` on `dope-v3.tex` via `\pdfoutput=1`).
3. Click **Process Files** and preview the compiled PDF to confirm page count and figures match [`dope-v3.pdf`](dope-v3.pdf).
4. Suggested arXiv metadata:
   - **Primary category:** `cs.LG` (Machine Learning)
   - **Cross-lists:** `stat.ML` (Machine Learning — Statistics), `cs.DB` (Databases)
   - **Title:** `D.O.P.E.: Downstream Objective-Preserving Encoding — Synthetic Training Rows from a 2,048-Byte File`
   - **Authors:** Jepson Taylor, Ricky Ray Butler, Jordan Morrow, Alton Alexander, Jordan Hildebrandt, Besterlyne Jirongo
   - **Comments:** Release v3.0.0 technical report. Code, signed engine binary, and benchmark receipts: `https://github.com/neverhuman/dope`
   - **License:** arXiv.org perpetual, non-exclusive license to distribute (or CC BY 4.0)

## Rebuild and verify locally

Whenever the v3 manuscript or scoreboard receipts are updated, rebuild and verify the arXiv package in one command:

```bash
just arxiv-v3
```

To verify zero drift and run the isolated `pdflatex` compile test without modifying files:

```bash
bash docs/whitepaper-v3/scripts/build_arxiv_bundle.sh --check --verify
```
