# DOPE v3.0.0 white paper

**D.O.P.E.: Downstream Objective-Preserving Encoding. Release v3.0.0: compiling a table into at
most 2 KiB with per-table codec selection, and a twin-referenced evaluation.**

The paper describes the v3.0.0 release encoder (`dope fit --format auto-v8`), the `DSE8` codec,
the FIT-only evidence bank and the `dope_kpi` evaluation, and reports the scoreboard of
[`bench/real-world/results/scoreboard/v3.0.0/`](../../bench/real-world/results/scoreboard/v3.0.0/README.md).
Earlier releases appear only as baselines. The v1 and v2 manuscripts
([`docs/whitepaper/`](../whitepaper/), [`docs/whitepaper-v2/`](../whitepaper-v2/)) are unchanged.

No number is typed into the manuscript. Every quantity is a `\VThree{...}` macro written by
`scripts/render_v3_paper.py` from the committed receipts, contracts and Rust constants, and
`--check` rejects a digit typed into a section.

## Layout

| Path | Role |
| --- | --- |
| `dope-v3.tex` | Preamble, title block and `\input` of each section |
| `sections/*.tex` | One file per section (`00-abstract` to `09-reproducibility`) and per appendix (`appendix-*`) |
| `references-v3.bib`, `dope-v3.bbl` | Bibliography and the bibtex output the PDF was built with |
| `dope-v3.pdf` | The built paper (deterministic: `scripts/build_v3_pdf.sh`) |
| `figures/hero-v3.pdf` | Figure 1: the README hero `bench/real-world/figures/hero-v3-light.svg`, converted with `rsvg-convert -f pdf` |
| `generated/v3-macros.tex` | Every `\VThree` macro |
| `generated/v3-results-table.tex` | Table 2: the README arm table, same rows, columns and values, wrapped into two panels |
| `generated/v3-comparison-table.tex` | Table 3: the README comparison row with each condition and guard of the decision rule |
| `generated/v3-methods-table.tex` | Table 4: each third-party generator against `auto-v8` (`compare-methods.json`) |
| `generated/v3-render-manifest.json` | SHA-256 of every input and output, and the hero conversion command, tool and hashes |
| `scripts/render_v3_paper.py` | Renderer and checker (stdlib only) |
| `scripts/v3_paper_tables.py` | Port of `dope_kpi render-readme` cell formatting; fails unless every cell equals the README block |
| `scripts/v3_paper_methods.py` | Loads and checks the third-party receipts, writes their macros, checks the prose about them |
| `scripts/test_render_v3_paper.py` | Unit tests |
| `scripts/build_v3_pdf.sh` | Deterministic `latexmk` build in `target/paper-v3/` |

## Third-party baselines

The scoreboard directory holds one run of four third-party generators (SDV GaussianCopula, CTGAN
and TVAE; arfpy ARF) with library defaults and no byte cap: `primary-methods.json`,
`compare-methods.json`, `MANIFEST.json` (versions, recipes, licences, commands) and
`methods-report.md`. The renderer adds the four arms to Table 2 between the DOPE arms and the
controls, writes Table 4 from `compare-methods.json`, and fails unless the three receipts are all
present (or all absent), name the release receipts' schedule, real-data reference, panel, contract
and bootstrap, and repeat every release arm of `primary.json` field for field. These rows are not
in the README block.

## Build and verify

From the repository root:

```bash
python3 docs/whitepaper-v3/scripts/render_v3_paper.py --figures   # needs rsvg-convert; only when the hero SVG changes
python3 docs/whitepaper-v3/scripts/render_v3_paper.py              # macros, tables, manifest
python3 docs/whitepaper-v3/scripts/render_v3_paper.py --check      # drift, hero provenance, digit lint
bash docs/whitepaper-v3/scripts/build_v3_pdf.sh                    # dope-v3.pdf and dope-v3.bbl
```

`just paper-v3` runs the last three. `just paper-check` and `just paper-tests` include the v3
paper, and `ops/ci/paper-clean.sh` deletes its generated files, rebuilds it and requires a
zero diff.

When the scoreboard receipts change (for example after a rebase onto a new release), re-run
`render_v3_paper.py` (with `--figures` if the hero SVG changed) and `build_v3_pdf.sh`. The text
cites values only through macros, but its comparative wording (which arm is higher, which
guards fail, that no third-party arm has been scored) is typed; `prose_errors` in the renderer
checks each such claim against the receipts and names the one that no longer holds.
