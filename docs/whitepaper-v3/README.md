# DOPE v3.0.0 white paper

**D.O.P.E.: Downstream Objective-Preserving Encoding. Release v3.0.0: synthetic tables from a file of
at most 2 KiB, ranked against independent generators by Transfer Score.**

The paper describes the v3.0.0 encoder (`dope fit --format auto-v8`) and the `DSE8` codec, and ranks
it by Transfer Score, 100 × (1 + TRG), against four third-party generators scored on the same
cells: SDV GaussianCopula, CTGAN and TVAE, and arfpy ARF. It reports only this release; earlier
DOPE releases are not compared (owner decision, 2026-11-24), and `render_v3_paper.py --check`
fails if a section cites one. The earlier manuscripts ([`docs/whitepaper/`](../whitepaper/),
[`docs/whitepaper-v2/`](../whitepaper-v2/)) are archived unchanged.

No number is typed into the manuscript. Every quantity is a `\VThree{...}` macro written by
`scripts/render_v3_paper.py` from the committed receipts, contracts and Rust constants, or a cell
of a table written by the Rust `dope_kpi render-paper-tables`, and `--check` rejects a digit typed
into a section.

## Layout

| Path | Role |
| --- | --- |
| `dope-v3.tex` | Preamble, title block and `\input` of each section |
| `sections/*.tex` | One file per section (`00-abstract` to `09-reproducibility`) and per appendix (`appendix-*`) |
| `references-v3.bib`, `dope-v3.bbl` | Bibliography and the bibtex output the PDF was built with |
| `dope-v3.pdf` | The built paper (deterministic: `scripts/build_v3_pdf.sh`) |
| `figures/hero-v3-methods.pdf` | Figure 1: `bench/real-world/figures/hero-v3-methods-light.svg` (`dope_kpi render-hero --methods`), converted with `rsvg-convert -f pdf` |
| `generated/v3-macros.tex` | Every `\VThree` macro |
| `generated/v3-ranking-table.tex` | Table 2: the ranking by Transfer Score (`dope_kpi render-paper-tables`) |
| `generated/v3-paired-table.tex` | Table 3: DOPE minus each generator, from `evidence/paper/v3-vs-*.json` (`dope_kpi render-paper-tables`) |
| `generated/v3-render-manifest.json` | SHA-256 of every input and output, and the hero conversion command, tool and hashes |
| `scripts/render_v3_paper.py` | Renderer and checker (stdlib only) |
| `scripts/v3_paper_tables.py` | Port of `dope_kpi render-readme` cell formatting; fails unless every DOPE row equals the README block |
| `scripts/v3_paper_methods.py` | Loads and checks the third-party receipts and the four DOPE-versus-method pairs (exact sign reversal of `compare-methods.json`), writes their macros, checks the prose about them |
| `scripts/v3_paper_final_look.py` | Loads and checks the final-look receipts (Addendum F); kept for the scoreboard, not cited by the paper |
| `scripts/test_render_v3_paper.py` | Unit tests |
| `scripts/build_v3_pdf.sh` | Deterministic `latexmk` build in `target/paper-v3/` |

## Third-party baselines

The scoreboard directory holds one run of four third-party generators (SDV GaussianCopula, CTGAN
and TVAE; arfpy ARF) with library defaults and no byte cap: `primary-methods.json`,
`compare-methods.json` (each method against DOPE), `MANIFEST.json` (versions, recipes, licences,
commands) and `methods-report.md`. `evidence/paper/v3-vs-<method>.json` re-reduces the same
scores in the paper's direction, DOPE against each method; the renderer fails unless each is
the exact sign reversal of its row in `compare-methods.json`. The renderer also fails unless the
receipts are all present (or all absent), name the release receipts' schedule, real-data
reference, panel, contract and bootstrap, and repeat every release arm of `primary.json` field
for field.

## Build and verify

From the repository root:

```bash
python3 docs/whitepaper-v3/scripts/render_v3_paper.py --figures   # needs rsvg-convert; only when the hero SVG changes
python3 docs/whitepaper-v3/scripts/render_v3_paper.py              # macros, manifest
python3 docs/whitepaper-v3/scripts/render_v3_paper.py --check      # drift, hero provenance, digit lint
bash docs/whitepaper-v3/scripts/build_v3_pdf.sh                    # dope-v3.pdf and dope-v3.bbl
```

`just paper-v3` runs the last three. `just paper-check` and `just paper-tests` include the v3
paper, and `ops/ci/paper-clean.sh` deletes its generated files, rebuilds it and requires a
zero diff.

When the scoreboard receipts change (for example after a rebase onto a new release), re-run
`render_v3_paper.py` (with `--figures` if the hero SVG changed) and `build_v3_pdf.sh`. The text
cites values only through macros, but its comparative wording (which arm is higher, which
guards fail, which outcome each pair has) is typed; `prose_errors` in the renderer
checks each such claim against the receipts and names the one that no longer holds.
