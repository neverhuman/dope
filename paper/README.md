# The DOPE White Paper

The white paper of the DOPE repository is **`dope1.tex`** (`target/paper/dope1.pdf`, [compiled PDF on v1.0.0 release](https://github.com/neverhuman/dope/releases/download/v1.0.0/dope1.pdf)) and its companion supplement **`dope1-supplement.tex`** (`target/paper/dope1-supplement.pdf`, [compiled supplement PDF on v1.0.0 release](https://github.com/neverhuman/dope/releases/download/v1.0.0/dope1-supplement.pdf)):

> **D.O.P.E.: Downstream Objective-Preserving Encoding — Learning to Compile Tables into Small Generative Programs That Train Real-World Models**

It covers the standalone pure-Rust `dope` engine in [`engine/`](../engine/), its sixteen-table real-world benchmark in [`bench/real-world/`](../bench/real-world/), and the unified `DOPE-MLC-v2` (`1,000` datasets) and `DOPE-MLC-v3` (`1,069` datasets, `4,276` dataset–tier cells, pure-Rust `SafeTensors` parity) multi-level deep-learning landmark encoder study in [`research/levels/`](../research/levels/) ([`docs/multilevel-landmark-dl-study.md`](../docs/multilevel-landmark-dl-study.md)).

- `dope1.tex`: preprint layout (`target/paper/dope1.pdf`).
- `dope1-venue.tex`: identical body in double-blind ICLR 2027 layout (`target/paper/dope1-venue.pdf`).
- `dope1-supplement.tex`: proofs, artifact format, settings, and per-table results (`target/paper/dope1-supplement.pdf`).

The body is `dope1-content.tex` and Figure 1 is `dope1-hero.tex`, shared by both builds. Every result is a `\G{...}` macro rendered from the committed benchmark records by `bench/real-world/render.py`, and every setting is a `\Cfg...` macro read from the engine's Rust source at render time, so the white paper cannot drift from the code or the records.

| Path | Contents | Owner |
| --- | --- | --- |
| `dope1.tex`, `dope1-venue.tex`, `dope1-supplement.tex`, `dope1-content.tex`, `dope1-hero.tex`, `dope1-preamble.tex` | The DOPE white paper (preprint and venue builds) and supplement | L3 |
| `macros.tex` | Shared macros; reads generated macro files first, then typesets visible `[pending]` placeholders | L3 |
| `references-dope1.bib`, `references.bib` | Verified bibliography entries | L3 |
| `build.sh` | Checks the contract macros and the DOPE benchmark surfaces, builds the PDFs into `target/paper/`, fails on unresolved references or citations, and requires a generated hero as Figure 1 on page 1 | L3 |
| `scripts/contract_macros.py` | Writes `generated/contract-macros.tex`; `--check` refuses drift and any headline-threshold disagreement | L3 |
| `generated/dope1-*` | Figures, tables, and macros of the DOPE white paper rendered by `bench/real-world/render.py` (hash-listed in `bench/real-world/results/render-manifest.json`) | L2 (tools/claims) |

## Build

```bash
python3 -B paper/scripts/contract_macros.py   # regenerate contract macros after a contract change
bash paper/build.sh                           # verify macros & surfaces and build target/paper/dope1*.pdf
```

Toolchain: pdfTeX from TeX Live 2023, `latexmk` 4.83, and BibTeX 0.99d, using standard packages (`amsmath`, `booktabs`, `graphicx`, `xcolor`, `natbib`, `hyperref`, `amsthm`, `cleveref`, `xr-hyper`, `microtype`, `enumitem`, `caption`, `tikz`, and `mathptmx`). The PDFs are built into `target/paper/`.

## Generated inputs

The manuscript reads these files when they exist and prints a red `[pending]` box when they do not, so a missing input can never look like a measurement:

| File | Source | Used in |
| --- | --- | --- |
| `generated/dope1-macros.tex`, `dope1-config.tex` | `bench/real-world/render.py`: results (`\G{One...}`, `\G{Mlc...}`) from `summary.json` and `multilevel-convergence-snapshot.json`; settings (`\Cfg...`) from the engine and harness sources | White paper and supplement |
| `generated/dope1-hero.pdf`, `dope1-convergence.pdf`, `dope1-tables.pdf`, `dope1-coverage.pdf`, `dope1-families.pdf`, `dope1-audit.pdf` | `bench/real-world/render.py` (PNG twins in `bench/real-world/figures/`) | Figure 1 and experiments |
| `generated/dope1-table.tex`, `dope1-multilevel.tex`, `dope1-equalbytes.tex`, `dope1-ablation.tex`, `dope1-pertable.tex`, `dope1-throughput*.tex` | `bench/real-world/render.py` | Experiments and supplement |
| `generated/contract-macros.tex` | `paper/scripts/contract_macros.py` from `contracts/tdl-contract-v1.json`: `\TauT`, `\TauD`, `\HeadlinePoint`, `\ByteGrid`, `\SeedDesign`, `\MetricName`, `\ContractId`, `\ContractState`, bootstrap plan | Metric definition and protocol |

## Rules

- Follow [docs/claims-policy.md](../docs/claims-policy.md): an evidence label on every table and figure, "smallest tested", never an optimum, no privacy claim, and no unqualified auditor in a headline.
- Do not type result numbers into the `.tex` sources. Route reviewed inputs through the renderer so all numbers remain hash-bound to committed receipts.
- The DOPE preprint uses the `article` class; `dope1-venue.tex` is built from the same body and stays double-blind (no names, affiliations, or repository URLs).

## Contract agreement

`python3 -B paper/scripts/contract_macros.py --check` runs inside `build.sh` and inside the `paper-clean` CI job. It fails when any of these disagrees with the contract's headline operating point (`contracts/tdl-contract-v1.json`, `headline_operating_point`):

- `generated/contract-macros.tex`;
- the `thresholds` and `metric_name` in `generated/headline-results.json`;
- the `\TdlTransferThreshold`, `\TdlDriverThreshold`, and `\TdlMetricName` macros that the claims renderer writes;
- the README's `KiB@<operating point>` form and its threshold wording.
