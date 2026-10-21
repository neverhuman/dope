# TDL paper package

The canonical manuscript package for **D.O.P.E.: Compiling Tables into Small Programs That
Preserve What Models Learn** (the metric name TDL awaits owner decision D-01). This is the active
paper landing page for the manuscript in `main.tex` and supplement in `supplement.tex`. The
manuscript has the shape of a full conference paper: problem and metric, a short transfer theory
(retention identities for squared and log loss, two corollaries, a size bound), the compiler, the
protocol, results and limitations. Figure 1 shows the compression frontier (worst-auditor
retention against charged bytes, and retained lift per KiB); Figure 2 contains the authenticated
foundation-model diagnostic bars. Table 1 is the generated frontier table. Every number is
historical selection-exposed validation or a labeled local demonstration; prospective TDL
confirmation remains gated on E05. Build with `bash paper/build.sh`; it writes
`target/paper/main.pdf` and `target/paper/supplement.pdf`. The frozen historical v2 validation
paper remains available as [`docs/whitepaper/dope-mfs.pdf`](../docs/whitepaper/dope-mfs.pdf)
and [`docs/whitepaper/supplement.pdf`](../docs/whitepaper/supplement.pdf), with its
qualifications documented in [`docs/whitepaper/`](../docs/whitepaper/README.md).

| Path | Contents | Owner |
| --- | --- | --- |
| `main.tex`, `supplement.tex` | active manuscript and supplement | L3 |
| `macros.tex` | shared macros; reads generated macro files first, then typesets visible `[pending]` placeholders | L3 |
| `references.bib` | starts as a copy of the historical paper's DOI-checked bibliography; add only verified entries | L3 |
| `build.sh` | checks the contract macros, then builds both PDFs into `target/paper/` and fails on unresolved references or citations | L3 |
| `scripts/contract_macros.py` | writes `generated/contract-macros.tex`; `--check` refuses drift and any headline-threshold disagreement | L3 |
| `generated/` | numbers, tables and figures rendered from `generated/headline-results.json` (tools/claims), from the pinned v2 ledgers (`compression-*`, `docs/whitepaper/scripts/compression_frontier.py`) and from the committed quickstart run (`quickstart-macros.tex`, `docs/whitepaper/scripts/quickstart_block.py`) | L2 |
| `related.bib` | related-work entries copied from the whitepaper's verified `related.bib`, plus Crossref-verified additions | L3 |
| `archive/` | review ledgers and readiness notes moved out of the historical paper directory (T-111) | L3 |

## Build

```bash
python3 -B paper/scripts/contract_macros.py   # regenerate the contract macros after a contract change
bash paper/build.sh        # check them, then write target/paper/main.pdf and target/paper/supplement.pdf
```

Toolchain: pdfTeX from TeX Live 2023, `latexmk` 4.83 and BibTeX 0.99d, using only standard
packages (`amsmath`, `booktabs`, `graphicx`, `xcolor`, `natbib`, `hyperref`). The PDFs are build output under `target/paper/`.

## Generated inputs

The manuscript reads these files when they exist and prints a red `[pending]` box when they
do not, so a missing input can never look like a measurement:

| File | Source | Used in |
| --- | --- | --- |
| `generated/contract-macros.tex` | `paper/scripts/contract_macros.py` from `contracts/tdl-contract-v1.json`: `\TauT`, `\TauD`, `\HeadlinePoint`, `\ByteGrid`, `\SeedDesign`, `\MetricName`, `\ContractId`, `\ContractState`, bootstrap plan | metric definition, protocol, supplement A and D |
| `generated/headline-macros.tex` | `generated/headline-results.json` (T-070): `\ResultStatus` and any printed value | title block, evidence section |
| `generated/compression-frontier.pdf` | compression_frontier.py, from pinned historical ledgers with method-specific cohort counts | Figure 1 |
| `generated/hero-frontier.pdf` | foundation-model diagnostic renderer from its authenticated CSV | Figure 2 |
| `generated/compression-table.tex` | generated worst-auditor frontier table with per-method cohort counts | Table 1 |
| `generated/result-card.tex` | retained historical result card and evidence labels | historical package compatibility |
| `generated/contract-table.tex`, `byte-ledger-table.tex`, `auditor-registry-table.tex`, `generator-registry-table.tex`, `dataset-dispositions.tex` | contract, charge manifests and registries | supplement |

## Rules

- Follow [docs/claims-policy.md](../docs/claims-policy.md): an evidence label on every table and
  figure, "smallest tested", never an optimum, no privacy claim, and no unqualified auditor in a
  headline.
- Do not type a result number into the TeX sources. Route a reviewed, authenticated input through
  its appropriate generator. Frozen historical inputs, including generated/headline-results.json,
  retain their original bytes and scope; new measurements need their own admitted source.
- Use the neutral `article` class until a submission build exists. A venue build will be a
  separate target generated from the same sources.
- The abstract's result sentences carry selection-exposed validation numbers through generated
  macros, with that label stated in the same sentence. Replace them with the sealed-test numbers
  when the sealed run exists (owner decision; this relaxes the earlier rule of an abstract without
  results).

## Contract agreement

`python3 -B paper/scripts/contract_macros.py --check` runs inside `build.sh`, and so inside the
paper-clean CI job. It fails when any of these disagrees with the contract's headline operating
point (`contracts/tdl-contract-v1.json`, `headline_operating_point`):

- `generated/contract-macros.tex`;
- the `thresholds` and `metric_name` in `generated/headline-results.json`;
- the `\TdlTransferThreshold`, `\TdlDriverThreshold` and `\TdlMetricName` macros that the claims
  renderer writes;
- the README's `KiB@<operating point>` form and its threshold wording.

A contract change therefore requires regenerating the macros and updating the README in the same
change.

The frozen quick-start Adult/Abalone reports use incomplete prototype encoder byte accounting. Current JSON receiver reload is fixture-tested separately; do not treat those historical subtotals or ratios as complete transfer sizes or new measurements.
