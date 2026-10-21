# TDL paper package

The canonical manuscript package for **D.O.P.E.: How Few Bytes Carry What a Table Teaches**
(metric name TDL; owner decision D-01). This is the active paper landing page for the
manuscript in `main.tex` and supplement in `supplement.tex`, both in the ICLR 2027 layout
(`iclr2027_conference.sty`, final-copy mode with the venue header removed). Figure 1 is the
transfer-per-byte frontier (`generated/transfer-per-byte.pdf`): published generators on the
historical panel, and the equal-byte study against real rows. Every result is labelled
development validation and selection-exposed; prospective TDL confirmation remains gated on E05. Build with `bash paper/build.sh`; it writes
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
| `generated/` | numbers, tables and figures rendered from `generated/headline-results.json` | L2 (tools/claims) |
| `generated/equal-bytes-*`, `generated/transfer-per-byte.*`, `generated/readme-equal-bytes.md` | macros, tables, figures and the README block of the equal-byte study, rendered by `docs/whitepaper/scripts/equal_bytes_render.py` and authenticated by the claims firewall | L3 |
| `equal-bytes.bib` | entries copied from the historical bibliographies plus two with DOI or venue URL | L3 |
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
| `generated/hero-frontier.pdf` | the hero figure generator (T-072), from the same CSV as the README image | Figure 1 |
| `generated/result-card.tex` | the result card, one evidence label per row | Table 1 |
| `generated/contract-table.tex`, `byte-ledger-table.tex`, `auditor-registry-table.tex`, `generator-registry-table.tex`, `dataset-dispositions.tex` | contract, charge manifests and registries | supplement |

## Rules

- Follow [docs/claims-policy.md](../docs/claims-policy.md): an evidence label on every table and
  figure, "smallest tested", never an optimum, no privacy claim, and no unqualified auditor in a
  headline.
- Do not type a number into the `.tex` sources. Add it to `generated/headline-results.json` and
  render it.
- The manuscript uses the ICLR layout in final-copy mode; `\iclrfinalcopy` removed gives the
  anonymous review build.
- Abstract numbers are generated macros, each in a sentence that names its evidence status.
  Replace them with sealed-test macros when that run exists.
- Small integers inside mathematics come from generated constants (`\One`, `\Two`, ...) so the
  public lint can keep refusing bare digits.

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
