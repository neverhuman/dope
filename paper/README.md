# TDL paper package

The canonical manuscript package for **D.O.P.E.: How Few Bytes Carry What a Table Teaches**
(metric name TDL awaits owner decision D-01). This is the active paper landing page for the
manuscript in `main.tex` and supplement in `supplement.tex`, both in the ICLR 2027 layout
(`iclr2027_conference.sty`, final-copy mode with the venue header removed). Figure 1 is the
transfer-per-byte frontier (`generated/transfer-per-byte.pdf`): published generators on the
historical panel, and the equal-byte study against real rows. Every result is labelled
development validation and selection-exposed; prospective TDL confirmation remains gated on E05. Build with `bash paper/build.sh`; it writes
`target/paper/main.pdf` and `target/paper/supplement.pdf`. The frozen historical v2 validation
paper remains available as [`docs/whitepaper/dope-mfs.pdf`](../docs/whitepaper/dope-mfs.pdf)
and [`docs/whitepaper/supplement.pdf`](../docs/whitepaper/supplement.pdf), with its
qualifications documented in [`docs/whitepaper/`](../docs/whitepaper/README.md).

The package also holds the DOPE 1.0 engine manuscript, **D.O.P.E.: Downstream Objective-Preserving
Encoding — How few bytes does a table's predictive structure need?**: `dope1.tex` (preprint),
`dope1-venue.tex` (the same body in the double-blind ICLR 2027 layout) and `dope1-supplement.tex`
(proofs, artifact format, settings and per-table results). It describes the standalone engine in
[`engine/`](../engine/) and its sixteen-table benchmark in [`bench/real-world/`](../bench/real-world/).
The body is `dope1-content.tex` and Figure 1 is `dope1-hero.tex`, shared by both builds. Every result
is a `\G{...}` macro rendered from the benchmark records by `bench/real-world/render.py`, and every
setting is a `\Cfg...` macro read from the engine's Rust source at render time, so that paper cannot
drift from the code or the records.

| Path | Contents | Owner |
| --- | --- | --- |
| `main.tex`, `supplement.tex` | active manuscript and supplement | L3 |
| `dope1.tex`, `dope1-venue.tex`, `dope1-supplement.tex`, `dope1-content.tex`, `dope1-hero.tex`, `dope1-preamble.tex` | DOPE 1.0 engine manuscript (preprint and venue builds) and supplement | L3 |
| `macros.tex` | shared macros; reads generated macro files first, then typesets visible `[pending]` placeholders | L3 |
| `references-dope1.bib` | entries added for the DOPE 1.0 manuscript | L3 |
| `references.bib` | starts as a copy of the historical paper's DOI-checked bibliography; add only verified entries | L3 |
| `build.sh` | checks the contract macros and the DOPE 1.0 benchmark surfaces, builds every PDF into `target/paper/`, fails on unresolved references or citations, and requires a generated hero as Figure 1 on page 1 | L3 |
| `scripts/contract_macros.py` | writes `generated/contract-macros.tex`; `--check` refuses drift and any headline-threshold disagreement | L3 |
| `generated/` | numbers, tables and figures rendered from `generated/headline-results.json` | L2 (tools/claims) |
| `generated/equal-bytes-*`, `generated/transfer-per-byte.*`, `generated/readme-equal-bytes.md` | macros, tables, figures and the README block of the equal-byte study, rendered by `docs/whitepaper/scripts/equal_bytes_render.py` and authenticated by the claims firewall | L3 |
| `generated/dope1-*` | figures, tables and macros of the DOPE 1.0 manuscript from `bench/real-world/render.py` (hash-listed in `bench/real-world/results/render-manifest.json`) | L2 (tools/claims) |
| `equal-bytes.bib` | entries copied from the historical bibliographies plus two with DOI or venue URL | L3 |
| `archive/` | review ledgers and readiness notes moved out of the historical paper directory (T-111) | L3 |

## Build

```bash
python3 -B paper/scripts/contract_macros.py   # regenerate the contract macros after a contract change
bash paper/build.sh        # check them, then write main, supplement and the three dope1 PDFs to target/paper/
```

Toolchain: pdfTeX from TeX Live 2023, `latexmk` 4.83 and BibTeX 0.99d, using only standard
packages (`amsmath`, `booktabs`, `graphicx`, `xcolor`, `natbib`, `hyperref`; the DOPE 1.0 manuscript
also uses `amsthm`, `cleveref`, `xr-hyper`, `microtype`, `enumitem`, `caption`, `tikz` and `mathptmx`). The PDFs are build output under `target/paper/`.

## Generated inputs

The manuscript reads these files when they exist and prints a red `[pending]` box when they
do not, so a missing input can never look like a measurement:

| File | Source | Used in |
| --- | --- | --- |
| `generated/dope1-macros.tex`, `dope1-config.tex` | `bench/real-world/render.py`: results (`\G{One...}`) from `summary.json`; settings (`\Cfg...`) from the engine and harness sources | DOPE 1.0 manuscript and supplement |
| `generated/dope1-hero.pdf`, `dope1-tables.pdf`, `dope1-coverage.pdf`, `dope1-families.pdf`, `dope1-audit.pdf` | the same renderer (PNG twins in `bench/real-world/figures/`) | DOPE 1.0 Figure 1 and experiments |
| `generated/dope1-table.tex`, `dope1-equalbytes.tex`, `dope1-ablation.tex`, `dope1-pertable.tex`, `dope1-throughput*.tex` | the same renderer | DOPE 1.0 experiments and supplement |
| `generated/contract-macros.tex` | `paper/scripts/contract_macros.py` from `contracts/tdl-contract-v1.json`: `\TauT`, `\TauD`, `\HeadlinePoint`, `\ByteGrid`, `\SeedDesign`, `\MetricName`, `\ContractId`, `\ContractState`, bootstrap plan | metric definition, protocol, supplement A and D |
| `generated/headline-macros.tex` | `generated/headline-results.json` (T-070): `\ResultStatus` and any printed value | title block, evidence section |
| `generated/hero-frontier.pdf` | the hero figure generator (T-072), from the same CSV as the README image | Figure 1 |
| `generated/result-card.tex` | the result card, one evidence label per row | Table 1 |
| `generated/contract-table.tex`, `byte-ledger-table.tex`, `auditor-registry-table.tex`, `generator-registry-table.tex`, `dataset-dispositions.tex` | contract, charge manifests and registries | supplement |

## Rules

- Follow [docs/claims-policy.md](../docs/claims-policy.md): an evidence label on every table and
  figure, "smallest tested", never an optimum, no privacy claim, and no unqualified auditor in a
  headline.
- Do not type result numbers into the `.tex` sources. Route reviewed inputs through
  the corresponding renderer; frozen headline inputs retain their original bytes and scope.
- The manuscript uses the ICLR layout in final-copy mode; `\iclrfinalcopy` removed gives the
  anonymous review build.
- Abstract numbers are generated macros, each in a sentence that names its evidence status.
  Replace them with sealed-test macros when that run exists.
- Small integers inside mathematics come from generated constants (`\One`, `\Two`, ...) so the
  public lint can keep refusing bare digits.
- The DOPE 1.0 preprint uses the `article` class; `dope1-venue.tex` is a separate target built from
  the same body and must stay double-blind (no names, affiliations or repository URLs).

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
