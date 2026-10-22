# Historical white paper: DOPE v2 validation

**Frozen 2026-10-15 (T-111).** `dope-mfs.tex` and `supplement.tex` are the historical v2
validation paper, *DOPE: Generative Compression of Predictive Structure*. They are kept
reproducible and are not extended. See the
[**Historical Paper Errata (`WC01`)**](../archive/historical-paper-errata.md) for the
producer-level clarification of the frozen paper's `driver_fidelity`
(`0.905 [0.844, 0.922]` and `0.975` vs. `0.923`) as an absolute feature-target
correlation proxy (`absolute_target_correlation`) rather than holdout
permutation-importance agreement (`D80`, which remains `unmeasured`). It is superseded:
the current paper is the DOPE 1.0 engine manuscript,
[`paper/dope1.tex`](../../paper/dope1.tex) (see [`paper/`](../../paper/README.md)). The review ledgers that used to
sit here are in [`paper/archive/`](../../paper/archive/README.md).

The paper reports DOPE against copula, tree, forest, neural and diffusion baselines on 100
PMLB regression tables. The reported profile is the compact residual
`features12_steps2048` (at most 12 inputs, hidden width 16) at five fit seeds. The shared
measurements are null-normalized retention under three auditors at sizes n and 4n, two
controls, and charged artifact bytes. Every lineage and auditor is selection-exposed, and
the official test split is sealed. The release score MFS-v3 is printed as null. Its
evidence label is `selection_exposed_validation` (see the
[claims policy](../claims-policy.md)).

The current manuscript is [`paper/dope1.tex`](../../paper/dope1.tex). For the TDL manuscript's future
first-page review, [`COMPREHENSION_v2.md`](../internal/whitepaper/COMPREHENSION_v2.md) preregisters five
unaffiliated technically literate reviewers, the questions, and the pass threshold.
No reviews have been collected. [`COMPREHENSION_v1.md`](../internal/whitepaper/COMPREHENSION_v1.md) remains
an earlier, separate historical check and its scores are not pooled with v2.

## Build

| Command | What it does |
| --- | --- |
| `just setup` | creates the Python environment for the paper scripts (`target/venv`) |
| `just paper` | regenerates every `generated/` file, every figure, the README benchmark block and all four PDFs |
| `just paper-check` | re-derives macros from their ledgers, checks figure hashes, and rebuilds through `scripts/paper_build.py --check` |
| `bash scripts/build_pdf.sh` | PDFs only, through the same authenticated driver (`paper_build.py --pdf-only`) |
| `bash ops/ci/paper-clean.sh` | what CI runs: delete `generated/`, rebuild from committed inputs, and require a zero diff |

One build writes four PDFs: `dope-mfs.pdf`, `supplement.pdf`, and the anonymous
submission variants `dope-mfs-anonymous.pdf` and `supplement-anonymous.pdf` (read
with [`../archive/historical-paper-errata.md`](../archive/historical-paper-errata.md)
for the `WC01` association-proxy erratum). Build
evidence stays in the checkout's `target/paper-build/`.

**Toolchain.**

- pdfTeX and `latexmk` from TeX Live 2023.
- The repository's `iclr2027_conference.sty` and `.bst`, with the class running header
  cleared, and natbib author-year citations.
- Python 3 with the packages that `just setup` installs into `target/venv`.

## Inputs

- `ops/ci/paper-review-inputs.json` lists every pinned result publication, ledger and
  receipt the build reads, with byte length and SHA-256. `paper_build.py` refuses a file
  whose bytes differ, or one not admitted by that list.
- `ops/ci/paper-inputs/` holds the immutable source snapshots from `b774807`
  (`inputs.lock.json`, provenance, licenses, the extra bibliography and the sanitized
  compute-cost receipts). `ops/ci/paper-prepare.py` authenticates them and materializes
  them into `generated/`.
- `research/benchmark/review_fixes/paper-build.json` declares each review reduction:
  its script, arguments and outputs.
- The analysis contract is `research/benchmark/review_fixes/predeclare_v2.json`.
- The five-seed v2 panel (fit seeds 11, 23, 37, 53 and 71) is reduced by
  `research/benchmark/review_fixes/v2_panel.py`.

## Figure provenance and expected hashes

Figures are drawn by `scripts/render_figures.py` and `scripts/v2_figures.py` from the
pinned inputs. The README images come from the same `generated/v2-headline.json` as the
paper. The expected hashes are committed beside the outputs:

| Hash file | Covers |
| --- | --- |
| `generated/figure-hashes.json` | the main figure set |
| `generated/teaser-figure-hashes.json`, `generated/pareto-figure-hashes.json` | the page-one teaser and the bytes figure |
| `generated/readme-figure-hashes.json` | `docs/readme/pareto.png` and `docs/readme/paired.png` |
| `generated/architecture-figure-hashes.json` | the architecture figure and its sources |
| `generated/mfs-v3-figure-hashes.json`, `generated/retained-figure-hashes.json`, `generated/published-baseline-figure-hashes.json` | historical MFS-v3, retained-panel and published-baseline figures |

`just paper-check` and the clean rebuild compare the regenerated files with these hashes
and with the committed PDFs. A mismatch fails the build.

## What may change a number

Nothing, while the paper is frozen. Before the freeze, tables were edited only from
committed, hash-pinned validation publications. `research/benchmark/verify_paper_numbers.py`
fails if a printed value disagrees with its ledger. `scripts/local_receipts.py --check`
rejects a changed original or aggregate before its values reach the paper. These rules
also hold:

- Rows from different cohorts stay separate.
- Official test partitions stay closed.
- Empirical privacy text is not a differential-privacy or HIPAA claim.
- A paired validation difference does not certify superiority.

## Packaging

The release package is the four committed PDFs from one build. The scored DOPE
artifacts, the fit and validation partitions and the synthetic-row bundle are not
published. The committed ledgers reproduce every reduction; publishing the artifacts
themselves is Stage 4 work, confirmed by the owner (see
[REVIEW_GAPS.md](../internal/whitepaper/REVIEW_GAPS.md)).
