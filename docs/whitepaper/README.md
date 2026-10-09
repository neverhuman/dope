# DOPE white paper

IEEE journal draft of the generator comparison. The measured DOPE model is
the compact residual `features12_steps2048`: at most 12 inputs, hidden
width 16, no embedding layer. Each method is selected on its own published
objective. The shared measurements are null-normalized retention and charged
artifact bytes, plotted in `figures/retention-bytes.pdf`. Master Fitness
v2 is defined and left null until every required component is present and
every hard gate passes.

## Build

From this directory:

```bash
bash scripts/build_pdf.sh
```

The manuscript is a single-column `article` with `iclr2027_conference` and
Times. Citations are natbib author-year (`\citep` / `\citet`). The class
running header is cleared. `references.bib` is the reference list.

## What is allowed to change a number

Edit tables only from committed, hash-pinned validation publications. The
retained panels merged on 2026-10-07 are reduced by
`scripts/retained_evidence.py`: TabSyn's eight-lineage scaled-default
subset, classical default/native samples on 100 lineages, and the
authenticated deterministic first12 diagnostics. Generated source digests,
JSON pointers, lineage means, intervals and paired differences are in
`generated/retained-evidence.json`; coverage and conditional ETAs are in
`generated/baseline-coverage.json`. Original receipt paths are in the
referenced benchmark publications. `just paper` regenerates these inputs
and their figure before building the PDFs; build evidence stays in the
checkout's `target/paper-build/`.

Keep the 97-lineage density panel and the
21-lineage CTGAN/TVAE panel in separate rows. Official test partitions stay
closed. Empirical privacy text in this draft is not a differential-privacy
or HIPAA claim. PTF-v1, release-safe L3, and paired superiority stay out of
the abstract until the frozen gates say otherwise. The new mean-of-three
retained panels and legacy median-of-three blocks stay separate. A paired
validation difference does not certify superiority.

## Original metadata and seed scope

`research/benchmark/results/paper-original-metadata-v1/index.json` binds
210 original fit receipt/report pairs and compressed numeric replay traces,
plus safe preparation outcomes and a filename-only provenance observation.
The capture date is explicit; earlier historical custody is not reconstructed.
`generated/original-field-map.json` records file hashes and field selectors.
The loss PDF and bootstrap summary now regenerate from committed inputs on
machines without campaign scratch. `local_receipts.py --check` rejects a
changed original or aggregate before its values can enter the paper.

Fit seed 11 describes the primary displayed DOPE and legacy comparisons.
The retained ARF and Forest-Flow five-fit appendices have their own coverage;
their additional seeds do not establish DOPE fit variance or a release score.
