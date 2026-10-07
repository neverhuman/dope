# DOPE white paper

IEEE journal draft of the generator comparison. The measured DOPE model is
the compact residual `features12_steps2048`: at most 12 inputs, hidden
width 16, no embedding layer. Each method is selected on its own published
objective. The shared measurements are null-normalized retention and charged
artifact bytes, plotted in `figures/retention-bytes.pdf`. Master Fitness
v2 is defined and left null until every required component is present and
every hard gate passes.

## Build

From this directory, with TeX Live and `IEEEtran`:

```bash
pdflatex -interaction=nonstopmode dope-mfs
bibtex dope-mfs
pdflatex -interaction=nonstopmode dope-mfs
pdflatex -interaction=nonstopmode dope-mfs
```

The citation style is numeric, in the sentence, through `\cite`. That is the
IEEE convention: a claim about CTGAN points at Xu et al. as `[n]`, and the
reference list is `references.bib`.

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
