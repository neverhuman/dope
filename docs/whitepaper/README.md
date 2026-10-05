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

Edit tables only from `research/benchmark/RESULTS_STATUS.md` or from a newer
ledger that replaces it. Keep the 97-lineage density panel and the
21-lineage CTGAN/TVAE panel in separate rows. Official test partitions stay
closed. Empirical privacy text in this draft is not a differential-privacy
or HIPAA claim. PTF-v1, release-safe L3, and paired superiority stay out of
the abstract until the frozen gates say otherwise.
