# GEM-T source and reproduction readiness

GEM-T remains unavailable for this campaign. The inspected scope consists of
the pinned primary paper's landing page and HTML body, three GitHub repository
metadata queries, and one same-name candidate's metadata, commit, tree and README.
The broad name query returned 100 of 5,446 repositories in one page. Exact paper
identifier and title queries returned zero. These are bounded observations;
neither zero exact-query results nor a false API `incomplete_results` flag proves
global source absence. No verified original executable was found in this scope.

The plausible same-name candidate `lugsresdefala/GEM-T` is pinned to
`9bc654075e211f36d5ccb282cac9f6cd500bb23b`. Its metadata and selected README describe
an unrelated application and deployment. It is not accepted as an author or
independent implementation of this method; its license metadata is not used to
judge GEM-T's source rights. The cached paper's GitHub links belong to renderer
support tools. The article's CC BY-NC-ND notice establishes no executable-code
license. This receipt makes no universal or permanent source-unavailability claim.

[The receipt](results/gemt-unavailable.json) binds every inspected primary page,
query response and candidate metadata/README by SHA256 and points to the frozen
private verification. Raw responses and paper text remain on scratch.
Citation: `li2025gem`; [primary paper](https://arxiv.org/abs/2509.17752v1).

Paper sections 3.5.2–3.5.3 describe checkpoint selection using training statistical
quality: separately average column and column-pair similarity scores, then average
those two means. Higher quality is preferred. This identifies the paper objective,
but its executable metric, default implementation, generator grid, ties and
mapping to training-derived grouped validation remain unverified and unfrozen.
The shared DOPE KPI is not substituted as its selection objective.

No author code/runtime or ML initializer is executed; no datasets, samples or
weights are downloaded or read. No independent implementation or either of the
two required reported-experiment reproductions has been verified. Comparison
requires usable original source and full contracts, or an explicitly labelled
licensed independent implementation that reproduces two reported experiments.
Safe generator artifacts, complete inference bytes, seeds/replay, runtime custody
and all five admission locks remain required. No job is launched and this method
contributes no DOPE win or win-denominator entry. Official tests remain sealed;
MFS-v2/PTF-v1/release/superiority remain null. No formal DP or certification claim.
