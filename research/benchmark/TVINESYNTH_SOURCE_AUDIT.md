# TVineSynth author source audit

The original [author repository](https://github.com/ElisabethGriesbauer/T-Vine-Synth) is pinned at `1fd98c90771ab849d7056bc8917e956c60afd951`. Twenty source/documentation files match their Git blobs and a freshly fetched commit/tree inventory. The main source has an MIT license; the privacy-attack submodule has a separate, still unverified source/license/runtime closure. Bibliography entry: `griesbauer2025tvinesynth`.

## Native objective and applicability

[Appendix E of the author paper](https://arxiv.org/pdf/2503.15972v1) defines a weighted sum of normalized privacy and utility, maximized, with a user-chosen weight. Weight, normalization, sensitive-feature policy, privacy metric, feature ordering, truncation grid, seeds and tie-breaks must be fixed before tuning. The paper permits MAB or membership privacy gain as the privacy component.

The released utility implementation uses random forest binary classification and reports median AUC across 50 synthetic datasets. Its AUC input is thresholded class labels. Tree count is chosen from 100 through 2,000 by minimum mean OOB error; ties can return multiple choices. The released classification metric does not provide an author regression tuner for the S3 regression panel. Generator applicability and any default-only regression track remain pending. No shared DOPE KPI substitutes for native selection.

## Defaults and fit/sample contract

The single-dataset function defaults to `vine_estimation="parametric"`, which enters the `family_set="all"` branch. The multiple-truncation function and author executable use `"par"`, selecting parametric families. The executable uses truncation levels 1/5/10/15/20/26 for its 27-column example and 50 synthetic datasets. These are distinct author entry points, not a frozen campaign configuration.

The function indexes numeric columns with `attribute_names[-factor_position]`; an empty categorical-position vector selects no numeric columns. Pure numeric support needs a verified adapter. The author returns samples, so fitted vine state and empirical inverse marginals require owned artifacts and verified artifact-only sampling. Charge all learned state, metadata and projection bytes.

Author guidance specifies R 4.3.1 and Python 3.10.13, with unpinned R packages. Full runtime verification must precede dependency initialization. Test-named evaluation paths must receive only official-training-derived validation during tuning.

This receipt records source preparation only: no author code or dependency executed, no fit started, no matrix admission, formal DP, reproduction, DOPE win or production certification. Official tests remain sealed; MFS-v2, PTF-v1 and release-safe L3 are null.
