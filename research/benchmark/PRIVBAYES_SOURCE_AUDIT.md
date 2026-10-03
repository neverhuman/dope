# PrivBayes source availability

PrivBayes is unavailable for this study. The original `privbayes-new.zip` release dated 2017-02-18 is pinned by SHA-256 `e8bbd028607f4a393015bb8068648c98786bf880589979a86cb6d2ca1799651e`. Its 13 C++ source/header files match the archive. Eight embedded dataset members were not opened.

No code license grant was found in that archive, the inspected source files or the [author project listing](https://sourceforge.net/projects/privbayes/). The [release listing](https://sourceforge.net/projects/privbayes/files/) identifies the pinned release; the project website redirects to the project listing. This records the inspected evidence and the current study's admission decision. No author code was executed.

The separate [DataSynthesizer implementation](https://github.com/DataResponsibly/DataSynthesizer/tree/b969f618015d6313d03e109ad58aae2f8b12124d) has a verified MIT license at that commit. It has not satisfied the protocol's two reported-experiment reproduction requirement, so it receives no comparison admission or DOPE win.

The original `main_marginal.cpp` reports mean total variation error over all two-way marginals and over all three-way marginals separately, each minimized at fixed epsilon. It does not define a combined tuning scalar. `tools.h` normalizes each count vector and computes half its L1 distance. A future admission must freeze the native selection rule, representation, runtime, fit/sample contract and privacy-accounting disposition before fitting.

The [unavailable receipt](results/privbayes-unavailable.json) and [schema](results/privbayes-unavailable.schema.json) bind the original archive, source inventory and immutable rights proofs. Unavailable cells are excluded from DOPE win counts. MFS-v2, PTF-v1, release-safe L3 and paired superiority remain null. The local bibliography key is `zhang2017privbayes`.
