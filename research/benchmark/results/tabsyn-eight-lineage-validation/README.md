# Eight complete TabSyn validation lineages

This panel measures all 48 retained samples from the eight lineages with complete
`n/4n × sample seeds 101/211/307` coverage. Fit seed is 11. The configuration is
the frozen scaled author default; common metrics did not select or tune it.
These eight lineages do not establish coverage of the 100-lineage roster.

All methods use the frozen numeric TRAIN and TRAIN-derived validation inputs.
Official tests remain sealed. The sample CSV representation is already target
last; its one header row is skipped without permuting the columns again. Types
are the frozen TRAIN-only binary-category/continuous rule, rather than inferred
author semantic types. No model weights are loaded during evaluation.

The unchanged A952 evaluator measures per-column KS/TV, continuous correlation
and categorical contingency differences, unembedded alpha precision/beta recall,
five-fold row-group-disjoint CatBoost/logistic detection, DCR against fit and
validation, NNDR and distance membership inference. DOMIAS is an isolated KDE
equation: 36 cells are measured and 12 are explicitly inapplicable because they
include categorical columns. Privacy is empirical, without a formal DP claim.

Utility uses the existing CatBoost, linear and MLP auditors, seed 1729, and
null-normalized `(null − TSTR)/(null − TRTR)` retention. All 144 utility auditors
succeeded. The CSV includes losses and retention. Summaries first average the
three sample seeds within each dataset, then take the median over eight datasets.

| Sample size | CatBoost retention | Mean marginal error | CatBoost detection AUC | Distance MIA AUC |
|---|---:|---:|---:|---:|
| n | 0.885345 | 0.138162 | 0.625104 | 0.578286 |
| 4n | 0.911276 | 0.135998 | 0.669789 | 0.571424 |

These are descriptive validation outcomes. Mean marginal error averages the
declared KS/TV distances; it is not a composite fidelity certification score.
Detection and attack AUCs include neither a release pass nor a privacy guarantee.
MFS-v2, PTF-v1, release-safe L3 and superiority remain null. Charged artifact bytes
are preserved for each fitted artifact, including projection bytes; they must
not be summed repeatedly across its six samples.

`panel.json` binds every small immutable metric receipt and the exact input and
execution records. `figure-kpis.csv` is a reproducible scalar export. Samples and
weights stay on scratch. The execution record pins the actually executed driver
`0d9bce81…`, runtime inventory `8f9189d7…`, and source versions. The committed
driver adds a prospective constant-target fix: zero null loss is uninformative
and yields null retention, and restricts imports to the two frozen metric
implementations while rejecting added metric bytecode. It compiles captured
verified source bytes without reopening the source or cache. All measured null
losses here are positive, so these 48 values are unchanged. Source hashes and versions are verified; this is not a
claim of complete observed interpreter/import or production runtime closure.

The actual evaluation used CPU only, eight cores, four CatBoost threads, one
BLAS thread, nice 19, idle I/O and a kernel 4 GiB memory cap with swap disabled.
Before each sample the worker required at least 15% of physical RAM free after
its whole 4 GiB reservation. The 48 numerical evaluations took 49.574 seconds;
maximum process RSS was 318,398,464 bytes. Original sample-generation clocks
were preserved. Evaluation had its own clock and performed no generator refit.

Regenerate from the mirrored, hash-verified small receipts using
`research/benchmark/publish_retained_metrics.py`. Run
`python3 -B -m unittest research.benchmark.tests.test_retained_metrics -v` to
validate the schema, complete sample groups, scalar export and null claims.
