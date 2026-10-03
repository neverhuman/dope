# REaLTabFormer author-source admission audit

REaLTabFormer remains `source_audit_pending`. This is a static source audit;
no model, runtime, fit/sample or tuning job is admitted. Official tests stay
sealed; MFS-v2/PTF-v1/release/superiority are null and no DOPE win is counted.

## Source and rights

The author repository is [worldbank/REaLTabFormer](https://github.com/worldbank/REaLTabFormer) at commit
`73f239643f9ea5abc877f685ce927e986302ac2d`. [realtabformer-source.audit.json](realtabformer-source.audit.json) binds the
MIT archive, untruncated tree,
license, package version and 15 declared source/package/docs files have matched
hashes and Git blobs. Package version is 0.2.4. Citation key is
`solatorio2023realtabformer`. Author modules are parsed, never imported;
datasets, notebooks, images and weights are excluded. The first 14-file receipt
is preserved separately from the 15-file extension adding package VERSION.

## Released defaults

The required `model_type` is `tabular` for this single-table study. Its released
configuration uses `GPT2Config(n_layer=6)`; tabular fitting initializes model
weights from that configuration, rather than downloading a pretrained model.
Transformer configuration defaults still require a frozen dependency runtime.
Constructor defaults include 1000 epochs, batch 8, gradient accumulation 4,
random state 1029, train size 1, numeric precision 4 and mask rate 0.

Fit defaults include 500 sensitivity bootstraps, quantile 0.95, critic interval
5, stopping history 2 and three generated assessment rounds. Sampling defaults
include batch 128, CUDA device, constrained tokens and continuous-empty limit
10. The `seed_input` argument conditions generated rows; it is not an RNG seed.
A wrapper must verify fit/sample seed handling, row count and finite numeric
outputs across the requested sample sizes.

## Native objective and checkpoint policy

The [author paper](https://arxiv.org/pdf/2302.02041v1) reports regression ML
efficacy as R², higher being better.
The released `SyntheticDataBench` accepts a caller-supplied learner and returns
real/synthetic-trained predictions, rather than a default scalar or generator
search grid. A specific learner/scoring implementation and study tie-break
remain unfrozen. Native values must never be ranked against other methods' KPIs.
If no defensible native objective can be frozen, retain the author default and
mark tuning inapplicable; do not substitute DOPE retention or MFS.

The tabular training path has a separate author sensitivity policy: estimate a
bootstrap threshold from supplied training rows and repeatedly save acceptable
checkpoints below that threshold. The default reloads the latest saved
acceptable checkpoint, with a closest-above-threshold fallback. Optional
closest-to-bootstrap-mean loading is disabled by default. This is not a simple
monotone loss minimizer. That author policy must remain explicit in execution.
Its bootstrap/internal split uses supplied training rows; grouping, the fixed
study fit/validation partition and preprocessing need a verified binding before
selection. Official test rows cannot enter these routines.

## Execution gates

Dependencies have lower bounds rather than exact runtime versions. Before any
author initializer, inventory interpreter import roots, package source/bytecode
and pinned model configuration. Verify compatibility, memory, 600-second neural
fit timeout and process costs on admitted xbabe1/2/3 slots.

Saved state includes model weights and JSON configuration, learned vocabulary
and transformation state. The save routine can copy auxiliary checkpoints.
Charge every file needed to restore/sample, including projection; private
observed tokens and transformation state stay on scratch. Safe local loading,
complete artifact inventory, deterministic replay, row-copy/privacy/leakage and
real-vs-real controls are unverified. The sensitivity policy establishes no
formal DP claim. Author errors and detailed logs stay private.

This audit includes no dataset rows, generated samples, weights or measured
benchmark outcomes. No campaign source, runtime, cap or score is changed.
