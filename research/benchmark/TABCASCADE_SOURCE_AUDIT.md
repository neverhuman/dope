# TabCascade author-source admission audit

TabCascade stays `source_audit_pending`. This source audit admits no runtime,
model, fit/sample or tuning job. Official tests remain sealed;
MFS-v2/PTF-v1/release/superiority are null and no DOPE win is counted.

## Source and rights

The [author repository](https://github.com/muellermarkus/tabcascade) is pinned
at `22208aad2b209cdba8df3b61a1a261de46ecc994`.
[tabcascade-source.audit.json](tabcascade-source.audit.json) binds the root
MIT license, untruncated tree, paper and 41 selected source/configuration/lock
files. Each source matches its Git blob and SHA256. The full repository
archive, raw datasets, notebooks, images and pretrained checkpoints were
excluded. Author modules were parsed, never imported. Selected source is not
a complete executable runtime or third-party rights inventory.
Citation: `mueller2026cascaded` from the corrected local bibliography.

## Current author default and paper track

Current code uses the author's Python distributional-tree port, with
`use_R_version=False` explicitly passed by the core encoder. The paper used
the R implementation. The author's equivalence statement is not a study
reproduction receipt. Label the current-author Python and paper-faithful R
tracks separately; freeze any required original-source adaptation and verify
equivalence before claiming reproduction. This study implements neither port
independently and has reproduced no reported experiments.

The released configurations cover DT and GMM encoding. The DT default has
depth 8, joint low/high-resolution training for 30,000 steps, batches of 4096
and 200 generation steps per stage. Terminal EMA weights are used, rather
than a validation-selected checkpoint. YAML facts use `yaml.safe_load`;
dependency-resolved configuration and a campaign default remain unfrozen.

The package requires Python >=3.11 and ships `uv.lock`; installed interpreter,
import roots, bytecode, libraries and dependency/source rights remain unverified.
The optional R path additionally needs pinned R/rpy2/packages and their rights.
No installer or dependency initializer was run.

## Native objectives

The author headline emphasizes detection quality. The released LightGBM
detector returns `1 - (2 * max(0.5, best mean cross-validation AUC) - 1)`,
maximized. It uses five folds, depth 5 and 200 boosting rounds; the
[paper](https://arxiv.org/pdf/2601.22816v3), Appendix A.4, specifies 500 rounds.
The detection source/runtime, validation binding, iteration policy, seeds,
generator grid and ties must be frozen before native tuning. The study has
not selected or executed a scalar objective yet. If no defensible objective
can be frozen, retain the author default and mark tuning inapplicable.

Separately, the author ML efficacy interface reports the absolute gap between
real-trained and synthetic-trained LightGBM RMSE for regression, minimized.
Its default is also 200 boosting rounds versus 500 in the paper. It fits the
target standardizer on real training labels, but categorical encoding includes
real, synthetic and held-out rows. These representation and split choices need
an explicit study binding. Neither author metric may use official tests for
selection, substitute our retention/MFS score, or be ranked against other
methods' native KPI values.

## Execution gates

The core takes supplied tensors; the named-data wrapper creates splits and
learned encoding. Bind the frozen grouped official-training fit/validation
rows, fit preprocessing on fit rows, and verify zero-categorical handling
without artificial columns. The detector has an unconditional categorical
encoder; numeric-only utility subsampling above 100,000 rows also needs a
contract. Missing-value generation must match the selected representation.

Training has no explicit RNG initialization in the core; sampling forwards
constructor seed state to both stages. Verify wrapper fit/sample seeds,
complete finite rows, requested sizes and deterministic replay.

The core checkpoint contains model weights, class proportions, group means,
standard deviations and missing/inflated-group state. It omits the training
loader and original partition objects. It does not include the external model
configuration and data preprocessing required for full table restoration.
Charge all required model/configuration/group-statistics/encoding/projection
files. The loader explicitly uses `weights_only=False`; safe local loading and
round-trip custody remain unverified. Learned state, errors and detailed logs
stay private on scratch.

Copy/leakage/privacy, real-vs-real and fresh aggregate xbabe1/2/3 CPU/RAM/scratch
admission remain required, with the unchanged 600-second fit and 16-GiB GPU
caps. The audit contains no measured benchmark values or formal DP claim.
