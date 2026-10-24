# Observed CV landmark inputs

`slot-landmark-observed-bounded-v4` and its same-shape, same-initialization
`slot-landmark-observed-masked-bounded-v4` control consume the explicit
`fit-cv-gain23-observed-mask-squash-v2` recipe. They retain the larger v2 code
widths. Neither graph is a quality-qualified default.

An uninformative landmark is still a measurement when its CV gain is finite.
The new view keeps positive, weak, zero and negative gains from `complete` or
`uninformative` cells. Null, failed, unsupported and over-budget cells are
missing. Each present gain must have the same five-fold training-only witness
and exact declared native per-fold query caps. Values use `gain/(1+abs(gain))`;
23 separate validity bits distinguish a measured zero from missing data.
All-undefined inputs still refuse encoder admission.

Historical `fit-cv-gain23-mask-squash-v1` entries, v3 graphs, checkpoints and
fixtures retain their complete-only masks. A cache recipe must match its model
graph. The graph and cache identity also bind exact checkpoint resume. The
native CLI chooses its reader from the loaded model graph, not an independent
mask switch. The distribution path remains target-free; landmarks condition
only the task half. Sampling requires only code, schema and decoder weights.

## Derive without fitting

For a saved private v1 input directory, create a new directory:

```sh
python3 -B -m research.levels.cli derive-landmark-inputs \
  --input PRIVATE_MEASUREMENT --expected-manifest-sha256 MANIFEST_SHA \
  --expected-engine-sha256 ENGINE_SHA --out NEW_PRIVATE_VIEW
```

This command checks the exact manifest, report, options, fit-only IID scope and
fold evidence, copies the unchanged report and writes a new manifest/features
view. Its producer records original manifest/report digests and zero additional
native executions and optimizer updates. It reads no raw source CSV and never
reruns CV. The original measurement and failure receipts remain immutable.
Dataset admission, exact physical rows and CSV binding are checked again when
the view is consumed; derivation grants no source rights or training admission.

For newly scheduled measurements, `landmark-inputs --feature-recipe
fit-cv-gain23-observed-mask-squash-v2` explicitly produces v2 from the same
training-only native CV panel. The default producer remains v1. A v1 failure on
an already saved all-weak report must be adopted under a reviewed derivation,
not worked around by repeating its fits. Undefined v2 inputs still refuse.

An entire pinned private corpus can be derived with:

```sh
python3 -B -m research.levels.precalc derive-observed-view \
  --source-index-key CORPUS_INDEX_KEY \
  --expected-source-index-sha256 CORPUS_SHA --workers 4
```

The source index and every selected entry/member digest are verified. All
scheduled cases and non-ready dispositions remain in the new denominator.
Undefined reports become explicit unavailable entries; other validation or
publication errors stop the operation. Source v1 feature vectors are checked
against their unchanged recipe before deriving v2. New entries and corpus
indexes live under `feature-views/fit-cv-gain23-observed-mask-squash-v2/` in the
existing private prefix. Writes remain create-only, AES256, SHA256-bound and
read back. No raw source data is added to that prefix.

`python3 -B -m research.levels.precalc download-view --entry-key KEY
--expected-entry-sha256 SHA --out NEW_PRIVATE_DIRECTORY` materializes an exact
selected view. Use the returned pin in an input cache with
`format=dope-fit-cv-landmark-cache-v1` and the explicit v2 `feature_recipe`.
The cache container format stays compatible; its recipe identifies semantics.

The owned benchmark accepts `--landmark-input-cache PRIVATE_INDEX`. An input
graph requires a matching cache recipe; non-input graphs refuse it. The v4
comparison protocol pins cache identity, fit-only scope and separate
precomputation costs. Features attach to the stable owned physical table before
context/query construction, and only the encoder receives them. Per-case
receipts keep feature/manifest digests rather than raw scores. Missing or invalid
table inputs leave both scheduled methods explicitly failed, preserving the
complete matched denominator. Historical v2/v3 benchmark routes remain
available when no feature cache is supplied.

The finite precalculation registry covers canonical fit and five outer-fold
training partitions. Each has inner CV confined to that training partition;
outer validation/query/calibration/confirmation/official TEST labels never enter
landmarks. New arbitrary splits require separate row-bound measurements, and
group/time producers remain unqualified. Reuse keys include dataset/physical
identity, selected training rows, split seed, options, native binary, platform
and producer. The full raw native report remains available to future feature
recipes without another fit.

## Validation and remaining work

Run both independent Python suites:
`research.levels.tests.test_observed_landmarks` and
`research.levels.tests.test_landmark_views`. Tests cover measured zero/negative
values, undefined/nonfinite/corrupted/fold evidence, immutable deterministic
derivation, all-case accounting, cache/graph mismatch, exact CPU resume,
same-initialization controls, target-free isolation and no fit/source calls.
The independent hosted shards pin Linux Torch 2.7.0; native fixtures cover both
v3 and v4, all four tiers at the unchanged 0.002 code tolerance, exact prefixes
and recipient sampling. Tiny public fixtures are operator conformance, not
evidence of model accuracy.

A full-width BF16 GPU exact-resume/native proof and a newly predeclared
landmark-input versus masked-control quality study are still required. No
previous landmark-free study acquires input features retrospectively, and no
frozen negative outcome or acceptance criterion changes. Full real-corpus
training additionally requires reviewed family, rights and typed physical split
metadata. Precalculation does not automatically admit unresolved records.

## Multi-Level Hierarchical Landmark & Descriptor Extension (`DOPE-MLC-v2`)

Building on the observed 23-landmark recipe, `research/levels/descriptors.py` and
`research/levels/multilevel_model.py` implement a unified 3-level hierarchical
landmark and data-descriptor model (`DOPE-MLC-v2`) across **Medium** (9 Macro
landmarks, `4,096` coords), **High** (18 Macro + Larger landmarks, `8,192` coords),
and **Ultra** (all 23 landmarks + 9-coordinate Chatterjee $\xi$ spectrum, `10,240`
coords), paired with mandatory train-only optimal column scaling and categorical
encoding (`research/levels/scaling.py`). See
[Multi-Level Landmark & Data-Descriptor Deep Learning Study](multilevel-landmark-dl-study.md)
for the 19,190-split Stage 0 validation, 19,079-split GPU convergence, and
world-class model transfer and Elo leaderboard (`1680.1` #1 Elo).
