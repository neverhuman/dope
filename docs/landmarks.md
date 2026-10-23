# Native landmarks and controlled portable levels

This milestone implements a **Rust CPU landmark panel, private transfer references,
portable level selection, and artifact-derived multilevel descriptors** on top of the
existing format-1 generator. It does not ship trained universal encoder weights or assert
improved external retention, driver transfer, Elo, privacy, or single-inference speed.

The complete research design is in [the multilevel plan](plans/multilevel-implementation.md)
and [the supplied landmark specification](plans/landmark-signatures.md). Those plans remain
useful work orders; this guide describes the behavior that is actually implemented.

## Commands

Build with the pinned Rust toolchain:

```sh
cargo build --manifest-path engine/Cargo.toml --release --locked
DOPE=engine/target/release/dope
```

Describe a real **fit-role file**, with the target declared explicitly. Do not supply
calibration, confirmation, external evaluation or mixed-role files to this command:

```sh
$DOPE landmarks real fit.csv --target price --task regression --json -o private-real.json
$DOPE landmarks real fit.csv --target outcome --task classification \
  --group customer_id --json -o private-grouped.json
```

`--group` is removed from predictor features before schema inference. Equal groups/time
blocks stay in one fold. This is blocked cross-validation, not forward-chaining forecasting
validation. Fit-levels does not yet accept group/time roles; use it only for IID pilot data.
Missing target labels are rejected, rather than silently changing the scheduled population.

A source-free signature opens only the artifact:

```sh
$DOPE landmarks synthetic model.dope --rows 4096 --json -o synthetic-signature.json
```

Compare a fitted artifact against a clean real reference on a disjoint query file:

```sh
$DOPE landmarks compare model.dope --train fit.csv --query tune.csv --json -o private-transfer.json
$DOPE landmarks compare model.dope --train fit.csv --query tune.csv --broad --json -o private-panel.json
```

The reference schema and marginal imputation are fitted on `--train` only. The query never
fits either model. The CLI rejects identical rows across train/query after header alignment,
missing normalization and numeric text normalization. This conservative duplicate check is
not a substitute for source-lineage/group isolation. It cannot detect different records from
the same person, machine, household or source family. An unknown query target class fails;
it is never dropped. Missing classes in synthetic data keep the real reference class columns.

Create portable low, medium and high programs, but release only low:

```sh
$DOPE fit-levels fit.csv --target price --compute-through high \
  --export-levels low --out release-low --private-report private-session.json
```

Release all three ordinary levels and descriptor vectors:

```sh
$DOPE fit-levels fit.csv --target price --compute-through high \
  --export-levels all --descriptors --combine --out release-ordinary
```

Ultra requires both compute and release permission:

```sh
$DOPE fit-levels fit.csv --target price --compute-through ultra \
  --export-levels ultra --descriptors --out release-ultra
```

`all` means exactly low, medium, high. `small` aliases low; `med` aliases medium; `large`
aliases high. Duplicate aliases are rejected. Exporting an uncomputed level is an error.
A session needs at least 100 rows. Whole standalone generator caps are 2,048 / 4,096 /
16,384 / 65,536 bytes. These are complete artifact caps, not parameter counts, embedding
sizes or total release-directory caps. Explicit descriptors and the inventory add charged
bytes to the release. Very wide or label-heavy tables may not fit the low cap; the command
fails rather than violating the cap or silently omitting that stage.

## Runtime layout and responsibilities

| Area | Implementation |
| --- | --- |
| Panel identity and ordered recipes | `engine/dope-core/src/landmark/panel.rs` |
| Options, limits and seed separation | `landmark/mod.rs` |
| Fit-only raw-unit frame and fold-local robust representation | `landmark/frame.rs` |
| Physical-identity/class-stratified and group-blocked folds; deterministic caps | `landmark/folds.rs` |
| Numeric shape expansions, quantile knots, pair products, RFF, per-feature slicing | `landmark/basis.rs` |
| Shared multi-output Cholesky solves and nested ridge regularization | `landmark/ridge.rs` |
| Existing native tree/GBDT/network adapters; exact projected kNN | `landmark/learners.rs` |
| Ordered parallel panel, uncertainty, feature spectra | `landmark/signature.rs` |
| Typed cells, stability masks, contrasts, explicit private/global views | `landmark/report.rs` |
| Deterministic operation estimates and drop order | `landmark/cost.rs` |
| Cached real transfer reference, candidate scoring, paired bootstrap | `landmark/transfer.rs` |
| Portable fit/tune pilot, carry/replace, sample remapping, duplicate guard | `engine/dope-core/src/levels.rs` |
| CLI adapters and manifest-last release publication | `engine/dope-cli/src/{landmarks,levels}.rs` |
| Frozen layout and JSON coordinate schema | `landmark/layout-native-v1.json`, `schemas/landmark-coordinates-native-v1.schema.json` |

The engine adds no Python, subprocess, libtorch or GPU requirement. It reuses the existing
native evaluators, oblivious-tree booster, residual-network trainer, Chatterjee statistic,
CPU sampler and artifact reader. SHA-256 binds complete generator bytes, descriptor bytes
and the released schema. A digest provides integrity binding, not model authenticity.

## Measurement protocol

`LandmarkFrame::real` accepts only explicitly selected fit row indices. It strips names,
recovers numeric original units from the schema map, preserves categorical slots and checks
shape, target codes and unique physical identities. No row labels or source column names
enter its error text. `from_columns` accepts already typed original-unit numeric columns
and integer categorical codes with stable identities. `synthetic` builds an independent
frame from a program's samples; it needs no real table or private report.

Each outer fold fits robust numeric centers and scales on its training rows: median and
IQR / 1.349, standard deviation then 1 as fallbacks, clipping to ±5, and zero plus an indicator
for missing numerics. Categoricals use training-frequency-selected levels and an other
column. Category ordering never uses a target. Numeric target centering/scaling is also
training-only. Classification uses declared binary/multiclass columns and Brier loss for
signatures. Folds share preprocessing and each recipe receives a domain-separated RNG seed. Independent
fold jobs run in parallel even when work pruning retains just one recipe; results are
collected and reduced in fold order. Unused small-learner fold frames are not built.
Training rows and small-learner query rows are capped using stable identity hashes. The
default ridge cap is 8,192 rows per fold (raise `--ridge-rows` to at most 50,000);
`--small-rows` and `--knn-reference` expose the other bounded fitting controls.

For ridge, the outer training fold supplies an internal 80/20 split to select among
relative penalties 1e-4, 1e-2 and 1. The selected penalty is refit on all outer-training rows.
A Gram is accumulated in fixed row order; one Cholesky factor serves every class output.
Outer query labels never choose the penalty, features, knots, bandwidth or weights. Fold-local
normalization prevents general exact fold downdating: this implementation rebuilds Grams
rather than reusing an invalid global normalization. Basis widths above 511 are typed as
unsupported; encoded widths above 512 produce a bounded budget error.

The panel has the proposal's 23 ordered landmark IDs: linear, asinh, signed-root, polynomial,
saturation, absolute value, hinge, step, periodic, bump, row aggregates, pair products, random
Fourier features, missingness, best single feature, stump, depth-three oblivious tree, micro
GBDT, 1-NN, 16-NN, residual network and two quarter-data learning curves. An unsupported,
failed, over-budget or uninformative recipe occupies its original slot; it never becomes a
perfect score. Gain is pooled held-out improvement over the fold-fit constant predictor.
SE is the sample standard deviation of fold gains divided by sqrt(K). A cell is unstable
when SE exceeds max(0.02, 0.25 * abs(gain)); unstable cells remain reported but are masked
from global coordinates and contrasts.

Budget drops follow a fixed order beginning with quarter-data/tree/network and secondary
shape recipes; linear is mandatory. A budget too small for linear returns an error. The
work report gives estimated units, attempted cells, actual training/query counts and omitted
landmarks. Feature spectra consume the remaining budget and report how many slots ran.
Operation units are deterministic scheduling estimates, not a wall-clock deadline or a claim
that every counted primitive has identical cost.

### Why this has a distinct version

The supplied proposal names a future exact `landmarks-v1` recipe. This implementation is
**`landmarks-native-v1`**, with these deliberately explicit differences:

| Subject | Native-v1 behavior |
| --- | --- |
| Ridge penalty selection | Nested within outer-training rows; never chosen from reported query losses |
| Gram reuse | Shared across lambda solves and outputs within a design; rebuilt per fold for fold-local normalization |
| Numeric expansion selection | At most 64 numerics, ranked by fold-train Xi only when needed; pair features use Xi for their top eight |
| Periodic rank map | Training empirical rank map; binary numeric inputs add no periodic terms |
| Width limits | Whole basis >511 is unsupported; no silent partial-basis truncation |
| RFF | 256 features, seeded Gaussian frequencies, median sampled positive pairwise distance |
| Network | First 32 encoded columns, hidden 16, up to 300 steps; training-only stopping |
| Feature spectra | First up to 64 original slots, budgeted; Xi is an in-role descriptive coordinate |
| T03 heteroscedasticity | Always masked; it is not approximated with in-sample residuals |
| Global storage | 96 f32 coordinates plus explicit presence mask and complete recipe options in JSON; no f16 export claim |
| Transfer selection mirror | Exact existing linear/tree prediction implementations, in a fixed fit-only auditor representation; neural promotion remains gated |
| Transfer stability | Delta-method SE of real-reference gain from paired query-row losses; not repeat-CV stability |
| Confirmation and learned encoders | Not implemented by the portable pilot; reports say `not_run` / `not_measured` |

Bind the exact panel **and all options**, artifact hash and schema hash when training a
consumer. `GlobalCoordinates::decode` refuses mismatched versions/options, dimensions,
nonfinite numbers and invalid masked storage. Equal width is not sufficient compatibility.
Do not relabel these vectors as the proposed recipe. New algorithms/constants require a new
panel identity and retraining or an explicit validated migration.

## Transfer reference and selection

`TransferReference` encodes fit and query rows once under the real fit schema/marginals and
caches real-trained predictions. Every synthetic candidate is decoded to raw units/labels
and remapped into that same representation before fitting auditors. Remapping is direct
f64 arithmetic with label-index translation, avoiding CSV formatting/rounding and parser overhead. Core predictions call
the exact functions already used by self-audit; the new reference changes the data roles,
not those learner implementations. Real and synthetic training counts are capped at 4,096
by default. Class columns are fixed by the real fit schema.

Transfer uses the protocol's MSE/log-loss, rather than the descriptive signature's Brier
space. Null predictions come from real fitting rows. A cell is informative only when real
gain is positive and at least max(1% of null loss, 25% of the best reference improvement).
Core informativeness uses core references only: enabling the broad diagnostic panel cannot
alter the mirror's score. Stable informative core families determine the mirror minimum.
The broad family aggregate is diagnostic only; 1-NN never contributes to it because it is
a copying/memorization probe. Failed cells are recorded, not assigned perfect retention.

Private per-query-row null/real/synthetic losses support paired bootstrap. Each replicate
resamples identical query indices for both candidates and recomputes informativeness,
stability, weighted family aggregates and their minimum. It never bootstraps an already
collapsed minimum. Comparisons reject different reference identities; at least 90% of draws
must yield an informative aggregate for an interval to be returned. These intervals assume
independent query rows, not group/source-family independence.

The ordinary `fit` compiler's default audit and artifact format are unchanged. `fit-levels`
uses a new explicitly experimental `portable-levels-pilot-v1` protocol:

1. Freeze a deterministic 80/20 fit/tune partition, with query size capped at 5,000. Reject
   cross-role duplicate rows. Infer the reference schema on fit only.
2. Reuse one fixed real transfer reference for the session.
3. Compile one independent proposal per level with its own domain-separated seed and file cap.
   Existing internal fit selection sees only fit rows, not session tune rows.
4. Retain the previous selected standalone bytes when the new proposal lacks an informative
   improvement above 0.001. This is tune-selection, not a confirmation decision.
5. Return only allowlisted outputs. Do not return higher-level bytes/scores in public inventory.

There is no calibration or bounded confirmation stage yet; candidates are not advertised as
certified noninferior. There is no joint learned inference, shared trained table encoder,
learned fusion, progressive delta artifact, new decoder capacity, driver-transfer ranking or
new external benchmark result in this milestone. Existing fit options that force an over-cap
program are caught again at the level release boundary. Library callers requesting a custom
single budget or anonymized session are rejected by this pilot rather than silently ignored.

## Release custody and accounting

`Signature`, `TransferSignature` and `LevelSession` deliberately do not implement blanket
`Serialize`/`Debug`. Callers must choose `private_report()` or an explicit global-coordinate
view. Real signatures are never attached to `fit-levels` output. The CLI prints a notice on
real and transfer commands. Private reports contain measurements and must be stored under
private access controls even though their field names contain no original headers.

An exported descriptor is computed only from that exported program using 4,096 fixed-seed
samples and a fixed descriptor recipe. Low's descriptor cannot see medium/high/ultra state.
`--combine` concatenates only authorized descriptor blocks, with masks, level boundaries and
source hashes. It is a reproducible input baseline for future models, not a demonstrated
accuracy gain. A medium-only release emits only medium, even when its selected standalone
program carries a lower proposal. The generator has no external decoder dependency.

The destination directory must be new. Publication exclusively creates it, writes new files,
syncs their contents and writes `inventory.json` last. Existing files, directories and symlinks
are not overwritten. The inventory enumerates exact object bytes, SHA-256 hashes, file caps,
schema hashes, descriptor source hashes, payload bytes and the total including inventory.
Readers must require the inventory and verify hashes before using any member. This is
manifest-last publication, not an atomic directory rename or a durable transaction across
power failure. A killed process can leave an incomplete directory; discard it if no valid
inventory is present. Ordinary write failures remove only files created by this invocation.

The optional private report must be a new file outside the release directory. Public objects
contain no candidate report, training row counts, real landmarks, tuning curves, computed
higher-level inventory, local paths, source extrema or raw-row identities. Schema names and
class labels already present in an ordinary generator remain part of that generator; this
pilot does not add an anonymized format. Integrity hashes do not prove formal privacy.

## Validation and next engineering work

Run the dedicated gate locally:

```sh
bash ops/ci/landmarks.sh
bash ops/ci/engine.sh
```

The new CI workflow pins Rust 1.88, uses the repository's trusted disposable runner class,
and runs the same formatting, warnings-as-errors lint and focused integration tests. The
existing CPU gate still discovers all native tests. Tests exercise all 23 ordered recipes,
linear and class signal, no-signal masks, invalid shapes/codes/options, group isolation,
role mutation, version refusal, exact thread-count repeatability (1/2/8 threads), source-free
synthetic execution, duplicate refusal, paired comparison and lower-level byte independence.
CLI tests verify private notices, header canaries, release allowlists, complete byte accounting
and overwrite refusal. These are development fixtures, not the sealed external corpus.

Before promoting a learned system or changing default audit selection:

| Next work | Required evidence |
| --- | --- |
| Exact proposed panel, T03 and fold-downdating optimization | Independent reference goldens, fold-local transform proof, planted-mechanism E-L0 and timing E-L8 |
| Five-role shared compiler and confirmation | ML-01/03 row-role/schema mutation tests, frozen candidate graph and bounded confirmation receipts |
| Stronger mirror including a stable neural family | E-L1 on >=50 development tables and source-family paired external selection regret |
| Corpus J-L and leaderboard integration | Complete scheduled-cell records, failed-cell denominators, role/lineage manifests, private measurements kept private |
| Shared table encoder and decoder | ML-06/07/08 corpus and checkpoint, native package authenticity, target-free H_X invariance and E-L3 ablations |
| Multilevel learned fusion | Equal-total-byte single-code/concatenation controls and held-out model-selection/transfer gains |
| New generator capacities or transforms | ML-09/10 format admission and E-L5/E-L6 held-out learner/driver gates |
| Default changes and #1 Elo claims | Measured leader roster, driver transfer, lower-tail coverage and confidence intervals; no claims from these unit tests |

Do not collapse unrun research gates into successful acceptance checks. The supplied design
and this implementation guide together form the work order for the next agent.
