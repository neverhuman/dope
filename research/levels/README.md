# Experimental shared table encoding

This implements the missing learned path from [the multilevel design](../../docs/plans/multilevel-implementation.md).
The existing `fit-levels` portable generator remains a separate implementation.
Shared model weights are fixed local dependencies; every encoded table exports a
bounded FP16 code and charged positional schema. This is development software,
not a certified release, empirical privacy result, external accuracy or Elo claim.

## Implemented boundary

- Target-independent context selection, physical row identities, fit-only mixed
  transforms, explicit target moved last, and disjoint fit/tune/calibration/confirmation
  row roles. Group/time blocks stay together. Training consumes only fit/tune.
- Source-family-disjoint meta train/development/final declarations. The loader
  refuses final-role access and unresolved rights/families. Official TEST is never
  discovered or opened. Oversized tables fail explicitly rather than silently dropping rows.
- Feature attention, induced row attention and learned summary slots; separate
  distribution and target paths retain feature binding. The distribution path
  excludes targets. Graph dimensions and all materializations are bounded.
- Joint nested low/medium/high coordinates of 256/1024/4096 values. Ultra adds
  1024 residual values only after a trained ordinary checkpoint; lower weights
  freeze. Fixed FP16 quantization never derives a scale from higher coordinates.
- Code-only mixed numeric/categorical/missingness diffusion with 32-step DDIM
  by default (8/16/64 explicit alternatives). No context enters the decoder.
- Per-type normalized generation, differentiable ridge and three-step neural
  transfer proxies, grouped prediction-perturbation driver surrogate, masked
  23-landmark global-profile labels and subsample consistency. The one-step soft
  rows used for gradient proxies are an approximation, not exact TSTR evidence.
- AdamW, warmup/cosine schedule, gradient clipping, accumulation and fixed
  development episodes. Nonfinite objectives refuse; unsupported episodes and
  partial accumulation costs are recorded. A partial update is discarded and replayed.
- Atomic tensor-only model/optimizer/Python/NumPy/Torch/CUDA RNG/cursor checkpoints.
  Strict source, recipe and runtime identity is required to resume. No pickle.
- Native Rust/Candle CPU encoder, decoder, carrier and chunked sampler; closed
  operators, strict weight inventories, graph bounds and coordinate identities.
  Native recipient sampling opens no encoder weights or source table.

## Run a bounded GPU development pilot

Use an existing allocation with Python 3.12, torch 2.7.0, NumPy 1.26.4 and
safetensors 0.7.0. `requirements.txt` pins the reference runtime. Install into an
owned environment when needed; do not modify another job's environment.

```sh
python3 -B -m unittest discover -s research/levels/tests -v
mkdir -p target/shared-runs
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python3 -B -m research.levels.cli train \
  --config research/levels/configs/gpu-pilot.json --prior-only \
  --out target/shared-runs/pilot --stop-after 4
python3 -B -m research.levels.cli train \
  --config research/levels/configs/gpu-pilot.json --prior-only \
  --out target/shared-runs/resume \
  --resume target/shared-runs/pilot/checkpoints/step-000000004
python3 -B -m research.levels.cli export \
  --checkpoint target/shared-runs/resume/checkpoints/step-000000012 \
  --out target/shared-runs/model
python3 -B -m research.levels.dashboard --root target/shared-runs --port 18767
```

This small pilot uses owned mechanistic priors and is labeled
`synthetic_development_only`. Training mechanisms 0–6 and development mechanisms
7–8 are distinct; final mechanism 9 is not opened. It does not consume the
private S3 corpus or establish generalization to independent real sources.

Before **each launch**, inspect current GPU processes/ownership and CPU/RAM/disk.
The launcher refuses a competing GPU process, less than 20 GiB GPU capacity,
less than 8 GiB available host memory or less than 10 GiB free disk. It uses nice10,
explicit CPU threads and a 16 GiB CUDA allocator ceiling. Resource admission is
rechecked during accumulation. The single-GPU guard supports visible device 0;
other devices and DDP are explicit future allocations, not automatic fallbacks.
It never kills a competitor or modifies services. The wall allowance includes
initialization and validation, and is checked between bounded episodes; an in-flight
kernel/episode can finish after the boundary. Use an external scheduler deadline
if a hard process wall limit is required. Failed or stopped attempts retain receipts.

To start ultra, use a **new** run and the same graph/corpus/source identity with
`--phase ultra-residual --ordinary-base PATH_TO_ORDINARY_CHECKPOINT`. To resume
that residual stage, use its own checkpoint with `--phase ultra-residual --resume`.
This is a frozen-base development experiment; it does not qualify a new tier.

The learned sampler defaults to `reveal-v1`: categorical cells start masked,
each still-masked cell reveals with probability `1 / remaining_steps`, and
revealed categories stay fixed. This matches the training mask schedule and
guarantees completion at the final step. Python `sample` and native
`dope learned sample --categorical-schedule reveal-v1` use the same transition
rule. Their random streams differ; bit-identical cross-backend draws are not
claimed. `redraw-v1` preserves the initial experimental sampling recipe for
replaying frozen runs. This changes only the experimental learned sampler.

## Repeatable owned development comparison

Use a frozen trained package and an explicitly checksum-pinned previous native
generator. This command opens only owned mechanisms 7 and 8, using the same
physical fit/query roles, six tables, three sampling seeds and sample counts for
both methods. The old generator receives a whole-file byte cap equal to the
selected shared code including its positional schema. Shared weights and model
metadata are charged separately; training compute is not matched.

```bash
python3 -B -m research.levels.benchmark \
  --model target/shared-runs/model --baseline /absolute/path/to/previous/dope \
  --expected-baseline-sha256 FULL_VERIFIED_BINARY_SHA256 \
  --out target/shared-runs/comparison --levels low,medium,high,ultra \
  --categorical-schedule reveal-v1 --device cpu --wall-seconds 600
```

Request ultra only after exporting its trained residual checkpoint. `all` means
S/M/L. The protocol is saved before generating any development tables. Every
scheduled method/table/level/seed has a retained complete or failed record;
missing or duplicate cells refuse summary generation. The independent readout
is deterministic NumPy64 ridge, with regression MSE or binary Brier loss.
Uninformative real-model gains retain null retention, and sample seeds are not
treated as independent datasets. Paired loss deltas are descriptive; this small
development experiment never qualifies superiority, external accuracy or Elo.
The historical Candle neural auditor's ignored CPU device seed remains outside
this readout and requires separate qualification before neural replay claims.
Keep an external process deadline around the command if a hard wall is needed.

For an additional native control panel, pass `--auditor-engine /absolute/path/to/dope`
and `--expected-auditor-sha256 VERIFIED_SHA256`. Protocol v3 records this separate
binary and uses `linear-seeded-v1,gbdt,mlp-seeded-v1` with the same fit-only baseline
artifact representation for both methods. Native MSE/classification log loss is
auxiliary to the unchanged ridge/Brier primary metric. Every requested native
auditor must complete; an auditor failure retains a failed scheduled cell. The
ordinary protocol v2 path remains available without these options.

The new CPU neural recipe `candle-mlp-host-init-v1` generates weights and biases
with the repository's counter RNG and orders optimizer/early-stopping variables by
name. It uses no Candle global device RNG. Replay is tested across fresh
processes, interleaved seeds and one/four CPU thread pools on a pinned runtime;
bitwise equality across architectures is not claimed. The historical `mlp` and
`linear` names and default auditor roster retain their behavior. No new native
auditor output retrospectively qualifies the frozen ridge cohort or driver
fidelity; independent driver and external-source evidence remain required.

## Real data and native landmark corpus admission

Replace `--prior-only` with `--corpus PRIVATE_MANIFEST.json`. A
`dope-meta-corpus-v1` manifest contains `datasets`, each with:

- `family_id` resolved across derivatives; `meta_role` train/development/final;
- `source_role: official_train`, `rights: {admitted: true, license_id: ...}`;
- stable opaque `dataset_id` identifying physical row offsets independently of
  content, local `path`, `train_sha256`, `has_header`, explicit source `target_index`,
  source-order `column_kinds`, and `task` regression/classification;
- `split_policy` iid/group/time, `split_seed`, plus `group_index`/`time_index`
  where applicable. Time requires explicit `time_kind` numeric/iso8601. ISO
  timestamps must carry a timezone; numeric chronology is never lexical order;
- optional `landmarks` with local immutable `manifest_path`, `manifest_sha256`,
  `report_path`, and checksum-pinned `engine_sha256` from the role-scoped producer below.

Whole-TRAIN precalculation reports are diagnostic evidence; they are **refused as
gradient labels**, because they include reserved calibration/confirmation rows.
After source admission, generate a separate private fit-only artifact in a new
directory. The dataset JSON is one admitted dataset entry from the corpus manifest:

```sh
python3 -B -m research.levels.cli landmarks --dataset PRIVATE_DATASET.json \
  --engine /absolute/path/to/dope --expected-engine-sha256 VERIFIED_BINARY_SHA256 \
  --out target/private-fit-labels
```

The producer opens only that declared official TRAIN file, selects physical fit
rows before invoking the native panel, removes entity/time predictor identifiers,
and writes a target-last positional CSV, plan, report and manifest (manifest last).
Its reader recomputes the exact input and physical-row scope and checks source,
schema, split, recipe, binary and report hashes. Unsupported cells remain masked;
failed attempts retain their plan/report and static failure receipt. Directories
are private and existing outputs are refused. This first scope supports IID only:
group/time profile labels refuse until their native fold policy is qualified.
Real episodes without profile labels still support all declared split policies.
Content hashes bind source lineage but never seed row-role selection; changing
held-out values cannot reshuffle fit rows. Preserve the stable dataset identity.
These are new derived labels with separate lineage; never overwrite or relaunch
the finished whole-TRAIN corpus. No actual completed pilot used real profile labels.

No automatic inference of family, license or target is an admission decision.
The completed private precalculation index contains 3,266 reusable reports,
441 unsupported inputs and 152 native width refusals over 3,859 TRAIN records.
Keep every outcome; do not turn unresolved catalog metadata into training approval.
Use global profile gains/masks only: native spectrum feature indices refer to
inferred frame slots, not a verified raw-column driver mapping. Complete labels
and explicit masks are distinct; unsupported/uninformative slots are not zero gains.

A real corpus alternates real/prior episodes and balances families before dataset
members. At most two real tables are cached; a CSV is bounded at 200 MB,
200,000 rows and 2,000,000 cells. Inputs outside that envelope need a separately
versioned block/streaming preparation stage. There is no silent full-corpus claim.

## Native frozen encode and source-free sample

```sh
cargo +1.88.0 build --manifest-path engine/Cargo.toml --locked -p dope-cli
engine/target/debug/dope --threads 4 learned encode \
  --model target/shared-runs/model --csv PRIVATE_FIT_CONTEXT.csv \
  --target target --kinds numeric,categorical,numeric --task regression \
  --levels low --max-bytes 64KiB --out target/low.code
engine/target/debug/dope learned inspect \
  --model target/shared-runs/model --code target/low.code
engine/target/debug/dope --threads 4 learned sample \
  --model target/shared-runs/model --code target/low.code \
  --level low --rows 128 --seed 0 --steps 32 --out target/sample.csv
```

The input CSV must already be the authorized fit/context role. Kinds refer to
source order and the target is explicitly declared. The native reader selects
at most the graph's context cap by seed/physical row position, fits transforms
on those context rows only and retains their local physical offsets. No query
file is accepted. `--target-free` removes the target schema and task coordinates.
Selected exports serialize/hash only the admitted prefix. `all` is low/medium/high;
ultra is always explicit. Outputs publish atomically without overwriting a path.

Categorical outputs are positional **category IDs**, not reconstructed source
labels. Labels and raw headers never enter the code. Reconstructing original
category strings needs an explicitly authorized, charged mapping in a future
schema version. Numeric robust centers/scales are charged code metadata and
are a disclosure decision; this format does not promise de-identification.

A recipient directory needs only `model.json` and `decoder.safetensors` plus
the code. Encoder weights can be physically absent. `inspect` reports complete
code bytes and separate fixed encoder/decoder dependencies. Compare warm marginal
code bytes and cold dependency bytes separately. The reference Python and native
samplers use distinct RNGs: coupled operator parity is tested, bitwise random
sample parity between backends is not claimed.

## Evidence gates and remaining plan work

Conformance proves engineering properties, not the design's E0–E11 scientific
success conditions. The tiny tracked `engine/dope-core/tests/fixtures/learned-v1/`
weights are trained on one public synthetic optimizer update solely for operator
parity; regenerate with `python3 -m research.levels.fixtures --help` in the pinned
reference runtime. They are not production weights or private checkpoints.

Before a large campaign or promotion, resolve/admit the real source families and
rights, freeze development/final families, bind independent exact native auditors
and driver groups, and precommit matched data/seeds/training/selection budgets.
Required comparisons still include independent heads, equal-byte higher-width
controls, learned fusion, supported S/M/L and explicit ultra, and old-generator
comparisons. Exact tree audit is `not_run` in the gradient proxy; longer unroll
sensitivity and independent confirmation remain unmeasured. These controls and
external accuracy/Elo evaluation are pending, not inferred from falling loss.
Optional codecs, program embeddings, teachers and adaptive proposals remain
separate evidence-gated tickets in the master plan. Historical DPK3.2/v1 readers,
metric contracts and product defaults are unchanged.

The Linux CI conformance workflow uses the existing `neverhuman-lan` pool with
trusted pushes to main/codex branches and manual dispatch. Independent stdlib, PyTorch and X64/ARM64 native
shards can run concurrently. Thirteen slots are configured across xbabe1–4;
actual availability is owned by their existing controllers, not inferred from
configuration. GPU allocations remain separate from these CPU runners.

### Neural proxy unroll controls

Training JSON accepts `neural_unrolls` from 1 to 64 (default 3). Both the real
reference fit and differentiable synthetic fit use the same explicit count,
initialization, learning rate and regularizer. Completed episode records include
the actual count per tier, or null when the proxy did not run (including ramp
step zero). The checkpoint recipe binds it: changing the count refuses exact
resume. A warm-start stage explicitly records its ordinary base and new recipe
and resets the optimizer/RNG; it is a separate experiment. Use a frozen source
snapshot and fresh matched initialization for controlled comparisons.
Longer unrolls increase differentiation cost; retain
the existing allocation and external deadline, including failed attempts.

These are gradient approximations. A stronger real/null proxy ceiling does not
establish improved shared-model samples or parity with an exact native learner.
Independent neural/tree/driver evidence remains required.

### Versioned prior exposure controls

The default `owned-mechanisms-v1` corpus is retained for replay. In prior-only runs,
its two adjacent cursors reuse a physical table, so episode counts are exposures,
not independent table counts. The opt-in training flag
`--prior-recipe owned-mechanisms-unique-v2` gives each prior-only **training** cursor
a distinct family/seed table, cycling the same seven training mechanisms with the
same 256-row/eight-predictor shapes. Fixed development draws and the mixed
50/50 real/prior stream stay identical to v1; no source is admitted by this option.

Both recipes record opaque physical-table IDs and source provenance in completed
`episodes.jsonl` rows. Real records also bind their content digest. Count unique IDs
across the recorded resume lineage to distinguish new tables, repeated exposures
and physical rows; a cursor-dependent episode hash is not a unique-table count.
The corpus identity binds the explicit recipe. Cross-recipe resume or warm-start
is refused. A new source snapshot and fresh initialization are required for a
controlled exposure comparison; do not relabel previous checkpoints or imply
that this engineering control improves accuracy.

### Distinct physical development panels

The historical default `fixed-six-v1` keeps its six episode draws and
recipe representation for replay. Those draws cover **three distinct physical
prior tables**, each observed twice: four classification and two regression
exposures. Episode counts are not independent source counts.

For a future prior-only run, training JSON can explicitly set:

```json
{"validation_recipe": "owned-balanced-unique-v2", "validation_episodes": 40}
```

This opt-in observer uses twenty distinct owned classification tables and twenty
regression tables from development mechanisms 7 and 8. It accepts an even count
from 2 to 64. The training corpus and its physical stream are unchanged; the new
observer is bound into the checkpoint recipe, and changing its recipe or count
refuses exact resume. Real/mixed corpora refuse this owned-prior observer; they
need a separately admitted family-balanced panel rather than a silent fallback.
Existing frozen experiments, plots and benchmarks retain their original monitor.
The independent source fingerprint still changes with this implementation; replay
old checkpoints with their frozen source rather than bypassing source checks.

Each new validation event records its scheduled/completed episodes, distinct
physical tables, opaque family counts, task counts, context/query lineage hashes,
wall cost and panel digest. Only fit context and tune queries from meta-development
sources enter the observer. Calibration, confirmation and final sources stay out.
The observer uses no gradients or optimizer updates and restores the training RNG
and model mode. Draws and corruption RNG are fixed across checkpoints; the number
of physical tables, not the number of plotted observations, describes coverage.
The dashboard displays that coverage when it is recorded; historical metrics
without lineage remain explicitly unavailable.

Validation costs count against the existing run allowance. Time and resource
admission are checked before each episode; an in-flight episode can finish after
the boundary, so retain an external hard deadline. An interrupted panel publishes
its disposition and completed count with **no partial mean**, saves the last
completed optimizer boundary and never retries automatically. More monitor tables
increase overhead and do not establish accuracy, independent real-source coverage,
or permission to select a favorable checkpoint. Predeclare any selection rule and
keep every scheduled benchmark checkpoint and negative result.
