# Model size, pretrained artifacts, and a scalable ranking study

DOPE compiles a distribution for **one dataset and one declared target**. The existing
byte allocator is the right foundation for giving users more capacity when lift matters
more than size. This change adds a second hard cap, named presets, portable pretrained
artifact packaging, and a reproducible diagnostic leaderboard. It does not change any
frozen TDL, privacy, or research-kernel contract.

## User-selected capacity

```bash
# Fit within the complete-file cap, including schema and safetensors framing.
dope fit train.csv -t price --budget 12KiB -o prices.dope --report fit.json
# Constrain BOTH the artifact and the complete generator's scalar state.
dope fit train.csv -t price --budget 32KB --max-parameters 4000 -o prices.dope
# Familiar capacities; explicit --budget overrides the preset's byte cap.
dope fit train.csv -t price --size small
dope fit train.csv -t price --size medium --max-parameters 2000
dope fit train.csv -t price --size large --budget 24KiB
dope inspect prices.dope --json
```

| Preset | Whole artifact maximum | Intended starting point |
|---|---:|---|
| Small | 2 KiB | Minimal distribution to ship |
| Medium | 4 KiB | Existing default, balanced capacity |
| Large | 16 KiB | More capacity for trees, marginals, and dependence |

These are capacity presets, not quality tiers or certification labels. They use the same
bounded compiler. The self-audit selects a program within the cap; unused bytes are not
padding. Larger capacity does not guarantee larger measured lift. A table's schema and
minimum program may exceed a cap, in which case fitting returns a budget error before
writing an artifact. Forced `--plan` ablations must also obey both caps.

Units are explicit: `KB`/`MB` are decimal, `KiB`/`MiB` are binary; historical `k`/`m`
remain binary. Bare numbers are bytes. Nonpositive, nonfinite, and overflowing sizes
are refused. Parameter caps are positive integers. An omitted byte cap remains 4 KiB,
including when a parameter cap alone is supplied.

The Rust API exposes the identical controls:

```rust
use dope_core::{fit, FitOptions, ModelSize};
let options = FitOptions {
    size: Some(ModelSize::Large),
    budget: Some(24 * 1024),
    max_parameters: Some(4000),
    ..Default::default()
};
// let fitted = fit(&table, declared_target_index, &options)?;
```

The CLI is a thin caller of this API. This PR does not change the separate historical
`dope-kernel` service/MCP wire contract.

## What is a parameter?

`Program::parameters()` and fit/inspect JSON report **version 1 logical scalar slots**
for the *decoded full generator*. Counting only neural weights would hide the forest
and the distribution that generate the training rows.

| Component | Count |
|---|---|
| Network | All weights and biases (existing `Net::parameter_count`) |
| Forest | Thresholds + quantized leaf codes + one scale per tree |
| Marginals | Continuous knots + three scalars per atom; discrete probabilities except the implicit last one; numeric support values |
| Dependence | Tree edge correlations, factor loadings, or the unique full-matrix off-diagonal correlations |
| Regimes | Weights + means + scales; the fixed single standard-normal regime costs zero |
| Noise | Regression coefficients + residual knots + two scale bounds, or one classification temperature |
| Schema and missingness | Two affine scalars per numeric column including target, two target bounds, one missing-rate slot per feature |

Names, categorical labels/codes, feature/parent indices, architecture flags, and fixed
identity/diagonal entries do not count as numerical parameters; their complete stored
bytes still count. Quantizer metadata that reconstructs network coefficients is not
counted twice. This is a versioned logical-state convention, not statistical degrees
of freedom, trainable tensor count, compressed bytes, or RAM. Parameter counts across
unrelated architectures are not interchangeable; byte budgets remain the deployment KPI.

Allocation considers smaller network plans, trims forests, and admits only dependence
upgrades satisfying both caps. The returned decoded program is counted, so the count
matches `inspect` and source-free reload, including calibration and quantization.
The search is bounded, not an exhaustive optimality proof under arbitrary tiny caps.

## Pretrained small, medium, and large

```bash
# Train internally on rights-cleared data. No network/publication occurs here.
dope fit-tiers train.csv -t price --task regression --seed 0 --device cpu \
  --dataset-id public-house-prices-v1 \
  --source https://example.org/dataset \
  --data-license CC-BY-4.0 --out target/public-house-prices-v1

# A recipient needs only the selected artifact and the DOPE binary.
dope inspect target/public-house-prices-v1/small.dope --json
dope sample target/public-house-prices-v1/small.dope -n 10000 -o synthetic.csv
```

`fit-tiers` produces `small.dope`, `medium.dope`, `large.dope`, `MODEL_CARD.md`, and a
`catalog.json` completion marker. It refuses existing output directories, fits all three
before creating output, and records the training CSV SHA-256, binary SHA-256, seed,
backend, engine version, target index, requested/inferred task, whether `--anonymize`
was used, data source and license, artifact SHA-256, actual bytes, scalar
breakdown, and self-audit. Failed writes leave no valid completed catalog. Exact replay
requires the recorded binary, input bytes and backend; floats need not be identical
across hardware. Hashes establish integrity, not authenticity or release authorization.

The catalog explicitly says `qualification=training-self-audit-only` and
`release_authorized=false`. Before open-sourcing a zoo entry:

1. Admit a versioned dataset with explicit target, source digest, license and lineage;
   follow `docs/data-contract.md` for canonical campaign tables.
2. Split by physical entity/time where appropriate. Train/select only on training and
   validation; evaluate once on untouched real test rows with the three auditor families.
3. Review data redistribution and artifact licensing separately. Artifacts contain
   category labels, quantiles and names unless `--anonymize` replaced the names and
   labels with positional codes; `fit-tiers` is not a privacy release mechanism.
4. Publish the model card, test receipts, checksums, source/build provenance and approved
   artifact license in an immutable versioned release. Test source-free sampling.

No untrained placeholders or invented pretrained performance are shipped. This is the
working release builder; internal training on approved sources and open-source weight
publication remain follow-up work. A single cross-dataset S/M/L encoder/proposer would
be a different learned product: define its conditioning interface, train on disjoint
source lineages, freeze weights, and qualify zero-shot versus adaptation separately.
A dataset-trained generator must never be presented as such a foundation model.

## Master KPI: Budgeted Transfer Retention AUC, version 1

Keep the existing robust transfer objective, then score its deployment frontier over a
fixed log-byte range. For each informative auditor family, retention is

`R_f = (L_null − L_synthetic,f) / (L_null − L_real,f)`.

Use `R = min_f R_f`. The current ledger admits a family when the real-trained gain is
at least 1% of null loss. The ranking reducer consumes these stored robust values;
it does not implement a second auditor or infer new measurements.

For byte cap `B`, each configuration contributes the seed median of
`clip(R, 0, 1)` if that seed's complete artifact fits `B`, or zero if oversized/failed.
Take the best tested configuration at that cap. **BTR-AUC** is 100 times the normalized
trapezoidal area under this curve versus `log(B)` on the declared budget grid. Default:
17 powers of two, **1 KiB through 64 MiB**, with equal weight per doubling. Scores range
from 0 to 100. A 100-point score keeps all reference gain at every tested deployment cap.

This score rewards reaching useful transfer at small sizes while allowing larger
models to earn lift at larger budgets. It is preferable to dividing accuracy by bytes,
which can reward an almost useless tiny model. There is no claim that this scalar is a
universal business utility: publish the grid, curves, absolute losses, negative-retention
counts, timings and T90 alongside it. Values above one are saturated and harmful values
are floored for this scalar only; the original loss/retention ledger is preserved.

**Historical descriptive envelope.** The current plot takes the envelope of configurations
already evaluated on development test rows; it is an oracle diagnostic of tested options.
Its apparent frontier can benefit from selection and unequal search effort. It is not
an estimate of choosing a model on unseen data, and it does not qualify D80 or sealed TDL.
For the prospective study, select one configuration per method/budget using validation
only, freeze that selection, and record only those nominated configurations in the ranking
ledger. Fix synthetic row factor, auditor versions, compute allowance, and repeated seeds.

Missing scheduled cells are `incomplete`, never zero. A recorded fit failure is zero and
stays in seed medians. An `ok` cell with no finite informative score is `unqualified`.
Conflicting duplicate receipts are refused. The historical schedule is inferred from the
union of observed configurations: this cannot detect a configuration absent everywhere.
Supply `--schedule schedule.json` for any new campaign; its method entries are explicit
`[[budget_label, seed], ...]` arrays. Supply `--datasets datasets.json` with the
predeclared dataset-ID list too, so entirely absent dataset jobs stay visible. Only datasets complete for every alternative enter
the aggregate Elo; the report also lists every per-dataset missing method.

## Elo as the readable relative ranking

Each complete dataset is one match per pair. A method wins when its BTR-AUC exceeds
its opponent by more than **1 percentage point**; closer results draw. Batch
Bradley–Terry logistic fitting uses the standard Elo probability scale:

`P(A beats B) = 1 / (1 + 10^((Elo_B − Elo_A) / 400))`.

Ratings are centered at 1500. A fixed unit-precision normal prior on log-odds keeps
perfect winners finite and shrinks small cohorts. There is no streaming K-factor or
match-order effect. Ratings depend on the admitted methods, datasets, grid, tie margin,
and prior; persist that protocol with the results. They are not accuracy percentages.

Overall 95% percentile intervals resample **datasets**, preserving paired alternatives,
and refit Elo (1,000 draws, fixed seed). They do not cover every uncertainty source or
show statistically proven superiority. Local per-dataset Elo uses the same batch fit on
that dataset's available complete methods; it is a local relative score, not confidence
from one observation. Never compare local scores across different opponent sets.

```bash
python3 bench/real-world/elo.py
python3 bench/real-world/elo.py --check
# Broader common cohort without the two sparsely evaluated neural methods:
python3 bench/real-world/elo.py --methods dope,rows,kernel,copula,arf \
  --out bench/real-world/results/elo-cpu
python3 -m unittest discover -s bench/real-world -p 'test_*.py' -v
```

Outputs: `elo.json` (full protocol, input/reducer hashes, curves, rankings), `elo.md`,
`datasets.csv` (each alternative on each dataset), and deterministic `elo.svg` with
intervals plus `datasets.svg` with local ratings. The
[all-method panel](../bench/real-world/results/elo/elo.md) has seven common complete datasets; the
[five-method CPU panel](../bench/real-world/results/elo-cpu/elo.md) has sixteen. Neither is thousands of datasets or a sealed result.

## The study over thousands of datasets

Before running the larger study, freeze the following as a versioned protocol:

- **Cohort and leakage:** enumerate dataset IDs, licenses, digests, declared targets,
  source families and entity groups. Deduplicate related datasets before splitting
  development versus final holdout. Keep related sources in the same split. Bootstrap
  independent source lineages for the final uncertainty analysis; dataset bootstrap in
  this PR is only appropriate for independent tables.
- **Methods and resources:** pin implementations, compiler/runtime hashes, hyperparameter
  search budgets, time/memory/thread caps, hardware classes, synthetic row factors,
  train/validation/test splits, seeds, and expected cells before launching jobs.
- **Capacity sweep:** small/medium/large plus the fixed common byte grid; report absolute
  measured size even when a baseline cannot obey a requested cap. Parameter-matched
  experiments are a separate track with architecture-specific counting disclosures.
- **Selection:** validation chooses artifacts per cap; no test access during search or
  tier tuning. Freeze the selections and execute final test scoring once. D80 and
  empirical privacy require their own measured receipts and cannot be inferred from Elo.
- **Execution:** use the existing resumable one-JSON-line-per-cell runner, write shards
  with one writer each under `target/`, then merge by exact cell identity. Every timeout
  and crash gets a receipt; missing jobs remain incomplete until resolved. Partition
  deterministically by dataset/source family, not outcome. Never retune from partial
  final-test leaderboards.
- **Reporting:** require the predeclared schedule and dataset manifest, show common coverage and all missing
  cells, retain the raw loss/size/runtime records, report BTR-AUC, Elo, T90, tail harm,
  domain strata and latency. Publish prospective validation-selected and historical
  oracle-envelope results as separate cohorts. Use a larger source-cluster bootstrap
  with the preregistered power/precision analysis before making broad ranking claims.

The current artifact builder and reducer are runnable groundwork for that campaign.
They do not run an unbounded external-data campaign automatically when this PR merges.

## Unified 3-Level Landmark & Data-Descriptor Architecture (`DOPE-MLC-v2`)

For the cross-dataset 3-level landmark-conditioned architecture (`Medium` 4 KiB /
`4,096` coords, `High` 16 KiB / `8,192` coords, `Ultra` 64 KiB / `10,240` coords)
trained across 19,079 regression and binary classification splits and evaluated
across CatBoost, Tabular Kumo (`kumo_tabpfn`), LightGBM, XGBoost, MLP, and Linear
auditors (reaching `#1` overall Bradley–Terry Elo = `1680.1` and `Ultra` BTR-AUC =
`83.17` with Driver Attribution Fidelity $D = 0.8161$), see
[Multi-Level Landmark & Data-Descriptor Deep Learning Study](multilevel-landmark-dl-study.md).
