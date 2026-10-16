> **Archived 2026-10-15.** This is the decision history moved from `docs/vnext/DECISIONS.md` (T-007).
> It is kept verbatim below for traceability. Active decisions are in [docs/decisions.md](../decisions.md).
> References to `DECISIONS.md` in older documents point here.

# vNext contract and evidence decisions

`production/behavior-contract-v1.json` is a draft research contract. It is not
embedded by `build.rs` and activates no production tier. Numeric regression
task replacement is the initial study scope. The other four capability labels
define distinct workloads and require their own confirmation campaigns.

Development fitness selects candidates using development data. The scientific
endpoint measures paired real-query losses after a frozen choice. Release
authority requires the conjunction of verified identity, inventory, frozen
evaluation, scoped transfer, applicable privacy evidence, rights, authorization,
and software/security/audit gates. A finite score supplies none of that authority.

Losses are computed after inverting fit-only target transforms, in original
target units. Exact serialized sampling bytes constrain feasibility and define
a frontier; they are not a score weight. Required learner families are recorded
separately from held-out families. The proposed portfolio still needs exact
recipes and checkpoint pins. Population samples and optimized teaching/context
samples have different contracts and must use matched controls.

Unknown margins, rights, checkpoint pins, budgets, cohort identity, power,
isolation, retry policy, multiplicity, projection, or operator authority keep
`protocol_frozen=false` and `confirmation_allowed=false`. The schema rejects
frozen records with unresolved fields. Resolving those fields requires evidence,
not optimistic booleans. No sealed data is needed to validate this draft.

## Immutable migration boundary

The new contract records SHA-256 of the complete historical file bytes,
including the trailing newline. These pins differ from the embedded v2 digest,
which excludes the trailing newline. Preserve both conventions in their own
formats. Never rewrite historical scorers, receipt ledgers, or DPK3.2/3.3
fixtures to apply a new endpoint retroactively.

| Historical input | Full-file SHA-256 |
| --- | --- |
| `production/kpi-contract-v1.json` | `fe3abcd1cf4653c99a5d0c9505b7645a5fd968f86805dd5b7573ad29ac052760` |
| `production/kpi-contract.json` (active v2) | `d04a2672ed25e92baeb89bec51ea5ccddb04165701cb44d2e1d53ca32f6e219b` |
| `production/kpi-contract-v3.json` | `40cd035d838657de3bd5d27bb940d771cbae697ac90efb5beeab53a5bf7adcbb` |

A `ContractMigrationDiff` records predecessor/successor digests, a new identifier,
semantic changes, and migration proof. Historical receipts decode under their
original contracts. Post-outcome revisions create a new exploratory identity.
Withdrawal or rollback retains immutable ancestry and removes new authority;
it does not replace previous evidence.

Execution, transfer, empirical privacy, formal mechanism, legal route, and
release states are orthogonal. Empirical privacy never establishes formal DP
or HIPAA de-identification. A successful local code gate establishes software
behavior only; scientific and publication gates remain unmeasured here.

## Receipt and cache boundary

`rust/evidence.rs` and `research/transfer/evidence.py` introduce
`UnverifiedReceipt`, `EvaluationIdentity`, `VerificationPolicy`,
`VerifiedEvaluationReceipt`, `verify_receipt`, `scientific_cache_key`, and
`fit_replicate_key`. A verified value has no public Rust constructor. Python's
constructor requires the verifier's internal token and returns defensive copies;
this is an API boundary, not a sandbox against code running in that interpreter.

The caller supplies expected identity, executor digest, canonical receipt digest
and optional freeze digest from an independently trusted executor/custodian.
Copying these pins from a candidate receipt establishes no trust. This packet
does not implement executor signatures, isolation or custody admission; PR03
must establish those before any real confirmation. Fixtures use synthetic
policies only and do not qualify a source, learner, privacy route or release.

The complete identity includes source/code, data/family/task/split/schema,
feature and target transforms, generator/configuration/fit, artifact inventory,
sample/query, learner/checkpoint/recipe/metric/environment, all seed axes,
fit/requested/produced/consumed/query/context counts, precision, ensemble and
determinism policy. Aliases and new attempts do not create independent fits.
Changing any identity field changes the scientific cache key. The fit replicate
key conditions on data, split, generator/configuration, transforms and fit/global
seed, so samples, learners, filenames, state aliases and retries cannot inflate
fit replication.

`dope-finite-json-v1` sorts UTF-8 object keys, rejects duplicate keys and nonfinite
numbers, bounds integers to absolute value `2^53-1`, and formats floats with
seventeen digits after the decimal and a normalized exponent. Float and integer
scalars retain distinct representations. This is a versioned local wire rule,
not RFC 8785. SHA-256 file references hash actual bytes; receipt and cache keys
hash canonical bytes. Both runtimes share golden cache/replicate identities.

The three JSON schemas use a closed executable subset of Draft 2020-12.
Verifiers reject unsupported validation keywords and require exact integer
scalars even where a general JSON Schema implementation accepts integral floats.
Shape validation is followed by byte, DAG and semantic checks. Every referenced
dependency must exist, have its declared kind/size/digest, and be reachable.
Open directory descriptors with no-follow flags reject absolute, traversal and
symlink paths. Files are streamed; JSON manifests are capped at 16 MiB.

The local dependency manifests have these closed layouts:

- `dope-split-manifest` v1 binds data/query digests, partition role, fit and query
  row counts; its dependencies are exactly the data and query files.
- `dope-generator-fit` v1 binds data/split/generator/configuration/feature-transform/
  target-transform digests, fit/global seeds and the fitted-state digest.
- `dope-artifact-inventory` v1 lists sampling file digests. The list and dependency
  set must agree exactly; their actual byte sum is `sampling_payload_bytes`.
- `dope-paired-losses` v1 binds query/metric digests, original loss space, declared
  MSE or MAE units, and aligned real/synthetic/null query-loss arrays. Their means
  must match the receipt within `1e-12` absolute or relative tolerance.

Each manifest includes `format` and integer `version=1`. The executable verifier
binds resolved protocol workload and freeze commitments to the DAG. Confirmation
also requires the frozen cell identity, executor, selection receipt, evaluator
code and access-ledger digests. A hash verifies their bytes, not that their
contents establish genuine access control, lawful rights or unbiased inference.
Those checks remain separate required gates.

Only a complete, private, verified measurement can be committed to a cache.
Failed evaluation dispositions have null loss/evidence fields and never become
verified successes. Raw measurements retain unmeasured transfer/privacy,
no formal mechanism, unreviewed legal route and restricted-local release;
later qualified analyses and authorization require their own evidence.
Public projections are not accepted as protected paired-evidence receipts.

An fsynced private cache envelope contains the canonical receipt, its digest,
scientific key and completion marker; atomic rename commits it. Partial staging
files remain available for repair and cannot be cache hits. Every load checks the
marker and policy and reverifies dependency files. A final role cannot match a
development key. Cache access permissions and principal isolation remain the
custodian's responsibility; key separation alone cannot hide final outcomes.

## Parallel lane integration

Owner/test maps cover all paths in the Phase A COORD ownership list.
`just fast` discovers all transfer tests that land from the parallel lanes.
The Python boundary allowlist classifies `research/transfer/` as offline research;
it acquires no product-truth path or release authority.
This branch registers `evidence`. After PR04's owned `rust/transfer_metrics.rs`
lands, the integrator must add `pub mod transfer_metrics;` to `rust/lib.rs` and
run its module tests plus `just check`. The source is absent on this branch;
enabling that declaration now would break a standalone build. No parallel-lane
source is copied or edited here. PR05/35 needs no additional Rust registration.

## N02 narrow narrative decision (2026-10-14)

Scope: title, thesis and terminology of the historical-evidence paper
(`docs/whitepaper/dope-mfs.tex` and its supplement). This record does not
freeze the behavior contract, a knowledge-transfer claim, or any metric.
`production/behavior-contract-v1.json` and `protocol_frozen=false` are unchanged.

- **Title.** `DOPE: Generative Compression of Predictive Structure`. The
  acronym stays **Downstream Objective-Preserving Encoding** and is expanded
  once in the abstract. Kilobyte-led titles are retired: size is a charged
  axis, not the contribution.
- **Gated title.** `DOPE: Compressing Data into Transferable Knowledge` stays
  withheld. It needs a frozen held-out learner family and a held-out source
  family. On the historical 100-lineage panel neither exists for the DOPE
  profile. The N01 receipt (`target/vnext/20261013/receipts/N01.json`, sha256
  `1264ca2108dc70310a1e7082c5123b19e0ed1e461dfcf3310665aad3358ef239`) shows
  why: the profile comparison read every lineage under CatBoost, linear and
  MLP, and its twelve discovery/confirmation lineages span every source-family
  scope.
- **Predictive structure.** This term is operational. It means the
  feature-target relationships that a declared learner uses, measured as
  retention. Retention is the fraction of a real-trained learner's gain over a
  null that a learner trained only on synthetic rows recovers on real
  validation rows. Every use is relative to a workload: target, learner,
  synthetic size and loss. The term does not denote general knowledge,
  intelligence, mutual information or description length.
- **Compression.** Minimum description length, Occam's-razor bounds and the
  information bottleneck are motivation only. Small size proves nothing about
  learning: empty, constant and checkpoint-hiding generators are
  counterexamples. Every sampling byte is charged, and evidence comes only from
  learners trained on samples.
- **Generative compression.** The term already names learned image/video
  compression (Santurkar et al., arXiv:1703.01467). The paper uses it
  descriptively and claims no priority.
- **Terminology not adopted.** No `Data IQ`/`DIQ` label: Data-IQ (Seedat et
  al., NeurIPS 2022) already exists, and the label would imply an unvalidated
  intelligence scale. Data-IQ primarily stratifies examples by training
  dynamics and aleatoric uncertainty; as a secondary application it compares
  datasets, synthetic ones included. The paper cites it in related work as
  close evaluation prior art, not only as a name collision, and never
  describes it as a byte or density measure. Qualified Learning Density (QLD) is not adopted as a
  headline, ranking, gate or table metric. It stays an exploratory candidate
  for one artifact on one fixed workload, pending the N02 adversarial
  examples. vNext primary outputs remain the frontier and `B*_tested`, with
  censored, non-monotone and refusal states. No information-theoretic minimum
  is claimed.
- **Profile-selection disclosure.** The four bounded research profiles were
  compared by validation retention at fit seed 11 on the discovery,
  confirmation and all-lineage publications. Each publication keeps all four
  profiles, and no contemporaneous record states the selecting rule or the
  rows consulted. The paper therefore classifies the whole panel and all three
  auditors as selection-exposed. It withdraws the earlier "chosen on a
  six-lineage discovery panel and checked on six disjoint confirmation
  lineages" and "pre-registered profile" wording. The sealed test, which is
  pre-declared in `research/benchmark/review_fixes/predeclare.json`
  (`sealed_test`), remains the only selection-free estimate.

This decision changes manuscript prose, citation metadata and one registry
description string. It changes no ledger, reducer, generated macro, statistical
identity or scientific contract.
