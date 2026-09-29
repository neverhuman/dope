# Production campaign inputs

The files in this directory are immutable inputs to the V3 campaign. The KPI
contract is canonical JSON and its SHA-256 and BLAKE3 digests are embedded in
the Rust binary at build time. `freeze-contract` binds it to a source commit,
environment lock, corpus manifest, candidates, auditors, and workload before
any job is admitted.

The active v2 contract retains PTF-v1 as the primary release KPI. Its hard
gates include joint fidelity, membership and attribute attacks, exact and near
copies, and permutation importance Spearman (at least 0.70) and top-k Jaccard
(at least 0.50) when at least three features are informative. Missing applicable
rank evidence fails the gate. The v1 file remains available only for historical
receipt verification.

The contract also freezes MFS-v2: a geometric mean of utility transfer (0.30),
driver fidelity (0.20), distribution fidelity (0.20), structure fidelity
(0.15), coverage realism (0.10), and compactness (0.05). Privacy has zero soft
weight. `tstr-certification.json` records all hard gates, the measured Pareto
vector, raw metrics, and weight sensitivity. The score is `null` if any hard
gate fails or a required soft term is unmeasured. Coverage realism uses
deterministic PRDC precision, recall, density, and coverage on the sealed
holdout and a synthetic sample. Distribution fidelity combines marginal W1,
16-slice Wasserstein, and three-bandwidth RBF MMD diagnostics. Each gold sample
uses at most 128 rows and a shared embedding of at most 32 dimensions, with
numeric feature values, missingness indicators, and target. The PRDC neighbor
count is at most five and adapts to small samples. Fewer than six rows leave
the diagnostics unmeasured. The other terms use the gold retention lower bound,
permutation importance rank/top-k agreement, sampled pairwise correlation
fidelity, and artifact bytes relative to the tier cap.
Sensitivity values are diagnostics, never a substitute release score. Neither
an empirical privacy result nor a tier establishes formal
DP or HIPAA de-identification.

The production environment is Rust-only. GPU training and evaluation may link
libtorch 2.7 through the pinned optional `tch-rs` feature; release inference
must not link libtorch. A production freeze additionally requires source and
package hashes, an SBOM, CUDA/driver inventory, notices, and a model checkpoint
manifest. Python, sklearn, XGBoost, CatBoost, and TabPFN are not production
dependencies.

`embed-real-regression-cluster.sh` runs the production action encoder over the
real regression corpus on xbabe1/2/3. It first asks the native binary to validate
the trained router architecture, artifact formats and hashes, training-evidence
link, finite evidence, canonical bundle byte count, and fixed 4,168-dimensional
layout. Measured promotion failures remain visible in every manifest but do not
block this embedding-only workflow; all campaign and release promotion gates
remain unchanged. No remote directory is created and no job is dispatched when
embedding integrity fails. After qualification it smoke-tests an ordinary and a
target-only dataset, confirms that all three hosts have the same reconstructable
corpus inventory, assigns datasets by sorted index modulo three, runs CPU workers,
consolidates the unique
`<dataset>_<method>_<dimension>.json/.vs` files, verifies every vector and router
lineage (including the native zero-feature records), and writes shard plus merged
manifests. Build the release binary first, then run:

```bash
production/embed-real-regression-cluster.sh
```

Inference in this workflow is CPU-only; `DOPE_EMBEDDING_JOBS` controls worker
parallelism on each host.

Ordinary binaries identify as `0.3.0-alpha.1`. The `1.0.0-rc.1` identity is
written only by `build-rc`, after coverage, KPI, auditor, candidate, router,
receipt, disk, signature, and file-hash gates all pass. `campaign status
--json` uses `null` for router regret and PTF-v1 until frozen evidence exists;
job completion alone is never treated as a quality score.

`split-validation` derives a deterministic, path-free 60/40
validation-select/validation-cert submanifest from the corpus manifest. It
does not read dataset rows and keeps validation-cert marked sealed.
