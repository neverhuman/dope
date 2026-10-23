# Validation of the transfer-compression and Rust deployment upgrade

Base reviewed: `aac3fdadf93fdbd07811a0e16e53349b4d8eff40`.
This is engineering verification, not a scientific benchmark or release receipt.

| Check | Observed result |
| --- | --- |
| Baseline `cargo test --locked --lib`, before edits | 279 passed. Baseline evidence, not a candidate-wide pass. |
| `bash deploy/check.sh` on the candidate | Passed: formatting, all-target Clippy with warnings denied, five unit tests, one actual CLI integration test, and the normal dependency graph check. |
| `cargo build --manifest-path deploy/Cargo.toml --locked --offline --bin dope-runtime` | Passed, default CPU build. An unoptimized engineering build, not a release-size benchmark. |
| `tools/claims` unittest discovery | 44 passed in two earlier runs during the upgrade. A final rerun encountered six filesystem errors while copying fixture trees after the shared disk filled; it is not reported as a final full-suite pass. |
| Public claims firewall and deterministic renderer | Passed on the final publication surfaces. |
| Hero figure tests | Seven passed, including authenticated values, intervals and CSV agreement. |
| Historical README benchmark tests | Eight passed; generated result blocks and frozen paper links retained. |
| Python metric conformance | 129 golden cases passed, tolerance `1e-12`. |
| Active paper build | Main and supplement compiled; references resolved; generated hero confirmed as Figure 1 on page 1. Rendered pages inspected. |
| `git diff --check` and Rust formatting | Passed. |
| Full `just fast` and `just check` | **Not completed.** Both attempted; Rust dependency compilation failed with `No space left on device`. Earlier full-product dependency builds also received SIGKILL under shared resource pressure. These outcomes are retained as failures. |

Deployment fixtures cover nonzero decoder weights, feature permutation, multiple
batches, full-width seeds, historical CPU draw equality, corrected scalar
activation values, explicit profile differences, corruption/truncation, resource
bounds, full byte charging, unavailable CUDA, actual pack/inspect/sample commands,
exclusive outputs and redacted errors. They execute no vendor model and establish
no downstream transfer result.

The inference-only package compiles the same canonical source modules as the
product. Its normal graph has no Python/libtorch binding or compiler/campaign/
service module. Full product unit and integration suites remain required in
repository CI; the narrow package does not replace them.

## Required receiving checks

Run `just check`, the existing required CPU/security/audit jobs, and the new
`Rust deployment conformance / cpu` lane on a runner with adequate disk and RAM.
The historical paper's independent replay (`just paper-tests` and
`just paper-check`) remains a receiving gate and was not reported as passing in
this constrained local run. The active paper has its separate `just paper-tdl`
build.

CUDA compilation/execution, missing-driver startup, OOM behavior, CPU/GPU parity,
release-build size/latency, deployment notices/SBOM, trained-model activation
transfer, observational benchmarking and sealed confirmation remain unmeasured.
The SafeTensors carrier is a development profile; duplicate decoder state is
charged in full. No frozen contract, historical result, privacy label or scientific
qualification was promoted.

The finite development compute request is in
[`technical-upgrade.md`](technical-upgrade.md#10-finite-compute-request-for-the-pr-discussion)
and the PR discussion. Qualify numerical behavior and measure a timing pilot
before proposing a larger campaign; keep the official test split reserved.


## Standalone dependency security pin

The original standalone lock resolved Candle 0.11.0 to ZIP 8.6.0, which is affected by
RUSTSEC-2026-0367, RUSTSEC-2026-0368 and RUSTSEC-2026-0369. Its original failing
scan is retained in the integration evidence; no advisory is ignored.

The runtime temporarily pins the reviewed four-file
[0.11.0 backport](https://github.com/eveselove/planly-candle/commit/b76d808ad646c479b9f61b9e0c59503143543f12)
of [upstream Candle #4041](https://github.com/huggingface/candle/pull/4041).
Its parent is the published 0.11.0 source, 31f35b147389700ed2a178ee66a91c3cc25cc80d.
The delta changes only the ZIP dependency, fallible NPZ/PTH filename handling and
serialization regressions. CPU, CUDA and quantization implementation files retain
the published source bytes. Cargo.lock pins the exact dependency commit and ZIP
release; there is no branch tracking or local registry edit. The same dependency
scan runs locally and in the additional deployment-conformance CI job, whose JSON
and dependency inventory are retained as an artifact even on failure.

Return to the official published Candle dependency once it includes the ZIP 9
compatibility fix and passes the same locked conformance/advisory checks. This is
an engineering dependency repair, not measured CUDA execution or scientific qualification.

The engine now uses this same exact core/NN dependency pin after its independent
lockfile scan found ZIP 7.4.0 affected by those three advisories. The security lane
audits `engine/Cargo.lock` as well as the root and merge-guard locks; deployment
retains its separate scan. The root lock also moves `ctutils` from 0.4.3 to 0.4.4
for RUSTSEC-2026-0385, without an advisory exclusion. Source-pinned upstream
license bytes accompany the new NN dependency in the engine release notices.

Rust 1.88 cannot compile the newer Apple ARM NEON FP16 vector type, so the
Apple ARM target disables the optional `fp16` target feature and uses Candle's
conversion-based fallback. F32 model execution and carrier contracts are retained;
locked engine tests and learned operator/recipient parity qualify the change.
GPU execution and speed equivalence are not inferred from these CPU checks.
Previously measured comparator binaries and training snapshots remain frozen;
this dependency repair does not reinterpret their scientific receipts.
