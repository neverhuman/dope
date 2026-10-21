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
