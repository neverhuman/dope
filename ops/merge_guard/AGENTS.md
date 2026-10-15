# Native operations merge guard

Governance owns this independent Rust package. It issues GitHub GET requests
only, keeps bounded transport without retries, refuses unknown evidence, and
always emits merge_authorized=false. It cannot approve, queue, merge, change
controls or switch credentials. Product modules and release truth remain outside
this executable. Do not introduce Python runtime or policy exceptions.

From the repository root run `just native-guard-check` for format, Clippy and
all native scenario/boundary tests. Direct Cargo commands use
`--manifest-path ops/merge_guard/Cargo.toml --target-dir target` to reuse the
lane's existing root target. Keep the independent lock at existing root-locked
registry versions/checksums; never enable tool features in the product package.
Preserve every scenario/subcase mapping and version 1 canonical receipt digest.
Run required `just fast` and `just check` before committing. Security audits
both locks through the existing canonical lane; preserve failed evidence.
