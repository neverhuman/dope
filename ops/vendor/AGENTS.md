# Pinned upstream Torch sources

Rust-core owns this D20 patch. `tch/` and `torch-sys/` contain authenticated
upstream 0.20.0 runtime/build sources and licenses, not new DOPE implementations.
`torch-zip-provenance.json` records the retained files and the scoped ZIP patch.
Keep all other runtime files byte-identical to their recorded upstream hashes.
Do not add models, examples, downloaded runtimes or advisory waivers here.

Run `cargo test --locked --test feature_boundary`, the existing merge-guard
feature-boundary tests, and
`cargo check --locked --offline --features gpu-training,tch/doc-only` after a
patch change. Full security and native-notice gates still apply. The doc-only
compile is API evidence, not GPU, native-model or release qualification.

Migration: remove these local patches only after an admitted upstream Torch
0.20-compatible fix replaces ZIP <9 and the same proofs pass. A backend or
LibTorch ABI upgrade requires a separate coordinator decision and lease.
