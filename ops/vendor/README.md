# Torch 0.20 ZIP security patch (D20)

The root keeps optional `tch =0.20.0` and the existing LibTorch ABI. Both
`tch` and `torch-sys` 0.20.0 require ZIP 0.6 upstream, so Cargo cannot replace
that dependency with ZIP 9 using a ZIP-only patch. These two local path patches
retain only upstream library/build sources and licenses. Examples, model assets
and upstream development tools are omitted (about 4.2 MB retained).

`torch-zip-provenance.json` records crates.io archive SHA-256s, upstream file
hashes and the retained file hashes. Archives were checked against their original
registry lock checksums before selecting regular members. The scoped changes are:

- Both manifests pin registry ZIP 9.0.5, preserving the old AES, bzip2, deflate,
  time and zstd feature families. No package named `zip` below 9 remains.
- `tch/src/tensor/npy.rs` uses ZIP 9's `SimpleFileOptions` alias for its existing
  Stored NPZ writer and propagates ZIP filename/index errors. `torch-sys/build.rs`
  propagates ZIP 9's fallible name/path getters. The rest of the retained runtime
  source stays byte-identical.
- Manifests omit upstream example/test/dev targets that are not distributed here.

`cargo test --locked --test feature_boundary` verifies the optional Torch pin,
the resolved registry ZIP version, retained source hashes, codec roundtrips,
symlink confinement and privileged-mode masking. The pre-existing merge-guard
feature/isolation proof still runs through `just fast` independently.
`cargo check --locked --offline --features gpu-training,tch/doc-only` checks the
optional Rust API without downloading or linking a LibTorch runtime. It does not
qualify GPU behavior or certify a model.

Default CPU native notice generation still follows its normal/build graph. The
optional Torch licenses are retained beside their sources; these patches create
no advisory exception and do not alter the security scanner or audit policy.
