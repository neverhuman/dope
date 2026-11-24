# Install DOPE

## Signed Linux engine

The program a new user runs is the signed `dope` binary from
[dope-engine-v3.0.0](https://github.com/neverhuman/dope/releases/tag/dope-engine-v3.0.0).
It is Linux x86-64 (`glibc` 2.39 or newer) and needs `gh` 2.102.0 or newer.
The installer below is pinned to release commit
`e15c415bad9554f7c419dd240aa670d1a79a27fe`. It verifies `SHA256SUMS` and the
GitHub/Sigstore attestation for that tag and source commit, then installs
`~/.local/bin/dope`.

```bash
curl -fsSL https://raw.githubusercontent.com/neverhuman/dope/e15c415bad9554f7c419dd240aa670d1a79a27fe/scripts/install.sh \
  | bash -s -- \
      --tag dope-engine-v3.0.0 \
      --source-digest e15c415bad9554f7c419dd240aa670d1a79a27fe \
      --require-attestation \
      --no-weights
dope doctor
dope fit train.csv --target y --format auto-v8 --budget 2048 -o model.dope
```

`--no-weights` skips SafeTensors downloads. The analytic `auto-v8` path needs
no checkpoint. `--tag` selects the engine release and defaults to
`dope-engine-v3.0.0`. Installer `--v8-weights` demands
`dope-v8-weights.safetensors` from that same release and fails on
`dope-engine-v3.0.0`, which does not ship it. There is no signed
`dope-engine-v3.1.0` tag to pass.

Offline verification, the asset list, and a from-source package are in
[signed-engine-release.md](signed-engine-release.md). The table walkthrough is
[getting-started.md](getting-started.md). Command details for this binary are
in [engine/README.md](../engine/README.md).

The release assets are `dope-linux-x86_64`, `SHA256SUMS`, `dope.sigstore.json`,
`BUILDINFO.json`, `Cargo.lock`, `LICENSE`, `THIRD_PARTY_NOTICES`, and
`DEPENDENCIES.txt`. Confirm those names with
`gh release view dope-engine-v3.0.0` before you trust a mirror.

## Historical CPU alpha facade

The alpha facade and historical `dope-kernel` are separate executables from the
signed engine above. The facade installer preserves the kernel installation and
its historical readers.

The DOPE 1.0 engine CLI (`engine/dope-cli`) is also named `dope`. The signed
release above is that engine. This alpha section is an older local package:
keep only one `dope` on `PATH`, or call the facade by its absolute path under
the install prefix. A hash in this section checks byte identity. It is not the
Sigstore attestation on `dope-engine-v3.0.0`.

Build from a reviewed checkout with the pinned Rust toolchain and locked graph:

```sh
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS cargo build --locked --release --offline --bin dope
sha256sum target/release/dope
bash scripts/install-dope.sh "$PWD/target/release/dope" ACTUAL_SHA256 "$HOME/.local/dope-alpha" --dry-run
bash scripts/install-dope.sh "$PWD/target/release/dope" ACTUAL_SHA256 "$HOME/.local/dope-alpha"
```

Replace ACTUAL_SHA256 with the exact digest from a trusted build or reviewed
bundle. A hash checks byte identity; it is not a publisher signature. The prefix
must be absolute and fresh, with no symlink ancestors or existing owned files.
The installer requires Bash/coreutils and checks the real executable identity,
CPU profile, Cargo.lock/toolchain digests and complete pinned legal inputs.
It installs no Python, GPU stack, model weights, package manager or network client.

An extracted bundle uses this layout: `dope`, `LICENSE`, `Cargo.lock`,
`rust-toolchain.toml`, `scripts/{install-dope.sh,uninstall-dope.sh,native-legal.sha256,native-third-party.NOTICE}`,
plus `SHA256SUMS` and its release-specific SPDX SBOM. Verify the entire inventory
against a trusted manifest before running an installer; the installer does not
establish publisher authenticity. Run its script from the extracted bundle,
passing the bundle's absolute executable path and trusted digest.

Use a new output directory for the fixture tutorial:

```sh
mkdir alpha-demo
cd alpha-demo
DOPE="$HOME/.local/dope-alpha/bin/dope"
"$DOPE" doctor
"$DOPE" demo --out demo.dpk --report demo.json
"$DOPE" inspect --artifact demo.dpk
"$DOPE" sample --artifact demo.dpk --rows 100 --seed 7 --out sample.csv
wc -l sample.csv
```

The output is 100 headerless numeric CSV rows. `demo` is a built-in synthetic
fixture, not a scientific benchmark. Sampling uses the artifact without the
original fit table. The facade supports fully observed numeric regression;
prepared-view evaluation, official worker qualification and scientific/privacy
certification remain unavailable. See [CLI usage](cli.md) for an explicit CSV
schema workflow and budgets. This installation does not establish formal DP,
HIPAA de-identification, utility, model rights, independent reproduction or
release authorization.

```sh
bash /absolute/path/to/extracted/scripts/uninstall-dope.sh "$HOME/.local/dope-alpha" --dry-run
bash /absolute/path/to/extracted/scripts/uninstall-dope.sh "$HOME/.local/dope-alpha"
```

Uninstall validates all owned hashes first, refuses modified/incomplete/symlinked
inventories, and removes only its fixed members. Directories and unrelated files
are preserved. Installation refuses overwrite and retains incomplete candidates
for inspection rather than cleaning them automatically.

`cargo test --locked --test vnext_dope_install` covers identity, installation,
100-row artifact-only sampling and refusal/preservation cases. The smoke entry
`bash ops/ci/native-release-smoke.sh` retains kernel checks and builds, archives,
extracts and tests the facade with a current SBOM. Its recipient process has an
empty working directory and no Python on PATH; no repository files are bundled.
This functional recipient test does not unmount the host repository or establish
a general OS isolation boundary. Preserve that distinction when reporting results.
