# Install the native DOPE CPU alpha

The `dope` facade and historical `dope-kernel` are distinct executables. The
facade installer preserves the kernel installation and its historical readers.
The DOPE 1.0 engine CLI (`engine/dope-cli`, installed by `cargo install`) is a
third, separate program that is also named `dope`; keep only one `dope` on
`PATH`, or call the facade by its absolute path under the install prefix.
This source change provides Linux x86_64 CPU-alpha packaging; no signed public
release or download URL is established by these instructions.

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
