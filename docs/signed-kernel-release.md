# Signed research-kernel release

The research kernel's `dope-kernel` executable (root crate, version in `Cargo.toml`) is released
separately from the Rust engine ([signed-engine-release.md](signed-engine-release.md)). A
consumer, such as the VEOX cloud appliance, pins one exact commit that it has tested. That commit
is usually far behind `main`, so a release names a commit instead of building `main`'s head.

Signing establishes build identity: these bytes came from this repository's
`kernel-release.yml`, running on `main`, from the named commit. It does not certify model quality,
privacy or any tier.

## Cut a release

1. Pick a commit that is an ancestor of `main`. The tag is `dope-kernel-v<Cargo.toml version at
   that commit>`, and must not exist yet.
2. Dispatch the workflow from `main`:

   ```bash
   gh workflow run kernel-release.yml -R neverhuman/dope --ref main \
     -f tag=dope-kernel-v0.3.0-alpha.1 -f commit=<40-character sha>
   ```

3. The workflow checks the tag and commit, builds the commit with `ops/ci/kernel-release.sh`, signs
   `dope-kernel`, `BUILDINFO.json`, `Cargo.lock` and `SHA256SUMS` with GitHub artifact
   attestations, verifies them, and stages a **draft** release tagged at the commit.
4. A maintainer reads the draft and publishes it. Publishing creates the tag.

## Verify a release

```bash
gh release download dope-kernel-v0.3.0-alpha.1 -R neverhuman/dope -D ./kernel-release
bash scripts/verify-kernel-release.sh ./kernel-release dope-kernel-v0.3.0-alpha.1 <commit>
```

The script checks both attestations (signer workflow `kernel-release.yml`, source ref
`refs/heads/main`), the fixed inventory in `SHA256SUMS`, every hash, and that `BUILDINFO.json` names
the expected commit and version. Only then mark the binary executable.

## Reproduce the bytes

```bash
bash ops/ci/kernel-release.sh <commit> target/kernel-rebuild
sha256sum target/kernel-rebuild/dist/dope-kernel
```

The build reads only the commit's `git archive` export. It runs in
`rust@sha256:af306cfa71d987911a781c37b59d7d67d934f49684058f96cf72079c3626bfe0` (rust 1.88.0, Debian
bookworm, glibc 2.36), with fixed container paths remapped out of the binary and the repository's
release profile. Two such builds of the same commit are byte-identical, so anyone can check a
release without trusting the runner. The script refuses linkage beyond `libc`, `libm` and
`libgcc_s` and any glibc symbol newer than 2.36.

A `dope-kernel` built on a host toolchain, such as one from `cargo build --release`, has different
bytes even at the same commit, because it embeds host paths and links against the host's glibc.
