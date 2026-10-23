# Signed Rust engine and command-line deployment

The deployable product in this PR is `engine/dope-cli`'s `dope` executable, backed by
`engine/dope-core`. Customer fitting, sampling, prediction, schema encoding, built-in
retention evaluation and frontier search execute in Rust. No Python interpreter,
libtorch, Python package environment, shell helper, network service or training framework
is needed by those application commands.

Python may be used internally for deep learning training and research evaluation. Its
scripts, environments and checkpoints are not shipped as executable dependencies.
A deployment model is supported DOPE graph/schema/parameter data (`.dope`), not Python
pickle or arbitrary executable code. Arbitrary PyTorch/SafeTensors checkpoints are not
implicitly compatible with that format. The separate historical `deploy/dope-runtime`
and research-kernel DPK workflow retain their existing compatibility boundaries.

## The CLI contract

`dope --help` lists commands; each subcommand has `--help`. Global `--threads N`
sets the Rayon CPU worker pool before application work starts, accepts 1–1024 and works
before or after the subcommand. It limits CPU workers, not whole-process RAM or GPU
allocation. Without it, Rayon uses its usual default/environment configuration.

`fit --json` emits one complete fit report on stdout. `--report FILE` may be used
alongside it; `--quiet` conflicts with `--json`. Diagnostics and failures go to stderr;
CSV commands retain CSV stdout unless `--out` is supplied. Exit 0 means success; argument
validation failures use Clap's exit 2 and application failures exit 1.

`doctor --json` reports product/runtime identity, OS/architecture, available backends,
default device, worker count, executable SHA-256 and the absence of Python/libtorch
requirements. This is runtime diagnostics, not self-authentication: only independent
signature verification establishes the publisher/build identity.

README examples cover size and parameter controls, S/M/L, reproducible seeds, structured
reports, raw and ML-ready sampling, feature-only prediction, encoding, evaluation,
frontier search and throughput. Inputs still require the declared training target;
prediction inputs can omit it and must retain the named input features.

## Build and package

```bash
# New output directory required; this is a development build until signed.
bash ops/ci/engine-release.sh target/engine-release
```

The builder is scoped to Linux x86-64 GNU CPU. It uses the locked standalone engine
workspace, `--no-default-features`, a baseline `x86-64` CPU target and the existing
optimized release profile (LTO, stripping, abort-on-panic). It tests the Rust workspace,
builds the actual release executable, and exercises every customer command on synthetic
fixtures with an empty executable-helper `PATH`.

The normal/build dependency inventory refuses Python bindings, libtorch and implicit
GPU dependencies. ELF imports are also checked for Python/libtorch/CUDA/MKL linkage.
The package contains one application executable, model data supplied separately, and
build/license/verification sidecars. It contains no Python scripts or environments.
It can still link ordinary OS libraries such as libc/libm; this GNU build is not a
universally static image or a macOS/Windows release. CUDA and Metal remain separate
source builds and are not qualified by the signed CPU workflow.

Fixed package members:

| File | Purpose |
|---|---|
| `dope-linux-x86_64` | The single Rust application executable |
| `BUILDINFO.json` | Version, source commit, compiler, target, source cleanliness, binary and lock hashes |
| `Cargo.lock` | Exact engine package versions |
| `DEPENDENCIES.txt` | Actual default CPU normal/build closure |
| `LICENSE`, `THIRD_PARTY_NOTICES` | Project license, upstream dependency license bytes and Rust library notices |
| `SHA256SUMS` | Fixed allowlisted package inventory and digests |
| `dope.sigstore.json` | Signed provenance bundle, added by the trusted tag job |

The notice collector uses the actual locked normal/build graph, including build-time
packages. Three crate archives omit their upstream root license; version-specific
copies under `ops/licenses/engine/` retain the exact upstream source commit URL and
license text. Unknown omissions stop packaging rather than substituting a license name.

Local dirty-source builds are labelled in BUILDINFO and remain unsigned. A release tag
requires a clean tracked source tree, version equality with the compiled executable,
exact tag-to-HEAD agreement and ancestry on `origin/main`. Source code signing does not
automatically qualify model quality, scientific results or privacy claims.

## Signing workflow

`.github/workflows/engine-release.yml` has a read-only build/test job for pull requests
and main. Only a push of `dope-engine-v<engine version>` can enter the release job.
That job checks source ancestry before building and uses a commit-pinned `actions/attest`
with GitHub OIDC and short-lived Sigstore credentials. There is no long-lived signing
key to commit or distribute. It signs build provenance whose subjects include the exact
binary, checksum inventory and remaining fixed package members.

After signing, the job verifies both the binary and SHA256SUMS against:

- Repository `neverhuman/dope`.
- Signer workflow `neverhuman/dope/.github/workflows/engine-release.yml`.
- Exact `refs/tags/dope-engine-v<version>` source ref.
- Exact source commit being released.

There is no checksum-only or unsigned fallback. A signing or verification failure stops
publication. The workflow creates a **draft** GitHub Release with explicit assets;
maintainers publish it after the existing repository release gates and review. It does
not sign PR builds or create a public release merely because a PR was opened.

For maintainers, after merging and passing the applicable repository gates:

```bash
# Set the engine version first, then create the matching tag from admitted main.
git tag -a dope-engine-v1.0.0 -m 'DOPE engine 1.0.0'
git push origin dope-engine-v1.0.0
# Review the generated signed draft, its source commit and all required checks.
# Publish that draft through the normal release process.
```

This PR implements the workflow; actual GitHub OIDC issuance, transparency-log
registration and signed release publication are measured only when a trusted tag runs.
Local policy fixtures use a fake verification command to prove argument binding and
failure propagation. They are not cryptographic signing evidence.

## Verify before install

Install GitHub CLI from its official distribution. From this source checkout:

```bash
release_tag=dope-engine-v1.0.0
gh release download "$release_tag" --repo neverhuman/dope --dir release
release_commit=$(gh api "repos/neverhuman/dope/commits/$release_tag" --jq .sha)
bash scripts/verify-engine-release.sh release "$release_tag" "$release_commit"
# Only after successful verification:
install -Dm755 release/dope-linux-x86_64 "$HOME/.local/bin/dope"
install -Dm644 release/LICENSE "$HOME/.local/share/dope-engine/LICENSE"
install -Dm644 release/THIRD_PARTY_NOTICES "$HOME/.local/share/dope-engine/THIRD_PARTY_NOTICES"
```

The helper verifies the two signed subjects online using `gh attestation verify`, then
checks every fixed package hash and refuses symlinks, traversal inventory entries,
missing files and malformed tag/commit identities. It never executes the unverified
binary. The GitHub CLI is an installation-time verifier, not an application dependency.
Retain the bundle and exact source commit with your deployment record. Offline bundle
verification is possible with GitHub CLI's `--bundle` mode, but is not an unsigned
fallback and requires maintaining the correct trust policy independently.

For direct verification without the helper, repeat this command for `SHA256SUMS` as well
as the binary, then check the signed checksum inventory before installing:

```bash
gh attestation verify release/dope-linux-x86_64 --repo neverhuman/dope \
  --signer-workflow neverhuman/dope/.github/workflows/engine-release.yml \
  --source-ref "refs/tags/$release_tag" --source-digest "$release_commit" --format json
```

## Checks

- `bash ops/ci/engine.sh`: Rust formatting/lints/tests, without Python requirements.
- `bash ops/ci/engine-benchmark.sh`: separate offline Elo evidence checks; hosted CPU
  invokes its provisioned Python interpreter after installing NumPy.
- `bash ops/ci/engine-release.sh`: CPU graph, actual optimized executable, notice collection
  and complete customer-command smoke with no executable helpers on PATH.
- Rust integration tests: JSON stdout/file agreement, runtime digest and worker count,
  raw/ML output, predictions, explicit argument errors, verifier rejection paths and
  forbidden dependency examples.
- `actionlint` and `bash -n`: workflow and release-script checks.

Primary implementation references:
[GitHub signed attestations](https://github.com/actions/attest),
[GitHub CLI verification policy](https://cli.github.com/manual/gh_attestation_verify), and
[GitHub provenance documentation](https://docs.github.com/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds).
