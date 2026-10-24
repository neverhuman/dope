#!/usr/bin/env bash
# Build the research-kernel `dope-kernel` executable of one commit reproducibly. Signing is
# performed by .github/workflows/kernel-release.yml; this script never signs or publishes.
#
#   bash ops/ci/kernel-release.sh SOURCE_COMMIT [NEW_OUTPUT_DIRECTORY]
#
# The commit is exported with `git archive` and built inside a digest-pinned rust:1.88.0 image
# with fixed container paths, so the same commit gives the same bytes on any x86-64 Docker host
# (docs/signed-kernel-release.md). The working tree is never mounted.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $# -ge 1 && $# -le 2 ]] || { echo 'usage: kernel-release.sh SOURCE_COMMIT [NEW_OUTPUT_DIRECTORY]' >&2; exit 2; }
[[ "$(uname -s):$(uname -m)" == Linux:x86_64 ]] || { echo 'kernel_release_platform_unsupported' >&2; exit 1; }
for tool in git docker jq sha256sum objdump tar; do
  command -v "$tool" > /dev/null || { echo "kernel_release_tool_missing: $tool" >&2; exit 1; }
done
commit=$(git rev-parse --verify "$1^{commit}") || { echo 'kernel_release_commit_unknown' >&2; exit 1; }
out=${2:-target/kernel-release}
[[ ! -e "$out" && ! -L "$out" ]] || { echo 'kernel_release_output_exists' >&2; exit 1; }

# rust:1.88.0-bookworm (glibc 2.36), the toolchain rust-toolchain.toml names. Changing it changes
# the bytes; it is a reviewed edit here and in docs/signed-kernel-release.md.
builder='rust@sha256:af306cfa71d987911a781c37b59d7d67d934f49684058f96cf72079c3626bfe0'
glibc_ceiling=2.36
rustflags='--remap-path-prefix=/src=/build/dope --remap-path-prefix=/cargo/registry/src=/cargo-src'

mkdir -p "$out"/{src,cargo,target,dist}
out=$(cd "$out" && pwd)
git archive --format=tar "$commit" | tar -x -C "$out/src"
version=$(sed -n '/^\[package\]/,/^\[/s/^version = "\(.*\)"$/\1/p' "$out/src/Cargo.toml" | head -n1)
[[ -n "$version" ]] || { echo 'kernel_release_version_unreadable' >&2; exit 1; }

docker run --rm --network host -u "$(id -u):$(id -g)" \
  -e CARGO_HOME=/cargo -e CARGO_INCREMENTAL=0 -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
  -e RUSTFLAGS="$rustflags" \
  -v "$out/src":/src:ro -v "$out/cargo":/cargo -v "$out/target":/target -w /src \
  "$builder" cargo build --release --locked --bin dope-kernel --target-dir /target
binary="$out/dist/dope-kernel"
cp -- "$out/target/release/dope-kernel" "$binary"
cp -- "$out/src/Cargo.lock" "$out/dist/Cargo.lock"

# Only ordinary OS libraries may be linked; no CUDA, libtorch or Python reaches this build.
needed=$(objdump -p "$binary" | awk '$1 == "NEEDED" {print $2}' | sort | tr '\n' ' ')
[[ "$needed" == 'libc.so.6 libgcc_s.so.1 libm.so.6 ' ]] || { echo "kernel_release_linkage_refused: $needed" >&2; exit 1; }
glibc_max=$(objdump -T "$binary" | grep -oE 'GLIBC_[0-9.]+' | sed 's/GLIBC_//' | sort -uV | tail -n1)
[[ "$(printf '%s\n%s\n' "$glibc_max" "$glibc_ceiling" | sort -V | tail -n1)" == "$glibc_ceiling" ]] || {
  echo "kernel_release_glibc_above_ceiling: $glibc_max" >&2; exit 1; }
[[ "$("$binary" version | head -n1)" == "dope-kernel $version" ]] || { echo 'kernel_release_version_mismatch' >&2; exit 1; }

jq -n --arg commit "$commit" --arg version "$version" --arg builder "$builder" \
  --arg rustflags "$rustflags" --arg glibc_max "$glibc_max" --arg needed "$needed" \
  --arg sha256 "$(sha256sum "$binary" | cut -c1-64)" \
  '{schema: "dope.kernel-release.v1", source_commit: $commit, crate: "dope-kernel", version: $version,
    builder_image: $builder, command: "cargo build --release --locked --bin dope-kernel",
    rustflags: $rustflags, glibc_max: $glibc_max, needed: ($needed | split(" ") | map(select(. != ""))),
    dope_kernel_sha256: $sha256,
    claims: "build identity only; not a certified, privacy-qualified or tier-certified release"}' \
  > "$out/dist/BUILDINFO.json"
(cd "$out/dist" && sha256sum BUILDINFO.json Cargo.lock dope-kernel > SHA256SUMS)
cat "$out/dist/SHA256SUMS"
