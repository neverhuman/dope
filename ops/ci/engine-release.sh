#!/usr/bin/env bash
# Build the customer CPU CLI only. Signing is performed by the trusted tag job.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $# -le 1 ]] || { echo 'usage: engine-release.sh [NEW_OUTPUT_DIRECTORY]' >&2; exit 2; }
[[ "$(uname -s):$(uname -m)" == Linux:x86_64 ]] || { echo 'engine_release_platform_unsupported' >&2; exit 1; }
for tool in cargo rustc jq sha256sum readelf; do
  command -v "$tool" > /dev/null || { echo "engine_release_tool_missing: $tool" >&2; exit 1; }
done
out=${1:-target/engine-release}
[[ ! -e "$out" && ! -L "$out" ]] || { echo 'engine_release_output_exists' >&2; exit 1; }
target=x86_64-unknown-linux-gnu
manifest=engine/Cargo.toml
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$PWD/engine/target"}
export CARGO_INCREMENTAL=0
export RUSTFLAGS='-C target-cpu=x86-64'
unset CARGO_ENCODED_RUSTFLAGS
mkdir -p target/tmp
stage=$(mktemp -d "$PWD/target/tmp/engine-package.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
cargo tree --manifest-path "$manifest" --locked --no-default-features -p dope-cli \
  --target "$target" -e normal,build --prefix none --format '{p}' | \
  awk '{print $1 " " $2}' | sort -u > "$stage/DEPENDENCIES.txt"
bash ops/ci/engine-runtime-graph.sh "$stage/DEPENDENCIES.txt"
cargo test --manifest-path "$manifest" --locked --no-default-features --target "$target"
cargo build --manifest-path "$manifest" --locked --release --no-default-features --target "$target" --bin dope
binary="$CARGO_TARGET_DIR/$target/release/dope"
bash ops/ci/engine-release-smoke.sh "$binary"
# An explicit shared-library inventory rejects accidental Python/libtorch linkage.
readelf -d "$binary" > "$stage/elf.txt"
if grep -Ei 'NEEDED.*(python|torch|cuda|mkl)' "$stage/elf.txt"; then exit 1; fi
version=$("$binary" --version)
version=${version#'dope '}
source_commit=$(git rev-parse HEAD)
dirty=false
if ! git diff --quiet || ! git diff --cached --quiet; then dirty=true; fi
if [[ -n "${ENGINE_RELEASE_TAG:-}" ]]; then
  [[ "$ENGINE_RELEASE_TAG" == "dope-engine-v$version" && "$dirty" == false ]] || { echo 'engine_release_tag_or_source_invalid' >&2; exit 1; }
  [[ $(git rev-parse "$ENGINE_RELEASE_TAG^{commit}") == "$source_commit" ]] || exit 1
  git merge-base --is-ancestor "$source_commit" refs/remotes/origin/main
fi
cargo metadata --manifest-path "$manifest" --locked --format-version 1 --filter-platform "$target" > "$stage/metadata.json"
bash ops/ci/engine-notices.sh "$stage/DEPENDENCIES.txt" "$stage/metadata.json" "$stage/THIRD_PARTY_NOTICES"
cp "$binary" "$stage/dope-linux-x86_64"
cp engine/Cargo.lock "$stage/Cargo.lock"
cp LICENSE "$stage/LICENSE"
binary_digest=$(sha256sum "$binary"); binary_digest=${binary_digest%% *}
lock_digest=$(sha256sum engine/Cargo.lock); lock_digest=${lock_digest%% *}
jq -n --arg version "$version" --arg commit "$source_commit" --arg target "$target" \
  --arg digest "$binary_digest" --arg lock "$lock_digest" --arg rustc "$(rustc --version)" --argjson dirty "$dirty" \
  '{schema_version:1,product:"dope-engine",version:$version,source_commit:$commit,target:$target,
    implementation:"native_rust",python_required:false,libtorch_required:false,features:["cpu"],
    rustc:$rustc,dirty_source:$dirty,binary_sha256:$digest,cargo_lock_sha256:$lock,
    signature:"external-sigstore-attestation-required"}' > "$stage/BUILDINFO.json"
(cd "$stage" && sha256sum BUILDINFO.json Cargo.lock DEPENDENCIES.txt LICENSE THIRD_PARTY_NOTICES dope-linux-x86_64 > SHA256SUMS)
mkdir -p -- "$(dirname -- "$out")"
# Publish the staged directory only after all checks and notice collection pass.
mkdir -- "$out"
for name in BUILDINFO.json Cargo.lock DEPENDENCIES.txt LICENSE THIRD_PARTY_NOTICES dope-linux-x86_64 SHA256SUMS; do
  cp -- "$stage/$name" "$out/$name"
done
printf '%s\n' "Unsigned build staged at $out; deployment requires successful release attestation verification."
