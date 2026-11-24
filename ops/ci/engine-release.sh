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
v8_weights_dir=models/single-enc-v8
v8_weights_explicit=0
if [[ -n "${DOPE_V8_WEIGHTS_SOURCE_DIR:-}" ]]; then
  [[ -z "${ENGINE_RELEASE_TAG:-}" ]] || { echo 'engine_release_v8_weights_override_forbidden_on_tag' >&2; exit 1; }
  v8_weights_dir=$DOPE_V8_WEIGHTS_SOURCE_DIR
  v8_weights_explicit=1
fi
have_v8_weights=0
v8_manifest=""
v8_model=""
v8_weights_sha=""
v8_weights_bytes=0
if [[ -L "$v8_weights_dir" ]]; then
  echo 'engine_release_v8_weights_symlink_forbidden' >&2; exit 1
elif [[ -d "$v8_weights_dir" ]]; then
  v8_manifest="$v8_weights_dir/manifest.json"
  v8_model="$v8_weights_dir/model.safetensors"
  [[ -f "$v8_manifest" && ! -L "$v8_manifest" && -f "$v8_model" && ! -L "$v8_model" ]] || {
    echo 'engine_release_v8_weights_incomplete' >&2; exit 1
  }
  jq -e '.schema == "dope.single-enc-v8-weights/v1" and .file == "model.safetensors" and .arch == "single_enc_v8" and (.bytes | type == "number" and floor == . and . > 0 and . <= 1073741824) and (.sha256 | type == "string" and test("^[0-9a-f]{64}$"))' \
    "$v8_manifest" > /dev/null || { echo 'engine_release_v8_weights_manifest_invalid' >&2; exit 1; }
  v8_weights_sha=$(sha256sum "$v8_model"); v8_weights_sha=${v8_weights_sha%% *}
  v8_weights_bytes=$(wc -c < "$v8_model" | tr -d ' ')
  jq -e --arg sha "$v8_weights_sha" --argjson bytes "$v8_weights_bytes" \
    '.sha256 == $sha and .bytes == $bytes' "$v8_manifest" > /dev/null || {
    echo 'engine_release_v8_weights_digest_mismatch' >&2; exit 1
  }
  have_v8_weights=1
elif [[ -e "$v8_weights_dir" ]]; then
  echo 'engine_release_v8_weights_invalid_path' >&2; exit 1
elif [[ $v8_weights_explicit -eq 1 ]]; then
  echo 'engine_release_v8_weights_override_missing' >&2; exit 1
fi
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
cargo test --manifest-path "$manifest" --locked --no-default-features --target "$target" --no-fail-fast
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

if [[ $have_v8_weights -eq 1 ]]; then
  # Exercise the real release binary against the staged SingleEncV8 weights with an empty tool PATH.
  mkdir -p "$stage/no-tools"
  awk 'BEGIN {print "a,b,y"; for(i=0;i<160;i++){a=(i*17%997)/997;b=(i*31%991)/991;print a "," b "," 2*a+b*b}}' > "$stage/v8-train.csv"
  env -i PATH="$stage/no-tools" "$binary" --threads 2 fit "$stage/v8-train.csv" \
    --target y --task regression --format v8 --budget 2048 --device cpu \
    --v8-weights "$v8_model" -o "$stage/v8-weighted.dope"
  v8_art_bytes=$(wc -c < "$stage/v8-weighted.dope")
  (( v8_art_bytes > 0 && v8_art_bytes <= 2048 ))
  env -i PATH="$stage/no-tools" "$binary" --threads 2 sample "$stage/v8-weighted.dope" \
    -n 11 --device cpu --v8-weights "$v8_model" -o "$stage/v8-weighted.csv"
  [[ $(wc -l < "$stage/v8-weighted.csv") == 12 ]] || { echo 'engine_release_v8_weights_smoke_failed' >&2; exit 1; }
  env -i PATH="$stage/no-tools" "$binary" --threads 2 fit "$stage/v8-train.csv" \
    --target y --task regression --format auto-v8 --budget 2048 --device cpu \
    --v8-weights "$v8_model" -o "$stage/v8-auto.dope"
  v8_auto_bytes=$(wc -c < "$stage/v8-auto.dope")
  (( v8_auto_bytes > 0 && v8_auto_bytes <= 2048 ))
  cp -- "$v8_manifest" "$stage/V8_WEIGHTS_MANIFEST.json"
  cp -- "$v8_model" "$stage/dope-v8-weights.safetensors"
fi

if [[ $have_v8_weights -eq 1 ]]; then
  jq -n --arg version "$version" --arg commit "$source_commit" --arg target "$target" \
    --arg digest "$binary_digest" --arg lock "$lock_digest" --arg rustc "$(rustc --version)" \
    --argjson dirty "$dirty" --arg v8_sha "$v8_weights_sha" --argjson v8_bytes "$v8_weights_bytes" \
    '{schema_version:1,product:"dope-engine",version:$version,source_commit:$commit,target:$target,
      implementation:"native_rust",python_required:false,libtorch_required:false,features:["cpu"],
      rustc:$rustc,dirty_source:$dirty,binary_sha256:$digest,cargo_lock_sha256:$lock,
      v8_weights_sha256:$v8_sha,v8_weights_bytes:$v8_bytes,
      signature:"external-sigstore-attestation-required"}' > "$stage/BUILDINFO.json"
  members=(BUILDINFO.json Cargo.lock DEPENDENCIES.txt LICENSE THIRD_PARTY_NOTICES V8_WEIGHTS_MANIFEST.json dope-linux-x86_64 dope-v8-weights.safetensors)
else
  jq -n --arg version "$version" --arg commit "$source_commit" --arg target "$target" \
    --arg digest "$binary_digest" --arg lock "$lock_digest" --arg rustc "$(rustc --version)" --argjson dirty "$dirty" \
    '{schema_version:1,product:"dope-engine",version:$version,source_commit:$commit,target:$target,
      implementation:"native_rust",python_required:false,libtorch_required:false,features:["cpu"],
      rustc:$rustc,dirty_source:$dirty,binary_sha256:$digest,cargo_lock_sha256:$lock,
      signature:"external-sigstore-attestation-required"}' > "$stage/BUILDINFO.json"
  members=(BUILDINFO.json Cargo.lock DEPENDENCIES.txt LICENSE THIRD_PARTY_NOTICES dope-linux-x86_64)
fi
(cd "$stage" && sha256sum "${members[@]}" > SHA256SUMS)
mkdir -p -- "$(dirname -- "$out")"
# Publish the staged directory only after all checks and notice collection pass.
mkdir -- "$out"
for name in "${members[@]}" SHA256SUMS; do
  cp -- "$stage/$name" "$out/$name"
done
printf '%s\n' "Unsigned build staged at $out; deployment requires successful release attestation verification."
