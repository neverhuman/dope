#!/usr/bin/env bash
# Verify signatures before chmod/install/execute. Signature verification is required.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: verify-engine-release.sh RELEASE_DIRECTORY dope-engine-vVERSION SOURCE_COMMIT' >&2
  exit 2
fi
release_dir=$1
tag=$2
commit=$3
[[ "$tag" =~ ^dope-engine-v[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ && "$commit" =~ ^[0-9a-f]{40}$ ]] || {
  echo 'engine_release_identity_invalid' >&2; exit 2;
}
gh_upgrade_error() {
  echo "error: upgrade gh to 2.102.0 or newer with attestation verify --repo, --signer-workflow, --source-ref, --source-digest, --bundle and --format support" >&2
  echo "  https://github.com/cli/cli/releases/tag/v2.102.0" >&2
  echo "  note: Ubuntu 24.04's packaged gh is 2.45.0; use the release above or GitHub's apt repository" >&2
  exit 1
}
require_gh_policy() {
  local gh_version gh_help major minor flag
  command -v gh >/dev/null 2>&1 || gh_upgrade_error
  gh_version=$(gh --version 2>/dev/null) || gh_upgrade_error
  [[ "$gh_version" =~ ^gh[[:space:]]version[[:space:]]([0-9]+)\.([0-9]+)\.([0-9]+)([[:space:]]|$) ]] || gh_upgrade_error
  major=$((10#${BASH_REMATCH[1]})); minor=$((10#${BASH_REMATCH[2]}))
  (( major > 2 || (major == 2 && minor >= 102) )) || gh_upgrade_error
  gh_help=$(gh attestation verify --help 2>/dev/null) || gh_upgrade_error
  for flag in --repo --signer-workflow --source-ref --source-digest --format --bundle; do
    [[ "$gh_help" =~ (^|[[:space:]])${flag}([[:space:]=,]|$) ]] || gh_upgrade_error
  done
}
require_gh_policy
[[ -f "$release_dir/dope.sigstore.json" && ! -L "$release_dir/dope.sigstore.json" ]] || {
  echo 'engine_release_attestation_bundle_missing' >&2; exit 1;
}
for name in dope-linux-x86_64 SHA256SUMS; do
  [[ -s "$release_dir/$name" && -f "$release_dir/$name" && ! -L "$release_dir/$name" ]] || {
    echo 'engine_release_required_file_missing' >&2; exit 1;
  }
done
for name in dope-linux-x86_64 SHA256SUMS; do
  gh attestation verify "$release_dir/$name" --bundle "$release_dir/dope.sigstore.json" --repo neverhuman/dope \
    --signer-workflow neverhuman/dope/.github/workflows/engine-release.yml \
    --source-ref "refs/tags/$tag" --source-digest "$commit" --format json > /dev/null
done
# Only fixed package members may be named, even inside a signed checksum file.
expected_core=(BUILDINFO.json Cargo.lock DEPENDENCIES.txt LICENSE THIRD_PARTY_NOTICES dope-linux-x86_64)
expected_with_v8=(BUILDINFO.json Cargo.lock DEPENDENCIES.txt LICENSE THIRD_PARTY_NOTICES V8_WEIGHTS_MANIFEST.json dope-linux-x86_64 dope-v8-weights.safetensors)
mapfile -t lines < "$release_dir/SHA256SUMS"
have_v8_weights=0
if [[ ${#lines[@]} == "${#expected_core[@]}" ]]; then
  expected=("${expected_core[@]}")
  for stray in V8_WEIGHTS_MANIFEST.json dope-v8-weights.safetensors; do
    [[ ! -e "$release_dir/$stray" && ! -L "$release_dir/$stray" ]] || { echo 'engine_release_stray_v8_weights_forbidden' >&2; exit 1; }
  done
elif [[ ${#lines[@]} == "${#expected_with_v8[@]}" ]]; then
  expected=("${expected_with_v8[@]}")
  have_v8_weights=1
else
  echo 'engine_release_inventory_invalid' >&2; exit 1
fi
for i in "${!expected[@]}"; do
  digest=${lines[$i]:0:64}
  [[ "$digest" =~ ^[0-9a-f]{64}$ && "${lines[$i]}" == "$digest  ${expected[$i]}" ]] || { echo 'engine_release_inventory_invalid' >&2; exit 1; }
  [[ -f "$release_dir/${expected[$i]}" && ! -L "$release_dir/${expected[$i]}" ]] || exit 1
done
(cd -- "$release_dir" && sha256sum --check --status SHA256SUMS)
if [[ $have_v8_weights -eq 1 ]]; then
  command -v jq > /dev/null || { echo 'engine_release_jq_required' >&2; exit 1; }
  for name in dope-v8-weights.safetensors V8_WEIGHTS_MANIFEST.json; do
    [[ -s "$release_dir/$name" ]] || { echo 'engine_release_v8_weights_empty' >&2; exit 1; }
    gh attestation verify "$release_dir/$name" --bundle "$release_dir/dope.sigstore.json" --repo neverhuman/dope \
      --signer-workflow neverhuman/dope/.github/workflows/engine-release.yml \
      --source-ref "refs/tags/$tag" --source-digest "$commit" --format json > /dev/null
  done
  v8_sha=$(sha256sum "$release_dir/dope-v8-weights.safetensors"); v8_sha=${v8_sha%% *}
  v8_bytes=$(wc -c < "$release_dir/dope-v8-weights.safetensors" | tr -d ' ')
  jq -e --arg sha "$v8_sha" --argjson bytes "$v8_bytes" \
    '.schema == "dope.single-enc-v8-weights/v1" and .file == "model.safetensors" and .arch == "single_enc_v8" and .sha256 == $sha and .bytes == $bytes and .bytes > 0 and .bytes <= 1073741824' \
    "$release_dir/V8_WEIGHTS_MANIFEST.json" > /dev/null || { echo 'engine_release_v8_weights_manifest_invalid' >&2; exit 1; }
  jq -e --arg sha "$v8_sha" --argjson bytes "$v8_bytes" \
    '.v8_weights_sha256 == $sha and .v8_weights_bytes == $bytes' \
    "$release_dir/BUILDINFO.json" > /dev/null || { echo 'engine_release_buildinfo_v8_weights_mismatch' >&2; exit 1; }
  printf '%s\n' 'Verified signed Rust engine release with SingleEncV8 weights; binary, weights and package hashes match.'
else
  printf '%s\n' 'Verified signed Rust engine release; binary and package hashes match.'
fi
