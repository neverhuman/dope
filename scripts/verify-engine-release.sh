#!/usr/bin/env bash
# Verify signatures before chmod/install/execute. Signature verification is required.
set -euo pipefail
# shellcheck source=scripts/install.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/install.sh"
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
check_public_source "$tag" "$commit"
require_tools sha256sum jq
GH_VERIFIER=$(command -v gh) || gh_upgrade_error
gh_policy_supported "$GH_VERIFIER" || gh_upgrade_error
for name in dope-linux-x86_64 SHA256SUMS; do
  [[ -s "$release_dir/$name" && -f "$release_dir/$name" && ! -L "$release_dir/$name" ]] || {
    echo 'engine_release_required_file_missing' >&2; exit 1;
  }
  verify_attestation "$release_dir" "$tag" "$commit" "$name"
done
# Only fixed package members may be named, even inside a signed checksum file.
engine_inventory "$release_dir"
have_v8_weights=0
if [[ ${#EXPECTED_ENGINE[@]} -eq 6 ]]; then
  for stray in V8_WEIGHTS_MANIFEST.json dope-v8-weights.safetensors; do
    [[ ! -e "$release_dir/$stray" && ! -L "$release_dir/$stray" ]] || { echo 'engine_release_stray_v8_weights_forbidden' >&2; exit 1; }
  done
else
  have_v8_weights=1
fi
for name in "${EXPECTED_ENGINE[@]}"; do
  [[ -f "$release_dir/$name" && ! -L "$release_dir/$name" ]] ||
    die "missing or unsafe release asset: $name"
done
(cd -- "$release_dir" && sha256sum --check --status SHA256SUMS) ||
  die "engine release checksum mismatch"
if [[ $have_v8_weights -eq 1 ]]; then
  for name in dope-v8-weights.safetensors V8_WEIGHTS_MANIFEST.json; do
    [[ -s "$release_dir/$name" ]] || { echo 'engine_release_v8_weights_empty' >&2; exit 1; }
    verify_attestation "$release_dir" "$tag" "$commit" "$name"
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
