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
for name in dope-linux-x86_64 SHA256SUMS; do
  [[ -s "$release_dir/$name" && -f "$release_dir/$name" && ! -L "$release_dir/$name" ]] || {
    echo 'engine_release_required_file_missing' >&2; exit 1;
  }
done
command -v gh > /dev/null || { echo 'engine_release_github_cli_required' >&2; exit 1; }
for name in dope-linux-x86_64 SHA256SUMS; do
  gh attestation verify "$release_dir/$name" --repo neverhuman/dope \
    --signer-workflow neverhuman/dope/.github/workflows/engine-release.yml \
    --source-ref "refs/tags/$tag" --source-digest "$commit" --format json > /dev/null
done
# Only fixed package members may be named, even inside a signed checksum file.
expected=(BUILDINFO.json Cargo.lock DEPENDENCIES.txt LICENSE THIRD_PARTY_NOTICES dope-linux-x86_64)
mapfile -t lines < "$release_dir/SHA256SUMS"
[[ ${#lines[@]} == ${#expected[@]} ]] || { echo 'engine_release_inventory_invalid' >&2; exit 1; }
for i in "${!expected[@]}"; do
  digest=${lines[$i]:0:64}
  [[ "$digest" =~ ^[0-9a-f]{64}$ && "${lines[$i]}" == "$digest  ${expected[$i]}" ]] || { echo 'engine_release_inventory_invalid' >&2; exit 1; }
  [[ -f "$release_dir/${expected[$i]}" && ! -L "$release_dir/${expected[$i]}" ]] || exit 1
done
(cd -- "$release_dir" && sha256sum --check --status SHA256SUMS)
printf '%s\n' 'Verified signed Rust engine release; binary and package hashes match.'
