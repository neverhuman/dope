#!/usr/bin/env bash
# Verify a signed research-kernel release before chmod/install/execute.
#
#   scripts/verify-kernel-release.sh RELEASE_DIRECTORY dope-kernel-vVERSION SOURCE_COMMIT
#
# The attestation proves the files were made by kernel-release.yml running on main; BUILDINFO.json
# (covered by the signed SHA256SUMS) names the commit that was built. Rebuilding that commit with
# ops/ci/kernel-release.sh gives the same dope-kernel bytes (docs/signed-kernel-release.md).
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: verify-kernel-release.sh RELEASE_DIRECTORY dope-kernel-vVERSION SOURCE_COMMIT' >&2
  exit 2
fi
release_dir=$1
tag=$2
commit=$3
[[ "$tag" =~ ^dope-kernel-v[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ && "$commit" =~ ^[0-9a-f]{40}$ ]] || {
  echo 'kernel_release_identity_invalid' >&2; exit 2;
}
expected=(BUILDINFO.json Cargo.lock dope-kernel)
for name in "${expected[@]}" SHA256SUMS; do
  [[ -s "$release_dir/$name" && -f "$release_dir/$name" && ! -L "$release_dir/$name" ]] || {
    echo 'kernel_release_required_file_missing' >&2; exit 1;
  }
done
command -v gh > /dev/null || { echo 'kernel_release_github_cli_required' >&2; exit 1; }
command -v jq > /dev/null || { echo 'kernel_release_jq_required' >&2; exit 1; }
for name in dope-kernel SHA256SUMS; do
  gh attestation verify "$release_dir/$name" --repo neverhuman/dope \
    --signer-workflow neverhuman/dope/.github/workflows/kernel-release.yml \
    --source-ref refs/heads/main --format json > /dev/null
done
# Only fixed package members may be named, even inside a signed checksum file.
mapfile -t lines < "$release_dir/SHA256SUMS"
[[ "${#lines[@]}" == "${#expected[@]}" ]] || { echo 'kernel_release_inventory_invalid' >&2; exit 1; }
for i in "${!expected[@]}"; do
  digest=${lines[$i]:0:64}
  [[ "$digest" =~ ^[0-9a-f]{64}$ && "${lines[$i]}" == "$digest  ${expected[$i]}" ]] || { echo 'kernel_release_inventory_invalid' >&2; exit 1; }
done
(cd -- "$release_dir" && sha256sum --check --status SHA256SUMS)
version=${tag#dope-kernel-v}
jq -e --arg commit "$commit" --arg version "$version" \
  '.schema == "dope.kernel-release.v1" and .source_commit == $commit and .version == $version' \
  "$release_dir/BUILDINFO.json" > /dev/null || { echo 'kernel_release_buildinfo_mismatch' >&2; exit 1; }
printf '%s\n' "Verified signed research-kernel release $tag of $commit; package hashes match."
