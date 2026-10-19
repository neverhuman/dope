#!/usr/bin/env bash
# Install a caller-selected, checksum-pinned already-built native binary.
set -euo pipefail
if [[ $# != 3 && $# != 4 ]]; then
  printf '%s\n' 'usage: install-dope.sh BINARY SHA256 PREFIX [--dry-run]' >&2
  exit 2
fi
binary=$1
expected=$2
prefix=${3%/}
if [[ "$prefix" != /* || "$prefix" == / || "$prefix" == *'/../'* || "$prefix" == */.. ]]; then
  printf '%s\n' 'dope_install_absolute_clean_prefix_required' >&2; exit 1
fi
ancestor=$prefix
while [[ "$ancestor" != / ]]; do
  [[ ! -L "$ancestor" ]] || { printf '%s\n' 'dope_install_destination_symlink_refused' >&2; exit 1; }
  ancestor=$(dirname -- "$ancestor")
done
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
legal_dir="$prefix/share/dope"
if [[ $# == 4 && "$4" != --dry-run ]]; then
  printf '%s\n' 'dope_install_option_invalid' >&2
  exit 2
fi
if [[ ! "$expected" =~ ^[0-9a-f]{64}$ || ! -f "$binary" || -L "$binary" || ! -x "$binary" ]]; then
  printf '%s\n' 'dope_install_input_invalid' >&2
  exit 1
fi
case "$(uname -s):$(uname -m)" in
  Linux:x86_64) ;;
  *) printf '%s\n' 'dope_install_platform_unqualified' >&2; exit 1 ;;
esac
actual=$(sha256sum -- "$binary")
actual=${actual%% *}
if [[ "$actual" != "$expected" ]]; then
  printf '%s\n' 'dope_install_checksum_mismatch' >&2
  exit 1
fi
for parent in "$prefix" "$prefix/bin" "$prefix/share" "$legal_dir"; do
  if [[ -L "$parent" ]]; then
    printf '%s\n' 'dope_install_destination_symlink_refused' >&2
    exit 1
  fi
done
for destination in "$prefix/bin/dope" "$legal_dir/native.sha256" \
  "$legal_dir/LICENSE" "$legal_dir/THIRD_PARTY_NOTICES" "$legal_dir/installed.sha256"; do
  if [[ -e "$destination" || -L "$destination" ]]; then
    printf '%s\n' 'dope_install_destination_exists_use_new_prefix' >&2
    exit 1
  fi
done
# These frozen inputs come from the locked offline package graph, not ambient
# adapter notices. Validate them before creating any destination or candidate.
for input in LICENSE scripts/native-third-party.NOTICE scripts/native-legal.sha256 Cargo.lock rust-toolchain.toml; do
  if [[ ! -f "$package_root/$input" || -L "$package_root/$input" ]]; then
    printf '%s\n' 'dope_install_legal_inputs_unavailable' >&2
    exit 1
  fi
done
# Never let a package manifest choose paths for checksum traversal.
if [[ $(stat -c %s -- "$package_root/scripts/native-legal.sha256") -gt 1024 ]]; then
  printf '%s\n' 'dope_install_legal_manifest_invalid' >&2
  exit 1
fi
mapfile -t package_inputs < "$package_root/scripts/native-legal.sha256"
if [[ ${#package_inputs[@]} != 4 ]]; then
  printf '%s\n' 'dope_install_legal_manifest_invalid' >&2
  exit 1
fi
fixed_inputs=(LICENSE scripts/native-third-party.NOTICE Cargo.lock rust-toolchain.toml)
for i in "${!fixed_inputs[@]}"; do
  digest=${package_inputs[$i]:0:64}
  if [[ ! "$digest" =~ ^[0-9a-f]{64}$ || "${package_inputs[$i]}" != "$digest  ${fixed_inputs[$i]}" ]]; then
    printf '%s\n' 'dope_install_legal_manifest_invalid' >&2
    exit 1
  fi
done
for i in "${!fixed_inputs[@]}"; do
  actual=$(sha256sum -- "$package_root/${fixed_inputs[$i]}")
  if [[ "${actual%% *}" != "${package_inputs[$i]:0:64}" ]]; then
    printf '%s\n' 'dope_install_legal_inputs_checksum_mismatch' >&2
    exit 1
  fi
done
lock_digest=${package_inputs[2]:0:64}
toolchain_digest=${package_inputs[3]:0:64}
# Check product identity before writing any destination. No package manager,
# Python environment, optional dependency installation, or network is used.
identity=$("$binary" doctor)
required_identity=(
  '"os"[[:space:]]*:[[:space:]]*"linux"'
  '"architecture"[[:space:]]*:[[:space:]]*"x86_64"'
  '"implementation"[[:space:]]*:[[:space:]]*"native_rust"'
  '"name"[[:space:]]*:[[:space:]]*"dope"'
  "\"executable_sha256\"[[:space:]]*:[[:space:]]*\"$expected\""
)
for field in "${required_identity[@]}"; do
  if [[ ! "$identity" =~ $field ]]; then
    printf '%s\n' 'dope_install_identity_mismatch' >&2
    exit 1
  fi
done
required_profile=(
  "\"build_cargo_lock_sha256\"[[:space:]]*:[[:space:]]*\"$lock_digest\""
  "\"declared_rust_toolchain_sha256\"[[:space:]]*:[[:space:]]*\"$toolchain_digest\""
  '"gpu_training"[[:space:]]*:[[:space:]]*false[[:space:]]*[,}]'
)
for field in "${required_profile[@]}"; do
  if [[ ! "$identity" =~ $field ]]; then
    printf '%s\n' 'dope_install_package_profile_or_build_inputs_mismatch' >&2
    exit 1
  fi
done
if [[ $# == 4 ]]; then
  printf '%s\n' '{"operation":"would_install","implementation":"native_rust","network":false}'
  exit 0
fi
mkdir -p -- "$prefix/bin" "$prefix/share/dope"
candidate="$prefix/bin/.dope-$expected.pending"
if ! (set -o noclobber; : > "$candidate") 2>/dev/null; then
  printf '%s\n' 'dope_install_candidate_exists_preserve_prior_attempt' >&2
  exit 1
fi
cat -- "$binary" > "$candidate"
chmod 0755 -- "$candidate"
installed=$(sha256sum -- "$candidate")
if [[ "${installed%% *}" != "$expected" ]]; then
  printf '%s\n' 'dope_install_copied_checksum_mismatch' >&2
  exit 1
fi
mv -T --no-clobber -- "$candidate" "$prefix/bin/dope"
if [[ -e "$candidate" ]]; then
  printf '%s\n' 'dope_install_destination_appeared_preserve_candidate' >&2
  exit 1
fi
printf '%s\n' "$expected" > "$prefix/share/dope/native.sha256"
for entry in LICENSE THIRD_PARTY_NOTICES; do
  source="$package_root/LICENSE"
  legal_digest=${package_inputs[0]%% *}
  if [[ "$entry" == THIRD_PARTY_NOTICES ]]; then
    source="$package_root/scripts/native-third-party.NOTICE"
    legal_digest=${package_inputs[1]%% *}
  fi
  staged="$legal_dir/.$entry-$expected.pending"
  if ! (set -o noclobber; : > "$staged") 2>/dev/null; then
    printf '%s\n' 'dope_install_legal_candidate_exists_preserve_prior_attempt' >&2
    exit 1
  fi
  cat -- "$source" > "$staged"
  if [[ "$legal_digest" != "$(sha256sum -- "$staged" | cut -d ' ' -f1)" ]]; then
    printf '%s\n' 'dope_install_legal_copied_checksum_mismatch' >&2
    exit 1
  fi
  mv -T --no-clobber -- "$staged" "$legal_dir/$entry"
  if [[ -e "$staged" ]]; then
    printf '%s\n' 'dope_install_legal_destination_appeared_preserve_candidate' >&2
    exit 1
  fi
done
# Fixed owned paths only. The uninstaller validates this complete set before
# removing any member; it never executes paths from an untrusted manifest.
(cd -- "$prefix" && sha256sum -- bin/dope share/dope/native.sha256 \
  share/dope/LICENSE share/dope/THIRD_PARTY_NOTICES) > "$legal_dir/installed.sha256"
"$prefix/bin/dope" doctor
