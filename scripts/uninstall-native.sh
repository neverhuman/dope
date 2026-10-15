#!/usr/bin/env bash
# Remove only checksum-bound native install members from a caller's prefix.
set -euo pipefail
if [[ $# != 1 && $# != 2 ]]; then
  printf '%s\n' 'usage: uninstall-native.sh PREFIX [--dry-run]' >&2
  exit 2
fi
prefix=$1
if [[ $# == 2 && "$2" != --dry-run ]]; then
  printf '%s\n' 'native_uninstall_option_invalid' >&2
  exit 2
fi
binary="$prefix/bin/dope-kernel"
receipt="$prefix/share/dope-kernel/native.sha256"
legal_dir="$prefix/share/dope-kernel"
manifest="$legal_dir/installed.sha256"
for parent in "$prefix" "$prefix/bin" "$prefix/share" "$legal_dir"; do
  if [[ -L "$parent" ]]; then
    printf '%s\n' 'native_uninstall_destination_symlink_refused' >&2
    exit 1
  fi
done
if [[ ! -f "$binary" || -L "$binary" || ! -x "$binary" || ! -f "$receipt" || -L "$receipt" ]]; then
  printf '%s\n' 'native_uninstall_installed_receipt_required' >&2
  exit 1
fi
mapfile -t recorded < "$receipt"
if [[ ${#recorded[@]} != 1 || ! "${recorded[0]}" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'native_uninstall_receipt_invalid' >&2
  exit 1
fi
expected=${recorded[0]}
actual=$(sha256sum -- "$binary")
if [[ "${actual%% *}" != "$expected" ]]; then
  printf '%s\n' 'native_uninstall_modified_binary_preserved' >&2
  exit 1
fi
identity=$("$binary" doctor)
if [[ "$identity" != *'"implementation":"native_rust"'* || "$identity" != *"\"executable_sha256\":\"$expected\""* ]]; then
  printf '%s\n' 'native_uninstall_identity_mismatch' >&2
  exit 1
fi
owned=("$binary" "$receipt")
if [[ -e "$manifest" || -L "$manifest" || -e "$legal_dir/LICENSE" || -L "$legal_dir/LICENSE" || \
      -e "$legal_dir/THIRD_PARTY_NOTICES" || -L "$legal_dir/THIRD_PARTY_NOTICES" ]]; then
  for file in "$manifest" "$legal_dir/LICENSE" "$legal_dir/THIRD_PARTY_NOTICES"; do
    if [[ ! -f "$file" || -L "$file" ]]; then
      printf '%s\n' 'native_uninstall_complete_legal_manifest_required' >&2
      exit 1
    fi
  done
  current=$(cd -- "$prefix" && sha256sum -- bin/dope-kernel share/dope-kernel/native.sha256 \
    share/dope-kernel/LICENSE share/dope-kernel/THIRD_PARTY_NOTICES)
  if [[ "$current" != "$(cat -- "$manifest")" ]]; then
    printf '%s\n' 'native_uninstall_modified_package_preserved' >&2
    exit 1
  fi
  owned+=("$legal_dir/LICENSE" "$legal_dir/THIRD_PARTY_NOTICES" "$manifest")
fi
if [[ $# == 2 ]]; then
  printf '%s\n' '{"operation":"would_uninstall","implementation":"native_rust"}'
  exit 0
fi
rm -- "${owned[@]}"
# Preserve all directories, unrelated files, V1 environments and old evidence.
printf '%s\n' '{"operation":"uninstalled","implementation":"native_rust"}'
