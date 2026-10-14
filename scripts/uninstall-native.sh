#!/usr/bin/env bash
# Remove only the native binary and receipt installed into a caller's prefix.
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
if [[ $# == 2 ]]; then
  printf '%s\n' '{"operation":"would_uninstall","implementation":"native_rust"}'
  exit 0
fi
rm -- "$binary" "$receipt"
# Preserve all directories, unrelated files, V1 environments and old evidence.
printf '%s\n' '{"operation":"uninstalled","implementation":"native_rust"}'
