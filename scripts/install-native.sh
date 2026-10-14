#!/usr/bin/env bash
# Install a caller-selected, checksum-pinned already-built native binary.
set -euo pipefail
if [[ $# != 3 ]]; then
  printf '%s\n' 'usage: install-native.sh BINARY SHA256 PREFIX' >&2
  exit 2
fi
binary=$1
expected=$2
prefix=$3
if [[ ! "$expected" =~ ^[0-9a-f]{64}$ || ! -f "$binary" || -L "$binary" || ! -x "$binary" ]]; then
  printf '%s\n' 'native_install_input_invalid' >&2
  exit 1
fi
case "$(uname -s):$(uname -m)" in
  Linux:x86_64) ;;
  *) printf '%s\n' 'native_install_platform_unqualified' >&2; exit 1 ;;
esac
actual=$(sha256sum -- "$binary")
actual=${actual%% *}
if [[ "$actual" != "$expected" ]]; then
  printf '%s\n' 'native_install_checksum_mismatch' >&2
  exit 1
fi
if [[ -e "$prefix/bin/dope-kernel" || -L "$prefix/bin/dope-kernel" ]]; then
  printf '%s\n' 'native_install_destination_exists_use_new_prefix' >&2
  exit 1
fi
# Check product identity before writing any destination. No package manager,
# Python environment, optional dependency installation, or network is used.
identity=$("$binary" doctor)
if [[ "$identity" != *'"implementation":"native_rust"'* || "$identity" != *'"name":"dope-kernel"'* || "$identity" != *"\"executable_sha256\":\"$expected\""* ]]; then
  printf '%s\n' 'native_install_identity_mismatch' >&2
  exit 1
fi
mkdir -p -- "$prefix/bin" "$prefix/share/dope-kernel"
install -m 0755 -- "$binary" "$prefix/bin/dope-kernel"
installed=$(sha256sum -- "$prefix/bin/dope-kernel")
if [[ "${installed%% *}" != "$expected" ]]; then
  printf '%s\n' 'native_install_copied_checksum_mismatch' >&2
  exit 1
fi
printf '%s\n' "$expected" > "$prefix/share/dope-kernel/native.sha256"
"$prefix/bin/dope-kernel" doctor
