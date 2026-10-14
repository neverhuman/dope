#!/usr/bin/env bash
# Install a caller-selected, checksum-pinned already-built native binary.
set -euo pipefail
if [[ $# != 3 && $# != 4 ]]; then
  printf '%s\n' 'usage: install-native.sh BINARY SHA256 PREFIX [--dry-run]' >&2
  exit 2
fi
binary=$1
expected=$2
prefix=$3
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
legal_dir="$prefix/share/dope-kernel"
if [[ $# == 4 && "$4" != --dry-run ]]; then
  printf '%s\n' 'native_install_option_invalid' >&2
  exit 2
fi
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
for parent in "$prefix" "$prefix/bin" "$prefix/share" "$legal_dir"; do
  if [[ -L "$parent" ]]; then
    printf '%s\n' 'native_install_destination_symlink_refused' >&2
    exit 1
  fi
done
for destination in "$prefix/bin/dope-kernel" "$legal_dir/native.sha256" \
  "$legal_dir/LICENSE" "$legal_dir/THIRD_PARTY_NOTICES" "$legal_dir/installed.sha256"; do
  if [[ -e "$destination" || -L "$destination" ]]; then
    printf '%s\n' 'native_install_destination_exists_use_new_prefix' >&2
    exit 1
  fi
done
# These frozen inputs come from the locked offline package graph, not ambient
# adapter notices. Validate them before creating any destination or candidate.
for input in LICENSE scripts/native-third-party.NOTICE scripts/native-legal.sha256 Cargo.lock rust-toolchain.toml; do
  if [[ ! -f "$package_root/$input" || -L "$package_root/$input" ]]; then
    printf '%s\n' 'native_install_legal_inputs_unavailable' >&2
    exit 1
  fi
done
if ! (cd -- "$package_root" && sha256sum --check --status -- scripts/native-legal.sha256); then
  printf '%s\n' 'native_install_legal_inputs_checksum_mismatch' >&2
  exit 1
fi
mapfile -t package_inputs < "$package_root/scripts/native-legal.sha256"
if [[ ${#package_inputs[@]} != 4 || ! "${package_inputs[2]}" =~ ^([0-9a-f]{64})\ \ Cargo\.lock$ ]]; then
  printf '%s\n' 'native_install_legal_manifest_invalid' >&2
  exit 1
fi
lock_digest=${BASH_REMATCH[1]}
if [[ ! "${package_inputs[3]}" =~ ^([0-9a-f]{64})\ \ rust-toolchain\.toml$ ]]; then
  printf '%s\n' 'native_install_legal_manifest_invalid' >&2
  exit 1
fi
toolchain_digest=${BASH_REMATCH[1]}
# Check product identity before writing any destination. No package manager,
# Python environment, optional dependency installation, or network is used.
identity=$("$binary" doctor)
if [[ "$identity" != *'"implementation":"native_rust"'* || "$identity" != *'"name":"dope-kernel"'* || "$identity" != *"\"executable_sha256\":\"$expected\""* ]]; then
  printf '%s\n' 'native_install_identity_mismatch' >&2
  exit 1
fi
if [[ "$identity" != *"\"build_cargo_lock_sha256\":\"$lock_digest\""* || \
      "$identity" != *"\"declared_rust_toolchain_sha256\":\"$toolchain_digest\""* || \
      "$identity" != *'"gpu_training":false'* ]]; then
  printf '%s\n' 'native_install_package_profile_or_build_inputs_mismatch' >&2
  exit 1
fi
if [[ $# == 4 ]]; then
  printf '%s\n' '{"operation":"would_install","implementation":"native_rust","network":false}'
  exit 0
fi
mkdir -p -- "$prefix/bin" "$prefix/share/dope-kernel"
candidate="$prefix/bin/.dope-kernel-$expected.pending"
if ! (set -o noclobber; : > "$candidate") 2>/dev/null; then
  printf '%s\n' 'native_install_candidate_exists_preserve_prior_attempt' >&2
  exit 1
fi
cat -- "$binary" > "$candidate"
chmod 0755 -- "$candidate"
installed=$(sha256sum -- "$candidate")
if [[ "${installed%% *}" != "$expected" ]]; then
  printf '%s\n' 'native_install_copied_checksum_mismatch' >&2
  exit 1
fi
mv -T --no-clobber -- "$candidate" "$prefix/bin/dope-kernel"
if [[ -e "$candidate" ]]; then
  printf '%s\n' 'native_install_destination_appeared_preserve_candidate' >&2
  exit 1
fi
printf '%s\n' "$expected" > "$prefix/share/dope-kernel/native.sha256"
for entry in LICENSE THIRD_PARTY_NOTICES; do
  source="$package_root/LICENSE"
  if [[ "$entry" == THIRD_PARTY_NOTICES ]]; then
    source="$package_root/scripts/native-third-party.NOTICE"
  fi
  staged="$legal_dir/.$entry-$expected.pending"
  if ! (set -o noclobber; : > "$staged") 2>/dev/null; then
    printf '%s\n' 'native_install_legal_candidate_exists_preserve_prior_attempt' >&2
    exit 1
  fi
  cat -- "$source" > "$staged"
  if [[ "$(sha256sum -- "$source" | cut -d ' ' -f1)" != "$(sha256sum -- "$staged" | cut -d ' ' -f1)" ]]; then
    printf '%s\n' 'native_install_legal_copied_checksum_mismatch' >&2
    exit 1
  fi
  mv -T --no-clobber -- "$staged" "$legal_dir/$entry"
  if [[ -e "$staged" ]]; then
    printf '%s\n' 'native_install_legal_destination_appeared_preserve_candidate' >&2
    exit 1
  fi
done
# Fixed owned paths only. The uninstaller validates this complete set before
# removing any member; it never executes paths from an untrusted manifest.
(cd -- "$prefix" && sha256sum -- bin/dope-kernel share/dope-kernel/native.sha256 \
  share/dope-kernel/LICENSE share/dope-kernel/THIRD_PARTY_NOTICES) > "$legal_dir/installed.sha256"
"$prefix/bin/dope-kernel" doctor
