#!/usr/bin/env bash
# Build and exercise the supported Linux x86_64 native installer in target/.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$root"
case "$(uname -s):$(uname -m)" in
  Linux:x86_64) ;;
  *) printf '%s\n' 'native_release_smoke_platform_unqualified' >&2; exit 1 ;;
esac
command -v strace >/dev/null || { printf '%s\n' 'native_release_smoke_strace_required' >&2; exit 1; }
sbom=target/security/sbom.spdx.json
[[ -s "$sbom" ]] || { printf '%s\n' 'native_release_smoke_sbom_required_run_just_security' >&2; exit 1; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run="target/native-release-smoke/$stamp-$$"
mkdir -p -- "$run/package" "$run/install-prefix/bin"
binary=target/release/dope-kernel
# Ignore ambient machine-specific flags so this exercises the generic portable build.
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" cargo build --locked --release --offline -p dope-kernel --bin dope-kernel
install -m 0755 -- "$binary" "$run/package/dope-kernel"
first=$(sha256sum -- "$run/package/dope-kernel" | cut -d ' ' -f1)
# A second locked offline invocation records the same-worktree rebuild result.
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" cargo build --locked --release --offline -p dope-kernel --bin dope-kernel
install -m 0755 -- "$binary" "$run/package/dope-kernel-repeat"
second=$(sha256sum -- "$run/package/dope-kernel-repeat" | cut -d ' ' -f1)
[[ "$first" == "$second" ]] || { printf '%s\n' 'native_release_smoke_binary_not_repeatable' >&2; exit 1; }

cp -- Cargo.lock rust-toolchain.toml LICENSE scripts/native-third-party.NOTICE "$run/package/"
cp -- "$sbom" "$run/package/sbom.spdx.json"
(
  cd -- "$run/package"
  sha256sum -- dope-kernel Cargo.lock rust-toolchain.toml LICENSE native-third-party.NOTICE sbom.spdx.json > SHA256SUMS
  sha256sum --check --status -- SHA256SUMS
)

prefix="$run/install-prefix"
printf '%s\n' 'preserve-me' > "$prefix/unrelated-evidence.txt"
expected=$(sha256sum -- "$run/package/dope-kernel")
expected=${expected%% *}
bash scripts/install-native.sh "$run/package/dope-kernel" "$expected" "$prefix" --dry-run > "$run/install-dry-run.json"
bash scripts/install-native.sh "$run/package/dope-kernel" "$expected" "$prefix" > "$run/install-result.json"
[[ -x "$prefix/bin/dope-kernel" ]]
[[ "$("$prefix/bin/dope-kernel" doctor)" == *'"implementation":"native_rust"'* ]]

# Capture the installed doctor's network-related syscalls; it must make no connection.
strace -f -e trace=socket,connect,bind,listen,sendto,sendmsg -o "$run/doctor-network.trace" \
  "$prefix/bin/dope-kernel" doctor > "$run/doctor.json"
if rg -n 'socket\(|connect\(|bind\(|listen\(|sendto\(|sendmsg\(' "$run/doctor-network.trace"; then
  printf '%s\n' 'native_release_smoke_doctor_used_network_syscall' >&2
  exit 1
fi
bash scripts/uninstall-native.sh "$prefix" --dry-run > "$run/uninstall-dry-run.json"
bash scripts/uninstall-native.sh "$prefix" > "$run/uninstall-result.json"
[[ ! -e "$prefix/bin/dope-kernel" && "$(cat -- "$prefix/unrelated-evidence.txt")" == preserve-me ]]

rustc_version=$(rustc --version)
cargo_version=$(cargo --version)
platform="$(uname -s) $(uname -m)"
python3 - "$run" "$platform" "$rustc_version" "$cargo_version" "$first" "$second" <<'PY'
import hashlib
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

files = ["package/dope-kernel", "package/dope-kernel-repeat", "package/Cargo.lock",
         "package/rust-toolchain.toml", "package/LICENSE", "package/native-third-party.NOTICE",
         "package/sbom.spdx.json", "package/SHA256SUMS", "install-dry-run.json",
         "install-result.json", "doctor.json", "doctor-network.trace",
         "uninstall-dry-run.json", "uninstall-result.json"]
receipt = {
    "schema": "dope-native-release-smoke-v1",
    "platform_tested": sys.argv[2],
    "rustc": sys.argv[3],
    "cargo": sys.argv[4],
    "build": {"command": "cargo build --locked --release --offline -p dope-kernel --bin dope-kernel",
              "first_sha256": sys.argv[5], "repeat_sha256": sys.argv[6],
              "same_worktree_repeat_only": True,
              "independent_clean_machine_reproducibility": "not_established"},
    "installer": {"clean_target_prefix": True, "offline": True,
                  "doctor_network_syscalls": 0, "uninstall_preserved_unrelated_file": True},
    "release": {"signed": False,
                "signing_status": "blocked_no_authorized_signing_identity_established",
                "supported_platform_claim": "Linux x86_64 smoke only"},
    "bundled_assets": ["native binary", "Cargo.lock", "rust-toolchain.toml", "LICENSE",
                       "native-third-party.NOTICE", "SPDX SBOM", "SHA256SUMS"],
    "files": {name: digest(run / name) for name in files},
}
(run / "build-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
printf 'native_release_smoke_ok %s\n' "$run"

# The facade is a separate product identity; retain the kernel qualification above.
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}" \
  cargo build --locked --release --offline -p dope-kernel --bin dope
facade="$root/$run/facade-package"
mkdir -p -- "$facade/scripts" "$root/$run/extracted-facade"
cp -- target/release/dope "$facade/dope"
cp -- LICENSE Cargo.lock rust-toolchain.toml "$facade/"
cp -- scripts/install-dope.sh scripts/uninstall-dope.sh scripts/native-legal.sha256 \
  scripts/native-third-party.NOTICE "$facade/scripts/"
cp -- "$sbom" "$facade/sbom.spdx.json"
(cd -- "$facade" && sha256sum -- dope LICENSE Cargo.lock rust-toolchain.toml \
 scripts/install-dope.sh scripts/uninstall-dope.sh scripts/native-legal.sha256 \
 scripts/native-third-party.NOTICE sbom.spdx.json > SHA256SUMS)
tar -czf "$run/facade-package.tar.gz" -C "$facade" .
tar -xzf "$run/facade-package.tar.gz" -C "$run/extracted-facade"
(cd -- "$run/extracted-facade" && sha256sum --check --status -- SHA256SUMS)
DOPE_INSTALL_TEST_BINARY="$root/$run/extracted-facade/dope" \
 DOPE_INSTALL_TEST_RECEIPT="$root/$run/facade-install-receipt.json" \
 cargo test --locked --test vnext_dope_install \
 installer::installed_facade_bundle_runs_100_rows_and_preserves_unrelated_files -- --exact --nocapture
[[ -s "$run/facade-install-receipt.json" ]] || { printf '%s\n' 'facade_install_smoke_receipt_absent' >&2; exit 1; }
printf 'facade_install_smoke_ok %s\n' "$run/facade-install-receipt.json"
