#!/usr/bin/env bash
# Run the README quick start against published dope-engine-v3.0.0.
# Tokens are unset so attestation has to use the release bundle.
set -euo pipefail
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN
cd "$(dirname "$0")/../.."
root=$PWD
home_dir="$root/target/install-smoke/home"
work="$root/target/install-smoke/work"
rm -rf "$root/target/install-smoke"
mkdir -p "$home_dir" "$work"

command -v curl >/dev/null 2>&1 || { echo "install-smoke: curl is required" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "install-smoke: jq is required" >&2; exit 1; }

block=$(awk '
  /^## Quick start$/ { grab = 1; next }
  grab && /^## / { exit }
  grab && /^```bash$/ { inblock = 1; next }
  grab && inblock && /^```$/ { exit }
  grab && inblock { print }
' README.md)
[[ -n "$block" ]] || { echo "install-smoke: README quick start has no bash block" >&2; exit 1; }
[[ "$block" == *"--require-attestation"* && "$block" == *"--no-weights"* ]] || {
  echo "install-smoke: quick start is missing the signed-engine flags" >&2
  exit 1
}

export HOME="$home_dir"
export PATH="/usr/bin:/bin"
cd "$work"
bash -euo pipefail -c "$block"

export PATH="$HOME/.local/bin:$PATH"
version=$(dope --version)
[[ "$version" == "dope 3.0.0" ]]
sha=$(dope doctor --json | jq -r .executable_sha256)
[[ "$sha" == "8723955b1fcd59cab8996842a39b3a579c596c957ac1f89f291b0a9d1ebfaea0" ]]
[[ -s synthetic.csv ]]
bytes=$(wc -c < model.dope | tr -d ' ')
[[ "$bytes" -gt 0 && "$bytes" -le 2048 ]]
echo "install-smoke: dope 3.0.0 $sha model ${bytes}B"
