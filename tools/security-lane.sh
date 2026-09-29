#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ops/ci/lib.sh
mkdir -p target/security
if ! command -v cargo-audit >/dev/null 2>&1; then
  cargo install cargo-audit --version "$CARGO_AUDIT_VERSION" --locked
fi
cargo audit --json > target/security/cargo-audit.json
ci_require_artifact target/security/cargo-audit.json
printf '%s\n' 'jankurai-security-step={"label":"cargo-audit","tool":"cargo-audit","shell_command":"cargo audit --json","status":"ran","advisory":false,"exit_code":0}'
if ! command -v actionlint >/dev/null 2>&1; then
  mkdir -p target/tools/actionlint
  archive="target/tools/actionlint/actionlint_${ACTIONLINT_VERSION}_linux_amd64.tar.gz"
  curl --fail --location --silent --show-error --retry 3 \
    "https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/actionlint_${ACTIONLINT_VERSION}_linux_amd64.tar.gz" \
    --output "$archive"
  echo "be92c2652ab7b6d08425428797ceabeb16e31a781c07bc388456b4e592f3e36a  $archive" | sha256sum --check --status
  tar -xzf "$archive" -C target/tools/actionlint actionlint
  export PATH="$PWD/target/tools/actionlint:$PATH"
  if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$PWD/target/tools/actionlint" >> "$GITHUB_PATH"
  fi
fi
actionlint .github/workflows/*.yml
printf '%s\n' 'jankurai-security-step={"label":"actionlint","tool":"actionlint","shell_command":"actionlint .github/workflows/*.yml","status":"ran","advisory":false,"exit_code":0}'
if ! command -v gitleaks >/dev/null 2>&1; then
  mkdir -p target/tools/gitleaks
  archive="target/tools/gitleaks/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
  curl --fail --location --silent --show-error --retry 3 \
    "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" \
    --output "$archive"
  echo "5bc41815076e6ed6ef8fbecc9d9b75bcae31f39029ceb55da08086315316e3ba  $archive" | sha256sum --check --status
  tar -xzf "$archive" -C target/tools/gitleaks gitleaks
  export PATH="$PWD/target/tools/gitleaks:$PATH"
  if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$PWD/target/tools/gitleaks" >> "$GITHUB_PATH"
  fi
fi
gitleaks detect --source . --no-banner --redact --report-format json --report-path target/security/gitleaks.json
printf '%s\n' 'jankurai-security-step={"label":"gitleaks","tool":"gitleaks","shell_command":"gitleaks detect --source . --no-banner --redact","status":"ran","advisory":false,"exit_code":0}'
if command -v syft >/dev/null 2>&1; then
  mkdir -p target/jankurai/security
  syft dir:. --exclude './target/**' --exclude './.agent/**' -o spdx-json=target/security/sbom.spdx.json
  ci_require_artifact target/security/sbom.spdx.json
  syft dir:. --exclude './target/**' --exclude './.agent/**' -o cyclonedx-json=target/jankurai/security/sbom.json
  ci_require_artifact target/jankurai/security/sbom.json
  printf '%s\n' 'jankurai-security-step={"label":"syft","tool":"syft","shell_command":"syft dir:. -o spdx-json","status":"ran","advisory":false,"exit_code":0}'
  python3 - <<'PY'
import json
from pathlib import Path

document = json.loads(Path("target/security/sbom.spdx.json").read_text())
assert document["spdxVersion"] == "SPDX-2.3"
assert document["SPDXID"] == "SPDXRef-DOCUMENT"
assert isinstance(document["packages"], list) and document["packages"]
assert isinstance(document["relationships"], list)
PY
  printf '%s\n' 'jankurai-security-step={"label":"sbom-validation","tool":"sbom-validation","shell_command":"python3 validate SPDX-2.3 SBOM","status":"ran","advisory":false,"exit_code":0}'
fi
