#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repository_root="$(dirname "$(git rev-parse --git-common-dir)")"
auditor="${JANKURAI_BIN:-$repository_root/.agent/tools/jankurai-1.7.1/jankurai-1.7.1-x86_64-unknown-linux-gnu/jankurai}"
if [[ ! -x "$auditor" ]]; then
  auditor="$(command -v jankurai)"
fi
if [[ "$("$auditor" --version)" != "jankurai 1.7.1" ]]; then
  echo "Jankurai v1.7.1 is required" >&2
  exit 1
fi
mkdir -p target/jankurai
set +e
"$auditor" audit . --full --json target/jankurai/repo-score.json --md target/jankurai/repo-score.md --no-score-history --no-badge
audit_status=$?
set -e
bash agent/emit-audit-receipts.sh
exit "$audit_status"
