#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repository_root="$(dirname "$(git rev-parse --git-common-dir)")"
auditor="${JANKURAI_BIN:-$repository_root/.agent/tools/jankurai-1.7.1/jankurai-1.7.1-x86_64-unknown-linux-gnu/jankurai}"
if [[ ! -x "$auditor" ]]; then
  auditor="$(command -v jankurai)"
fi
test "$("$auditor" --version)" = "jankurai 1.7.1"
mkdir -p target/jankurai
base_ref="${JANKURAI_DIFF_BASE:-main}"
"$auditor" audit . --changed-fast --changed-from "$base_ref" --mode advisory \
  --json target/jankurai/audit-fast.json --md target/jankurai/audit-fast.md \
  --no-score-history --no-badge
