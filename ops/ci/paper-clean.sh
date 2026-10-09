#!/usr/bin/env bash
# A disposable CI checkout must regenerate the complete paper from committed inputs.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -n "$(git status --porcelain -- docs/whitepaper/generated)" ]]; then
  echo 'paper clean rebuild requires unchanged generated inputs' >&2
  exit 1
fi
mkdir -p target/paper-clean
# Preserve deletion evidence; only derived paper files are removed.
git status --short > target/paper-clean/status-before.txt
git stash list > target/paper-clean/stashes-before.txt
git log --all --not --remotes --oneline > target/paper-clean/local-commits-before.txt
find docs/whitepaper/generated -type f -print > target/paper-clean/generated-before.txt
rm -rf docs/whitepaper/generated
nice -n 10 ionice -c 3 just paper
# Include absent/untracked files as well as tracked drift in the clean-rebuild gate.
git diff --exit-code
if [[ -n "$(git status --porcelain -- docs/whitepaper/generated docs/whitepaper/figures)" ]]; then
  echo 'clean paper rebuild produced untracked or changed outputs' >&2
  exit 1
fi
