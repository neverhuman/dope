#!/usr/bin/env bash
# A disposable CI checkout must regenerate the complete paper from committed inputs.
set -euo pipefail
cd "$(dirname "$0")/../.."
output_paths=(
  docs/whitepaper/generated docs/whitepaper/figures
  docs/whitepaper/dope-mfs.pdf docs/whitepaper/dope-mfs-anonymous.pdf
  docs/whitepaper/supplement.pdf docs/whitepaper/supplement-anonymous.pdf
)
if [[ -n "$(git status --porcelain -- "${output_paths[@]}")" ]]; then
  echo 'paper clean rebuild requires unchanged generated outputs' >&2
  exit 1
fi
mkdir -p target/paper-clean
# Preserve deletion evidence; only derived paper files are removed.
git status --short > target/paper-clean/status-before.txt
git stash list > target/paper-clean/stashes-before.txt
git log --all --not --remotes --oneline > target/paper-clean/local-commits-before.txt
git ls-remote --tags origin > target/paper-clean/remote-tags-before.txt
git for-each-ref refs/tags --format='%(refname) %(objectname)' > target/paper-clean/local-tags-before.txt
if [[ -s target/paper-clean/local-commits-before.txt ]]; then
  # Local verification can include unpublished commits; preserve all refs first.
  nice -n 10 ionice -c 3 git -c pack.threads=1 bundle create target/paper-clean/before-clean.bundle --all
  git bundle verify target/paper-clean/before-clean.bundle
  git bundle list-heads target/paper-clean/before-clean.bundle > target/paper-clean/bundle-heads.txt
  head_commit=$(git rev-parse HEAD)
  if ! rg -q "^${head_commit} " target/paper-clean/bundle-heads.txt; then
    echo 'current commit is absent from the backup bundle' >&2
    exit 1
  fi
fi
find docs/whitepaper/generated -type f -print > target/paper-clean/generated-before.txt
rm -rf docs/whitepaper/generated
nice -n 10 ionice -c 3 just paper
# Include absent/untracked files as well as tracked drift in the clean-rebuild gate.
git diff --exit-code
if [[ -n "$(git status --porcelain -- "${output_paths[@]}")" ]]; then
  echo 'clean paper rebuild produced untracked or changed outputs' >&2
  exit 1
fi
