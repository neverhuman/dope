#!/usr/bin/env bash
# A disposable CI checkout must regenerate the complete paper from committed inputs.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == --v3-only ]]; then
  just paper-v3
  git diff --exit-code -- docs/whitepaper-v3
  exit
fi
output_paths=(
  docs/whitepaper/generated docs/whitepaper/figures
  docs/whitepaper/dope-mfs.pdf docs/whitepaper/dope-mfs-anonymous.pdf
  docs/whitepaper/supplement.pdf docs/whitepaper/supplement-anonymous.pdf
  docs/whitepaper-v3/generated docs/whitepaper-v3/figures
  docs/whitepaper-v3/dope-v3.pdf docs/whitepaper-v3/dope-v3.bbl
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
rm -f docs/whitepaper/figures/architecture.pdf
nice -n 10 ionice -c 3 just paper
nice -n 10 ionice -c 3 bash paper/build.sh
# The v3 paper: its macros are rebuilt from the receipts. The hero PDF and the two tables that
# the Rust `dope_kpi render-paper-tables` writes stay (a cargo test pins the tables to the
# receipts); render_v3_paper.py --check verifies their recorded SHA-256s.
rm -f docs/whitepaper-v3/generated/v3-macros.tex docs/whitepaper-v3/dope-v3.pdf docs/whitepaper-v3/dope-v3.bbl
nice -n 10 ionice -c 3 just paper-v3
nice -n 10 ionice -c 3 python3 -B -m unittest discover -s docs/whitepaper-v3/scripts -p 'test_*.py'
# Include absent/untracked files as well as tracked drift in the clean-rebuild gate.
git diff --exit-code
if [[ -n "$(git status --porcelain -- "${output_paths[@]}")" ]]; then
  echo 'clean paper rebuild produced untracked or changed outputs' >&2
  exit 1
fi
