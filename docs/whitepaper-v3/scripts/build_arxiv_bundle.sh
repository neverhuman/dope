#!/usr/bin/env bash
# Assemble and optionally verify a self-contained arXiv upload bundle for DOPE v3.0.0.
#
# Usage:
#   bash docs/whitepaper-v3/scripts/build_arxiv_bundle.sh            # write arxiv/ and dope-v3-arxiv.zip
#   bash docs/whitepaper-v3/scripts/build_arxiv_bundle.sh --verify   # write + isolated pdflatex compile test
#   bash docs/whitepaper-v3/scripts/build_arxiv_bundle.sh --check    # verify committed bundle has zero drift
#   bash docs/whitepaper-v3/scripts/build_arxiv_bundle.sh --check --verify
set -euo pipefail

cd "$(dirname "$0")/../../.."
REPO_ROOT="$PWD"
WP="docs/whitepaper-v3"
EVIDENCE="bench/real-world/results/scoreboard/v3.0.0/evidence"
EPOCH="${SOURCE_DATE_EPOCH:-1795392000}"

CHECK_ONLY=0
VERIFY_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --verify) VERIFY_BUILD=1 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "build_arxiv_bundle: unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

for supplement in addendum-k addendum-h/part-b; do
  if ! (cd "$EVIDENCE/$supplement" && sha256sum --check --strict --status SHA256SUMS); then
    echo "build_arxiv_bundle: $supplement evidence does not match SHA256SUMS" >&2
    exit 1
  fi
done

for required in \
  "$WP/dope-v3.tex" \
  "$WP/dope-v3.bbl" \
  "$WP/generated/v3-macros.tex" \
  "$WP/budget/data.tex" \
  "$WP/examples/install.bash" \
  "$WP/examples/toy-run.bash" \
  "$WP/examples/own-csv.bash" \
  "$WP/examples/docker.bash"; do
  if [[ ! -s "$required" ]]; then
    echo "build_arxiv_bundle: missing or empty required input: $required" >&2
    exit 1
  fi
done

STAGE_PARENT=$(mktemp -d "${TMPDIR:-/tmp}/dope-arxiv-stage.XXXXXX")
trap 'rm -rf -- "$STAGE_PARENT"' EXIT
STAGE="$STAGE_PARENT/arxiv"
mkdir -p "$STAGE/generated" "$STAGE/budget" "$STAGE/data/addendum-k" "$STAGE/data/addendum-h" "$STAGE/examples"

cp "$WP/dope-v3.tex" "$STAGE/dope-v3.tex"
cp "$WP/dope-v3.bbl" "$STAGE/dope-v3.bbl"
cp "$WP/generated/v3-macros.tex" "$STAGE/generated/v3-macros.tex"
cp "$WP/examples/install.bash" "$STAGE/examples/install.bash"
cp "$WP/examples/toy-run.bash" "$STAGE/examples/toy-run.bash"
cp "$WP/examples/own-csv.bash" "$STAGE/examples/own-csv.bash"
cp "$WP/examples/docker.bash" "$STAGE/examples/docker.bash"

for csv in served.csv rows.csv k3-format1.csv k3-group.csv k33-summary.csv panel-report.csv k5-training-summary.csv; do
  cp "$EVIDENCE/addendum-k/$csv" "$STAGE/data/addendum-k/$csv"
done
cp "$EVIDENCE/addendum-h/part-b/summary.csv" "$STAGE/data/addendum-h/summary.csv"

# Rewrite parent-directory paths in budget/data.tex to self-contained local data/ paths for arXiv.
sed \
  -e 's|../../bench/real-world/results/scoreboard/v3.0.0/evidence/addendum-k|data/addendum-k|g' \
  -e 's|../../bench/real-world/results/scoreboard/v3.0.0/evidence/addendum-h/part-b/summary.csv|data/addendum-h/summary.csv|g' \
  "$WP/budget/data.tex" > "$STAGE/budget/data.tex"

if grep -rn '\.\./' "$STAGE/budget/data.tex" >/dev/null 2>&1; then
  echo "build_arxiv_bundle: parent-directory path remained in staged budget/data.tex" >&2
  exit 1
fi

# Normalize permissions and timestamps for byte-deterministic zip archives.
find "$STAGE" -type d -exec chmod 0755 {} +
find "$STAGE" -type f -exec chmod 0644 {} +
find "$STAGE" -exec touch -h -d "@$EPOCH" {} +

TMP_ZIP="$STAGE_PARENT/dope-v3-arxiv.zip"
(
  cd "$STAGE"
  export TZ=UTC
  LC_ALL=C find . -type f | sed 's|^\./||' | sort | zip -X -q -9 -@ "$TMP_ZIP"
)

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  if [[ ! -d "$WP/arxiv" || ! -f "$WP/dope-v3-arxiv.zip" ]]; then
    echo "build_arxiv_bundle: committed $WP/arxiv or $WP/dope-v3-arxiv.zip missing; run just arxiv-v3" >&2
    exit 1
  fi
  if ! diff -ruN "$STAGE" "$WP/arxiv"; then
    echo "build_arxiv_bundle: drift detected in $WP/arxiv; run just arxiv-v3" >&2
    exit 1
  fi
  # Verify zip contains the exact same file tree and byte-identical members.
  UNZIP_CHECK="$STAGE_PARENT/unzip-check"
  mkdir -p "$UNZIP_CHECK"
  unzip -q "$WP/dope-v3-arxiv.zip" -d "$UNZIP_CHECK"
  if ! diff -ruN "$STAGE" "$UNZIP_CHECK"; then
    echo "build_arxiv_bundle: drift detected in $WP/dope-v3-arxiv.zip; run just arxiv-v3" >&2
    exit 1
  fi
  echo "OK: $WP/arxiv and $WP/dope-v3-arxiv.zip match committed sources ($(wc -c < "$WP/dope-v3-arxiv.zip") bytes)"
else
  rm -rf "$WP/arxiv"
  mkdir -p "$WP/arxiv"
  cp -a "$STAGE/." "$WP/arxiv/"
  cp "$TMP_ZIP" "$WP/dope-v3-arxiv.zip"
  echo "Built $WP/arxiv and $WP/dope-v3-arxiv.zip ($(wc -c < "$WP/dope-v3-arxiv.zip") bytes)"
fi

if [[ "$VERIFY_BUILD" -eq 1 ]]; then
  ISOLATED="$STAGE_PARENT/isolated-compile"
  mkdir -p "$ISOLATED"
  unzip -q "$WP/dope-v3-arxiv.zip" -d "$ISOLATED"
  # Confirm no prebuilt PDF or build script slipped into the upload zip.
  if find "$ISOLATED" -name '*.pdf' -o -name '*.py' -o -name '*.sh' | grep -q .; then
    echo "build_arxiv_bundle: unexpected PDF or script found in arXiv zip" >&2
    exit 1
  fi
  (
    cd "$ISOLATED"
    export SOURCE_DATE_EPOCH="$EPOCH"
    for pass in 1 2 3; do
      if ! pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape \
        '\pdftrailerid{}\pdfsuppressptexinfo=-1\input{dope-v3.tex}' >"pass-${pass}.log" 2>&1; then
        echo "build_arxiv_bundle: isolated pdflatex failed on pass $pass" >&2
        tail -40 "pass-${pass}.log" >&2
        exit 1
      fi
    done
    if grep -E "LaTeX Warning: (Reference|Citation) .* undefined|There were undefined references|Citation .* undefined" \
      dope-v3.log >&2; then
      echo "build_arxiv_bundle: unresolved references or citations in isolated arXiv build" >&2
      exit 1
    fi
  )
  iso_pages=$(pdfinfo "$ISOLATED/dope-v3.pdf" | awk '/^Pages:/ {print $2}')
  ref_pages=$(pdfinfo "$REPO_ROOT/$WP/dope-v3.pdf" | awk '/^Pages:/ {print $2}')
  if [[ "$iso_pages" != "$ref_pages" ]]; then
    echo "build_arxiv_bundle: page count mismatch (arXiv build=$iso_pages vs committed PDF=$ref_pages)" >&2
    exit 1
  fi
  echo "OK: isolated arXiv compile succeeded ($iso_pages pages, zero undefined refs/citations)"
fi
