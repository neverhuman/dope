#!/usr/bin/env bash
# Build docs/whitepaper-v3/dope-v3.pdf from the committed sources and generated files.
# The build runs in target/paper-v3 with fixed timestamps and no trailer ID, so the same
# TeX Live gives the same bytes; the PDF and the .bbl are then copied next to the source.
set -euo pipefail
cd "$(dirname "$0")/../../.."
out="$PWD/target/paper-v3"
mkdir -p "$out"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1795392000}"
export BIBINPUTS="$PWD/docs/whitepaper-v3:${BIBINPUTS:-}"
rm -f "$out"/dope-v3.{aux,bbl,blg,fdb_latexmk,fls,log,out,pdf}
# A committed .bbl beside the source would be read instead of the fresh one, and latexmk
# would then never run bibtex; it is a build output and is copied back below.
rm -f docs/whitepaper-v3/dope-v3.bbl
if ! (cd docs/whitepaper-v3 && latexmk -g -pdf -usepretex='\pdftrailerid{}\pdfsuppressptexinfo=-1' \
  -interaction=nonstopmode -halt-on-error -file-line-error -outdir="$out" dope-v3) \
  >"$out/build.log" 2>&1; then
  tail -40 "$out/build.log" >&2
  exit 1
fi
if grep -E "LaTeX Warning: (Reference|Citation) .* undefined|There were undefined references|Citation .* undefined" \
  "$out/dope-v3.log" "$out/dope-v3.blg" >&2; then
  echo 'dope-v3: unresolved references or citations' >&2
  exit 1
fi
cp "$out/dope-v3.pdf" docs/whitepaper-v3/dope-v3.pdf
cp "$out/dope-v3.bbl" docs/whitepaper-v3/dope-v3.bbl
echo "built docs/whitepaper-v3/dope-v3.pdf ($(pdfinfo docs/whitepaper-v3/dope-v3.pdf | awk '/^Pages:/ {print $2}') pages)"
