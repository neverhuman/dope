#!/bin/bash
# Rebuild the journal PDF and the supplement from the tex already in this directory.
# Auxiliaries are removed after a clean run. The PDFs stay.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1760000000}"
build_one() {
  local stem="$1"
  # latexmk reruns pdflatex and bibtex until the citation and reference labels settle.
  # A document with no bibliography does not invoke bibtex.
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "$stem"
  cp "$stem".log /tmp/"$stem"-3.log
  rm -f "$stem".aux "$stem".bbl "$stem".blg "$stem".log "$stem".out "$stem".toc "$stem".fls "$stem".fdb_latexmk
}
build_one dope-mfs
build_one supplement
pdfinfo dope-mfs.pdf | awk '/Pages|Page size/'
pdfinfo supplement.pdf | awk '/Pages|Page size/'
