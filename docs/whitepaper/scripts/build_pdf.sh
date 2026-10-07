#!/bin/bash
# Rebuild the journal PDF and the supplement from the tex already in this directory.
# Auxiliaries and logs stay in the checkout target directory; PDFs are copied here.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1760000000}"
build_root="$(realpath ../..)/target/paper-build"
mkdir -p "$build_root"
build_one() {
  local stem="$1"
  # latexmk reruns pdflatex and bibtex until the citation and reference labels settle.
  # A document with no bibliography does not invoke bibtex.
  latexmk -pdf -outdir="$build_root" -interaction=nonstopmode -halt-on-error -file-line-error "$stem"
  cp "$build_root/$stem.log" "$build_root/$stem-3.log"
  cp "$build_root/$stem.pdf" "$stem.pdf"
}
build_one dope-mfs
build_one supplement
# The anonymous PDF changes only the byline and the running header.
# The source dope-mfs.tex keeps Jepson Taylor and Alton Alexander.
python3 - << 'PY'
from pathlib import Path
src = Path("dope-mfs.tex").read_text()
pairs = (
    (
        "\\author{Jepson~Taylor and Alton~Alexander%\n"
        "\\IEEEcompsocitemizethanks{\\IEEEcompsocthanksitem J.~Taylor and A.~Alexander are with NEVERHUMAN Research.}}",
        "\\author{Anonymous%\n"
        "\\IEEEcompsocitemizethanks{\\IEEEcompsocthanksitem Affiliation withheld for review.}}",
    ),
    ("\\markboth{Taylor and Alexander}%", "\\markboth{Anonymous}%"),
)
for old, new in pairs:
    if src.count(old) != 1:
        raise SystemExit("author marker is not unique: " + old)
    src = src.replace(old, new, 1)
Path("dope-mfs-anonymous.tex").write_text(src)
PY
build_one dope-mfs-anonymous
rm -f dope-mfs-anonymous.tex
pdfinfo dope-mfs.pdf | awk '/Pages|Page size/'
pdfinfo dope-mfs-anonymous.pdf | awk '/Pages|Page size/'
pdfinfo supplement.pdf | awk '/Pages|Page size/'
