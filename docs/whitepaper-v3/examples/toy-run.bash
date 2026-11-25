(
  set -euo pipefail
  export LC_ALL=C
  dope="$HOME/.local/opt/dope/3.0.0/bin/dope"
  cd "$(mktemp -d "${TMPDIR:-/tmp}/dope-quick-start.XXXXXX")"

  fixture='BEGIN {
    print "a,b,y"
    for (i=first; i<end; i++) {
      a=(17*i%997)/997; b=(31*i%991)/991
      print a "," b "," 2*a+b*b
    }
  }'
  awk -v first=0 -v end=160 "$fixture" > train.csv
  awk -v first=160 -v end=200 "$fixture" > test.csv

  "$dope" --threads 1 fit train.csv --target y --format 1 --budget 2048 --seed 42 -o model.dope
  bytes=$(wc -c < model.dope)
  test "$bytes" -le 2048
  printf 'Artifact size: %s bytes\n' "$bytes"
  "$dope" --threads 1 sample model.dope -n 640 --seed 7 -o synthetic.csv
  "$dope" --threads 1 eval -m model.dope --train train.csv --test test.csv \
    --synth synthetic.csv --auditors linear --seed 7 --json evaluation.json
  printf 'Workflow files: %s\n' "$PWD"
)
