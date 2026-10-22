#!/usr/bin/env bash
# Exercise the actual deliverable with no executable helpers on its PATH.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $# == 1 && -f "$1" && -x "$1" && ! -L "$1" ]] || exit 2
binary=$(realpath -- "$1")
mkdir -p target/tmp
work=$(mktemp -d "$PWD/target/tmp/engine-smoke.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
native() { env -i PATH="$work/no-tools" "$binary" --threads 2 "$@"; }
make_csv() {
  awk -v first="$1" -v count="$2" 'BEGIN {print "a,b,y"; for(i=first;i<first+count;i++){a=(i*17%997)/997;b=(i*31%991)/991;print a "," b "," 2*a+b*b}}' > "$3"
}
make_csv 0 160 "$work/train.csv"
make_csv 160 40 "$work/test.csv"
cut -d, -f1,2 "$work/test.csv" > "$work/unlabelled.csv"
native doctor --json > "$work/doctor.json"
jq -e '.implementation == "native_rust" and .python_required == false and .libtorch_required == false and .threads == 2 and .backends == ["cpu"]' "$work/doctor.json" > /dev/null
native fit "$work/train.csv" -t y --task regression --size small --max-parameters 200 \
  --no-trees --no-audit --plan 3,2,0 --device cpu -o "$work/model.dope" \
  --report "$work/fit.json" --json > "$work/stdout.json"
diff <(jq -S . "$work/fit.json") <(jq -S . "$work/stdout.json")
native inspect "$work/model.dope" --json > "$work/inspect.json"
jq -e '.budget == 2048 and .bytes <= 2048 and .parameters.total <= 200' "$work/fit.json" > /dev/null
native sample "$work/model.dope" -n 11 --device cpu -o "$work/raw.csv"
native sample "$work/model.dope" -n 11 --space ml --device cpu -o "$work/ml.csv"
native predict "$work/model.dope" "$work/unlabelled.csv" -o "$work/predictions.csv"
native encode "$work/test.csv" --model "$work/model.dope" -o "$work/encoded.csv"
[[ $(wc -l < "$work/raw.csv") == 12 && $(wc -l < "$work/ml.csv") == 12 ]]
[[ $(wc -l < "$work/predictions.csv") == 41 && $(wc -l < "$work/encoded.csv") == 41 ]]
native eval -m "$work/model.dope" --train "$work/train.csv" --test "$work/test.csv" \
  --auditors linear --factor 1 --json "$work/eval.json" > /dev/null
native frontier --train "$work/train.csv" --test "$work/test.csv" -t y \
  --budgets 2KiB,4KiB --auditors linear --json "$work/frontier.json" > /dev/null
native throughput "$work/model.dope" -n 1024 --repeats 1 --device cpu > "$work/throughput.json"
jq -e '.rows == 1024 and .rows_per_second > 0' "$work/throughput.json" > /dev/null
native fit-tiers "$work/train.csv" -t y --task regression --device cpu \
  --dataset-id release-smoke-fixture --source synthetic-fixture --data-license MIT \
  -o "$work/tiers" > /dev/null
jq -e '.artifacts | length == 3' "$work/tiers/catalog.json" > /dev/null
printf '%s\n' 'Rust engine smoke passed: all customer commands run with an empty tool PATH.'
