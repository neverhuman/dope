dope="$HOME/.local/opt/dope/3.0.0/bin/dope"
"$dope" fit train.csv --target y --format auto-v8 --budget 2048 -o model.dope
wc -c < model.dope
"$dope" sample model.dope -n 1000 --seed 1 -o synthetic.csv
"$dope" fit train.csv --target y --format 1 --budget 2048 -o audit-schema.dope
"$dope" eval -m audit-schema.dope --train train.csv --test test.csv \
  --synth synthetic.csv --auditors linear --seed 1 --json evaluation.json
