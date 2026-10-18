# Command line workflows

## Offline healthcare simulation

The [healthcare simulation example](../examples/healthcare-sim/) demonstrates
the actual native CPU CLI on deterministic, explicitly simulated patient rows.
It does not use real patient data or claim clinical validity, performance on
unseen real patients, HIPAA de-identification, formal differential privacy, or
observational confirmation.

From the repository root, prepare and check the fixture, compile only its
preprediction-time fit rows, sample from the compiled artifact, then evaluate
the generated rows against a patient-disjoint later-time simulated holdout:

```sh
python3 examples/healthcare-sim/demo.py prepare --out target/healthcare-sim
python3 examples/healthcare-sim/demo.py verify --out target/healthcare-sim
cargo build --locked --bin dope-kernel
target/debug/dope-kernel compile-csv --csv target/healthcare-sim/fit.csv \
  --target event_30d --candidate chow_liu --seed 1729 \
  --deadline-seconds 60 --out target/healthcare-sim/model.dpk
target/debug/dope-kernel sample-mixed --artifact target/healthcare-sim/model.dpk \
  --rows 640 --seed 1729 --out target/healthcare-sim/sampled.json
python3 examples/healthcare-sim/demo.py evaluate --out target/healthcare-sim \
  --sampled target/healthcare-sim/sampled.json
```

`verify` checks exact feature headers, disjoint simulated patient sets, the
date boundary, and a negative future-label leakage probe. The receipt and
manifest report actual fixture, artifact, and sample sizes and measured
outcomes. The small classical evaluator uses a standard-library ridge
linear-probability model on CPU. The optional modern-model panel is available
only with a separately admitted native adapter and qualified installed model;
the example never substitutes a fixture for native success. Runtime objectives
must be based on measurements from the intended hardware. Simulated results do
not count as observational evidence.

Reference-run measurements on Linux x86_64 with 1 visible CPU: prepare 0.21 s,
verify 0.05 s, compile 0.16 s, sample 0.08 s, evaluate 0.18 s, and debug CLI
build 109 s. These single-run measurements do not support adopting the proposed
five-minute CPU or fifteen-minute GPU objective. The run produced 9,665-byte fit
and 2,469-byte evaluation CSVs, a 2,697-byte artifact, and a 146,482-byte typed
sample. The 80-row simulated evaluation Brier score was 0.1173 after sampled
training and 0.1164 after fit-row training; respective Brier skills versus the
same null were 0.0109 and 0.0183. `target/healthcare-sim/receipt.json` records
these values after completing the steps.

## General compile and sample

For other headered CSV data, the core CLI supports `compile-csv`, `inspect`, and
`sample`. The target defaults to the last CSV column unless `--target` names a
column or gives its zero-based index. The compiled artifact does not preserve
training rows. A successful compile is not a release or privacy certification.
Keep protected or otherwise sensitive data within its authorized environment.
