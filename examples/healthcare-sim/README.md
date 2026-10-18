# Offline simulated healthcare workflow

This is a deterministic software demonstration built entirely from explicitly
simulated patients. It uses no real patient records, clinical source, network,
or downloaded model. Its output is not clinical evidence, evidence about unseen
real patients, HIPAA de-identification, or formal differential privacy.

The fixture uses 160 fitting and 40 evaluation patients. All fitting encounters
occur before the split date and all evaluation encounters occur after it; patient
sets are disjoint. Each row predicts a simulated event in the next 30 days from
prior utilization, earlier visit count, a coarse prior lab category, a rare simulated condition,
and a care-path category. The lab has deterministic-seed missingness and three coarse observed levels. Features
are correlated by construction. Patient IDs and dates stay in the fixture
manifest and never enter the model CSV. Same-encounter and future-label columns
are forbidden by an exact feature allowlist.

## Four offline steps

Run from the repository root. `just fast` is the usual local Rust gate; build
only the CPU binary needed by this small smoke:

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

The four stages are prepare, compile, sample, and evaluate; the `verify` command
is a deterministic fixture check before the CLI run. Preparation and evaluation
use only Python's standard library. Compile and sample use the real Rust CLI;
the evaluator parses its JSON typed cells directly.
The evaluator compares a small ridge linear-probability model trained on
simulated fit rows with the same model trained on generated rows; both score on
the same simulated held-out rows. `receipt.json` contains actual result metrics
and artifact/sample byte counts. `manifest.json` records actual CSV byte sizes
through the files themselves and their SHA-256 digests. Re-running `prepare`
overwrites the fixture with identical bytes. Put all generated files under
`target/` so they remain local evidence.

One measured run used Linux x86_64 (`Linux xbabe2 6.8.0-146-generic`), with 1
visible CPU. Prepare took 0.21 s, fixture verify 0.05 s, compile 0.16 s, sample
0.08 s, and evaluation 0.18 s. The debug CLI build took 109 s. These are one
small-run measurements, not repeated service benchmarks. They do not justify
adopting the proposed five-minute CPU or fifteen-minute GPU service objective.
No GPU run is part of this demo. Measure repeated runs on intended hardware
before adopting either SLO.

An optional modern-model panel may only be run through an existing admitted
native adapter and its qualified installed checkpoint. This example contains no
mock native success, checkpoint discovery, acquisition, or qualification. When
no qualified adapter is supplied, the panel state is unavailable and contributes
no result. Simulated results remain separate from observational confirmation.

## Honest receipt

The receipt marks clinical, unseen-real-patient, HIPAA, formal-DP, and
observational-confirmation claims false. The simulated holdout is a software
fixture, not an external user study or independent source. A reader should treat
the result as a reproducible pipeline smoke only.
