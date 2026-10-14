# Public Airfoil regression demonstration

This example measures one public grouped development split. It does not certify
privacy, independent-source confirmation, a whole learner portfolio, or release.
The old `examples/readme-regression` example remains a toy smoke test.

The input is UCI Airfoil Self-Noise, dataset 291, DOI
[10.24432/C5VW2C](https://doi.org/10.24432/C5VW2C), by Brooks, Pope and Marcolini
(1989), licensed CC BY 4.0 according to the
[official source](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise).
The official archive and extracted file are SHA256 pinned in `run.py`. The
parser verifies 1,503 rows, five finite numeric predictors, an integer frequency
in Hz and a continuous sound-pressure target in dB. The source metadata's binary
angle hint is not used: the actual file contains 27 distinct angles.

Build and checksum-install the native binary as described in
[native-workflow.md](../../docs/native-workflow.md). Create a separate isolated
Python environment for this optional CPU demonstration, without system-site
packages, using the repository's pinned
[V1 CPU dependency requirements](../../validation/requirements-v1.txt).
The example imports only NumPy, scikit-learn and threadpoolctl for the learner;
native sampling itself requires no Python, Torch, GPU or checkpoint.

Run from the repository root with an installed native binary:

```sh
target/demo-env/bin/python -I -B examples/public-regression/run.py \
  --binary "$PWD/target/native-install/bin/dope-kernel" \
  --out target/public-demo/airfoil-run-1
```

The first invocation downloads the official 60,126-byte archive into the local
cache and verifies both archive and file digests. `--source LOCAL.dat` performs
the same verification without network. A new persistent output directory is
required, so earlier successful or failed evidence is retained.

The runner puts entire physical configurations on one side of a seeded 80/20
group split. Configuration means angle, chord, velocity and displacement
thickness; frequency is excluded so repeated frequency observations never cross
fit/query. Only fit rows reach the native schema transforms and compiler. The
fixed Chow–Liu symbolic profile uses a 60-second deadline and a complete
32,768-byte cap, including framing, schema and fitted affine state.

After the fit process exits, the runner reloads and samples at n and 4n. A fresh
native worker with an empty environment, closed inherited source descriptors and
default-deny syscall filter must reproduce each sample digest. Each arm fits the
same one-thread scikit-learn 1.7.2 StandardScaler/Ridge(alpha=1, SVD) recipe using
its own fitting rows. A separate closed-form ridge computation checks native
prediction parity. Every arm scores the same unseen original-unit real queries.

`receipt.json` and `report.html` show actual bytes, consumed rows, MSE/RMSE/MAE,
real/null losses, unbounded retention, replay/runtime inventory, repair counts,
versions and time. The report contains no source rows. Missing foundation-model
checkpoints, driver, geometry, privacy and confirmation remain explicit.
`attempts.jsonl` preserves started and terminal step costs even when a native
command or learner fails. The receipt identifies the runner checkout and actual
executable digest; the exact binary build attestation remains separate.
Results may be poor or retention negative; that does not turn the demonstration
into a failed privacy or scientific claim. Feature support remains bounded by
the fit-range sampler and residual tails remain unqualified.
