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
Stage command diagnostics contain raw-stream hashes, byte counts, time, exit
status and known static error codes. They exclude raw error text and private paths.
The integer frequency is serialized as integer CSV text; decimal float text is
refused by the native integer type. Earlier failed attempts are retained.
Results may be poor or retention negative; that does not turn the demonstration
into a failed privacy or scientific claim. Feature support remains bounded by
the fit-range sampler and residual tails remain unqualified.

An optional programmatic comparison accepts a `LocalCheckpointHook` from
`research.benchmark.local_checkpoint` through `run(..., local_checkpoint=hook)`.
The command above keeps its default unavailable state and does not discover,
install or select a model. A separately admitted local custodian supplies an
existing frozen-context `ObservedLearner` port, its registry `AdapterAdmission`,
and independent expected identity, eligibility and native-reference SHA256 pins:

```python
hook = LocalCheckpointHook(offline_adapter, registry_admission,
                           expected_identity_sha256=identity_pin,
                           expected_eligibility_sha256=eligibility_pin,
                           expected_native_reference_sha256=native_reference_pin)
receipt = demo.run(installed_binary, pinned_source_bytes, new_output_directory,
                   1729, local_checkpoint=hook)
```

The existing offline port verifies its local code, dependency, license and weight
files before loading. Native execution refuses fixture factories. No checkpoint
or vendor code is run without a separately supplied adapter; the example contains
no checkpoint acquisition or automatic download step. The programmatic custodian
must make the repository modules available in its admitted Python environment.
The hook receives only fit-arm labels and original-unit query features. It scores
real/n/4n predictions against the same query truth outside the worker, recording
exact recipe/checkpoint pins, provided and actually consumed context, raw losses,
prediction digest and observed worker cost. Context and query policies come from
the frozen adapter identity; an oversized request refuses rather than silently
changing its policy.

Only Linux CPU workers are supported: at most 16,384 provided fit rows, 4,096
query rows, 128 features and one million input cells. The adapter's total wall
budget (at most 180 seconds) is shared across optional arms, including preparation
and failed attempts. Each fresh spawned child has wall, CPU and virtual-address-space
limits, with at most 4 GiB; measured peak RSS and adapter usage are also checked.
Those limits do not establish hostile-code filesystem/network isolation or
immutable snapshot custody. The admitted custodian still owns those controls.
Optional errors retain failed elapsed costs and a static unavailable/failed state
while the main demonstration continues. All qualification/admission flags remain
false. Synthetic factory fixtures are labeled `interface_fixture_completed`,
never vendor performance. Actual checkpoint/reference/privacy evidence is absent
from the existing Airfoil receipt; this code addition does not rerun or qualify it.
The default receipt remains `dope-public-airfoil-demo-v1`. Supplying a hook selects
`dope-public-airfoil-demo-local-checkpoint-v1` with a separate versioned comparison
section, preserving the old unavailable field's string type and v1 decoding.

The supplied adapter/runtime callbacks must support Python spawn serialization,
and the programmatic caller must use a standard `if __name__ == "__main__":`
entry. A fresh interpreter receives declared fit labels and query features,
without inheriting the evaluator's query-label memory. Nonserializable callbacks
return an explicit optional failure.
