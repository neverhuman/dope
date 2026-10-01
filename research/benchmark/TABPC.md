# TabPC author runtime and validation adapter

`tabpc-source.lock.json` pins the Apache-2.0 author repository and its
GPL-3.0-or-later Cirkit submodule. This is an isolated research runtime;
neither upstream package nor trained artifacts enter the MIT product bundle.
The author sources, dependency sources, environment and compatibility patch
are hashed by the immutable scratch runtime lock before every adapter action.
The fit and sample entry points validate the runtime before importing NumPy,
Torch or the author packages. Two instrumented entry-point tests prove changed
dependencies are rejected without executing their initializers.
Malformed CSV or numeric fields produce a generic error with the original
exception context suppressed. A regression checks that neither the error nor
its formatted traceback contains the source field. This repair has a separate
runtime version and preserves all earlier fit and sample receipts.

The adapter calls the author's circuit trainer with the existing grouped fit
and validation partitions. Every preprocessor and the circuit structure fit
only on the fit partition. It selects checkpoints and configurations by the
author's transformed-space validation mean negative log likelihood. Native
likelihood values are not ranked across different methods. The bounded search
contains the author default and seven declared configurations from the
author's likelihood grid, with at most eight trials, including failures.
For News, the largest grid point uses the author's 1024-unit adjustment.

The author sampler resets CUDA memory statistics even when sampling on CPU
with CUDA hidden. The published one-line guard changes only that reset.
The original failure and exact sample parity after the patch are receipted.
Upstream source archives remain intact. A first runtime manifest also failed
closed on upstream documentation symlinks; its failure and corrected manifest
are retained separately.
An independent review also found that the first adapter imported NumPy/Torch
too early. The repaired runtime preserves the earlier GPU fit receipts and
their exact source version. For verification, copies of the toy artifacts
rebind only the adapter runtime metadata; every learned-parameter byte remains
unchanged, and four current CPU sample processes reproduce the earlier hashes.
No additional training is claimed or charged for this integrity-order repair.

The trainer retains the concatenated fit/validation tensor for serialization.
The adapter replaces it with an empty tensor carrying the same metadata before
calling the original store/load APIs. Exact samples before removal and after
serialization are checked. The complete artifact inventory charges circuit
weights, symbolic structure, every preprocessor, configuration and projection,
including the empty serialized metadata tensor. Learned marginal/quantile
parameters remain charged. They do not establish a privacy pass.

Two generated-input contracts cover regression and binary targets. Each checks
GPU fitting, finite native validation likelihood, deterministic serialization
and two independent CPU-only sample processes. These are contract probes;
they are neither benchmark results nor reproductions of reported experiments.
They confer no L3, MFS-v2, PTF-v1, formal DP or production certification claim.
Final campaign admission still requires the five consistent frozen locks.

## Complete native pilot validation

The corrected local bibliography records the author study as
`scassola2026sobering` (A Sobering Look at Tabular Data Generation via
Probabilistic Circuits).

The frozen Adult/California/News pilot now accounts for all 24 author-native
trials (eight per dataset): 13 successes, eight Adult GPU allocation failures,
and three News 600-second timeouts. Configuration selection minimizes only
transformed validation mean NLL. Adult has no successful native candidate;
its unavailable tracks confer no DOPE win.

The separate fixed-fit-23 common-validation schedule accounts for all 48
planned samples: 45 successes and three News tuned `8n` timeouts. All 24
planned `n`/`4n` metric cells completed and replayed exactly. Copies rebind
only runtime metadata to V5; every learned byte and charged artifact size
matches its original fit. Common validation starts no new fit or tuning.

The [report](results/pilot24-tabpc-native.md),
[JSON](results/pilot24-tabpc-native.json),
[schema](results/pilot24-tabpc-native.schema.json),
[CSV](results/pilot24-tabpc-native.csv), and
[SVG](results/pilot24-tabpc-native.svg)/[PDF](results/pilot24-tabpc-native.pdf)
retain default/native-selected outcomes, all failures, native likelihood,
three common utility auditors, copy controls, bytes, costs and immutable
receipt/source/runtime hashes. Regenerate with
`python3 -m research.benchmark.publish_tabpc_native` and
`python3 -m research.benchmark.publish_tabpc_native_figure`; `--from-json`
rebuilds tables without scratch access. These are descriptive single-fit
validation results, separate from the unfinished final five-fit campaign.
Official tests remain sealed and all gated scores remain null.

Run the boundary checks without optional upstream libraries:

```sh
python3 -m unittest research.benchmark.tests.test_tabpc_adapter -v
```

The scratch runtime is invoked through a JSON request:

```sh
python -m research.benchmark.tabpc_adapter /path/to/frozen.request.json
```

Fit requests name only verified `train.csv`, `validation.csv` and
`projection.json` inputs, their hashes, the runtime lock and its hash, a frozen
configuration, fit seed and new artifact directory. Sample requests carry the
complete expected artifact inventory, runtime identity, row count and sample
seed. The outer admitted worker enforces the 600-second fit deadline, 16-GiB
GPU ceiling, exclusive GPU ownership, disjoint CPU slots and scratch budget.
