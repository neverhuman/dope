# TabSyn numeric runtime and adapter preparation

The adapter uses the Apache-2.0 author source at
`cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`, whose 96 source files are recorded
in [the original source audit](TABSYN_SOURCE_AUDIT.md). That historical audit
keeps its original scope. This preparation covers numeric regression only.
The adapted training-loop code carries Apache-2.0 attribution in its
[license sidecar](tabsyn_adapter.py.license), with the exact audited
[license](tabsyn-adapter.LICENSE) and [notice](tabsyn-adapter.NOTICE) preserved.
This historical preparation does not admit the benchmark matrix, a GPU runtime, native tuning, mixed
categories, or a reproduction of the paper's reported experiments. Official
tests stay sealed; MFS-v2, PTF-v1, release-safe L3 and superiority remain null.
The separate [numeric operation contract](TABSYN_NUMERIC_ADMISSION.md)
defines the new conditional fit/sample/native interface. Its externally
anchored guard and actual CUDA qualifications remain required; generated
CPU evidence does not satisfy them.

The existing xbabe3 preparation provides Python 3.10.19, PyTorch 2.0.1/CUDA
11.7, NumPy 1.24.4, pandas 1.5.3, SciPy 1.10.1, sklearn 1.2.2 and XGBoost
1.7.6. [The requirements lock](requirements-tabsyn.lock.txt) pins all 29
resolved distributions by wheel hashes. Installed files, Python and system
ELF providers are inventoried in the externally anchored runtime lock. Adding
NVTX 11.7.91 resolves Torch's required `libnvToolsExt.so.1` without changing
author source. The explicit inventoried library path is required; ambient
preloads, interpreter ZIP appearance and provider drift reject before imports.
This inventory establishes the generated CPU scope. GPU driver resolution
and the author's 36 `gpu_hist` auditors require separate measured admission.

[The guard](tabsyn_runtime_guard.py) checks the reusable runtime and a second
externally anchored manifest for this adapter's controls. The controls occupy
a separate sibling directory, preserving the previous frozen preparation.
Execution uses the pinned interpreter with `-I -S -B` and an empty bytecode
cache. The guard binds all author files to the audited hashes before installing
the allowed import paths. A caller must supply both manifest hashes from its
launch declaration; hashes computed from an untrusted current file do not
establish custody.
The generated entry compiles the hash-verified guard bytes with a loader that
returns that owned code object and never reads cached bytecode or replacement
source. Injected cache controls check this boundary before model imports.

[The numeric adapter](tabsyn_adapter.py) retains the author model definitions
and diffusion sampler. Its disclosed adaptations are:

- Empty categorical reconstruction has zero CE and accuracy. Numeric MSE,
  KL and their gradients retain the author formulas.
- Numeric joint columns contain the target first, then projected inputs.
  A quantile transform is fit only on training rows. No new study split is
  generated, and validation contains only rows derived from official training.
- Categorical-only validation still assigns zero weight to numeric MSE.
  The first full VAE checkpoint is retained; sampling uses final encoder and
  decoder weights. Encoder and decoder construction precedes training, as in
  the author code, preserving that RNG consumption order.
- DataLoader uses zero workers for the isolated process contract. Batch size
  4096, shuffling, schedulers, beta decay and diffusion patience remain fixed.
- Direct sampling receives explicit rows, device and 50 steps. The original
  CLI reads `args.steps` without forwarding it and fixes rows to training size.
  The adapter checks `n/2n/4n/8n` and sample seeds 101, 211 and 307.
- Safetensors and numeric NPZ state replace pickle and the saved latent row
  matrix. Charged bytes include every saved checkpoint, final encoder/decoder,
  learned quantiles, latent mean, projection and completion manifest. Sampling
  verifies the entire externally supplied inventory before dependency imports.

The author budgets are 4000 VAE epochs and at most 10001 diffusion epochs,
with diffusion stopping after 500 epochs without improvement. The whole fit
ceiling is 600 seconds across preprocessing, both stages and export; a failed
fit retains partial files and cannot produce an eligible completion manifest.
The GPU ceiling remains 16 GiB and requires a future process watchdog. The
default public fit entry admits only an explicitly generated CPU fixture with
two epochs per stage. Real fits, GPU samples and native evaluation require the
separate operation grant and measured qualification closure. Without those,
they reject before numerical dependency initialization.

[The generator proposal](tabsyn-native-grid.proposal.json) retains the default
beta 0.01 and proposes 0.001 through the author-exposed parameter. This is a
campaign proposal; the author supplies no generator search grid. It is not a
frozen tuning or execution admission. Any future admitted native evaluation
must call immutable `_evaluate_regression` and maximize
`best_r2_scores.XGBRegressor.r2`, retaining all 36 auditor configurations,
the seeded synthetic shuffle, log-clipped fit and real labels, and raw internal
synthetic validation labels. Uninformative transformed targets cannot select
a native winner. Failed trial time remains charged; selection allows at most
eight trials and 12 hours per cell, then breaks equal native scores by artifact
bytes and configuration hash. Native values cannot rank different methods.

Focused opaque tests run with:

```sh
python3 -B -m unittest research.benchmark.tests.test_tabsyn_contract research.benchmark.tests.test_tabsyn_admission -v
```

[The generated contract](tabsyn_generated_contract.py) checks the original
empty-category failure, exact numeric loss and gradients, safe fit/sample
replay, twelve sample cells, provider/artifact drift and partial-trial
exclusion. Its launch additionally needs explicit mutual peer binding with
other research queues and fresh CPU/RAM/disk admission. Existing private
prototype measurements remain historical evidence; a changed adapter needs
its own fresh generated receipt before its runtime contract can be claimed.
