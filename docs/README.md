# Documentation

Start with [getting started](getting-started.md) to install the signed engine
and fit a table, and with [AGENTS.md](../AGENTS.md) if you are an automated
coding agent. The current manuscript is
[whitepaper-v3/dope-v3.pdf](whitepaper-v3/dope-v3.pdf).

| Path | What it is |
| --- | --- |
| [getting-started.md](getting-started.md) | Install `dope-engine-v3.0.0` and run `dope fit --format auto-v8` |
| [../AGENTS.md](../AGENTS.md) | Public engine usage, claim limits, and repository rules for editors |
| [whitepaper-v3/](whitepaper-v3/README.md) | **Current white paper** (`dope-v3.pdf`): Transfer Score for `auto-v8` against ARF, SDV Gaussian copula, TVAE, and CTGAN. Every number is a macro from committed receipts |
| [signed-engine-release.md](signed-engine-release.md) | Verify `scripts/install.sh`, `SHA256SUMS`, and the Sigstore attestation |
| [whitepaper-v2/](whitepaper-v2/README.md) | Archived DOPE v2.0.0 manuscript: [`dope-v2.tex`](whitepaper-v2/dope-v2.tex) ([`dope-v2.pdf`](whitepaper-v2/dope-v2.pdf)), Format 2, SingleEncV7, and the v2 scoreboard |
| [../bench/real-world/results/scoreboard/v3.0.0/](../bench/real-world/results/scoreboard/v3.0.0/README.md) | The v3.0.0 scoreboard (DEV estimate): TRG, the receipts behind the README `DOPE1 SCOREBOARD` block and hero, and the command that re-renders them |
| [../bench/real-world/results/scoreboard/v2.0.0/](../bench/real-world/results/scoreboard/v2.0.0/README.md) | The v2.0.0 scoreboard: TRG (the twin-referenced gap) definition, the `dope_kpi` receipts behind the README `DOPE1 SCOREBOARD` block and hero, and the command that regenerates them |
| [../paper/](../paper/README.md) | Archived v1.0.0 manuscript: [`paper/dope1.tex`](../paper/dope1.tex) and [`paper/dope1-supplement.tex`](../paper/dope1-supplement.tex) |
| [multilevel-landmark-dl-study.md](multilevel-landmark-dl-study.md) | Canonical 1,000-dataset (`DOPE-MLC-v2`) and 1,069-dataset (`DOPE-MLC-v3`) deep learning study, round-by-round convergence tables, and `v8` receipt provenance |
| [model-sizes-and-elo.md](model-sizes-and-elo.md) | Complete model weight inventory (`SafeTensors`, `.pt`, `.dml3`), SHA-256 digests, wire byte accounting, and Bradley–Terry Elo tables |
| [jopedime-integration.md](jopedime-integration.md) | JopeDime and JopeDimeMicro source pin at `dope-engine-v3.0.0` (`e15c415b`), the `v1.0.0` MLC checkpoint, encoder compatibility, and v3 synthetic-row generation |
| [optional-landmarks-qualification.md](optional-landmarks-qualification.md) | Optional and progressive landmarks qualification (`none` / `progressive` / `full`): quality per mask level, retention, N300 subsample agreement, progressive transitions, and none-vs-full latency by regime (resident/session gate; strict-cold reported separately), plus `resident_release_check` usage |
| [observed-landmark-inputs.md](observed-landmark-inputs.md) | Observed landmark-input helper contract (`compute_observed_landmark_inputs_default`) and zero-filled unobserved slot semantics |
| [math.md](math.md) | The program, the metric, the transfer theory, where the bytes go, and the research roadmap |
| [metric.md](metric.md) | Transfer Description Length (TDL): definitions, gates, result states |
| [claims-policy.md](claims-policy.md) | What the public surface may claim, evidence labels, banned phrases |
| [benchmark.md](benchmark.md) | Public-table benchmark protocol, datasets, baselines, and auditors |
| [architecture.md](architecture.md) | Pure-Rust engine product, compatibility boundary, and contract inputs |
| [cli.md](cli.md) | Historical `dope-kernel` facade (`compile-raw`, `fit-table`). The signed engine commands are in [engine/README.md](../engine/README.md) |
| [data-contract.md](data-contract.md) | The mandatory target-last table contract |
| [testing.md](testing.md) | Deterministic test commands, budgets, and repair evidence |
| [audit.md](audit.md), [audit-rubric.md](audit-rubric.md) | Jankurai gate, exceptions, and finding severity |
| [release.md](release.md) | Certification and publication gates |
| [decisions.md](decisions.md) | Decisions in force and pending owner decisions |
| [REPO_MAP.md](REPO_MAP.md) | Top-level directory map |

Every number in the white paper and README is emitted from a committed, SHA-pinned benchmark receipt (`bench/real-world/results/summary.json` and `bench/real-world/results/multilevel-convergence-snapshot.json`; for the v3.0.0 and v2.0.0 scoreboards, the receipts in `bench/real-world/results/scoreboard/v3.0.0/` and `bench/real-world/results/scoreboard/v2.0.0/`).
