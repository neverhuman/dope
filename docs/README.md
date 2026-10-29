# Documentation

Start with the [repository map](REPO_MAP.md) if you are new, and with
[the math](math.md) if you want to know why DOPE works and what to build next.

| Path | What it is |
| --- | --- |
| [../paper/](../paper/README.md) | **The DOPE White Paper**: [`paper/dope1.tex`](../paper/dope1.tex) (`target/paper/dope1.pdf`) and [`paper/dope1-supplement.tex`](../paper/dope1-supplement.tex) (`target/paper/dope1-supplement.pdf`), covering the standalone Rust `dope` engine, 16-table real-world study, and 1,000+ dataset `DOPE-MLC-v2`/`v3` study |
| [multilevel-landmark-dl-study.md](multilevel-landmark-dl-study.md) | Canonical 1,000-dataset (`DOPE-MLC-v2`) and 1,069-dataset (`DOPE-MLC-v3`) deep learning study, round-by-round convergence tables, and `v8` receipt provenance |
| [model-sizes-and-elo.md](model-sizes-and-elo.md) | Complete model weight inventory (`SafeTensors`, `.pt`, `.dml3`), SHA-256 digests, wire byte accounting, and Bradley–Terry Elo tables |
| [jopedime-integration.md](jopedime-integration.md) | **Downstream Integration & Pinning Guide (`JopeDime` & `JopeDimeMicro`)**: `v1.0.0` weight asset + signed `dope-engine-v1.1.0` (`15e12aa3`) source pinning, `[B, 256] + [B, 64, 17] -> [B, 629]` and `[B, 512]` contracts (`DopeEncoder629` and Rust `MlcV3Model`), `encoder_629` and `resident_release_check` examples, CPU export, CUDA capture, and proposed downstream projection optimization |
| [optional-landmarks-qualification.md](optional-landmarks-qualification.md) | Optional and progressive landmarks qualification (`none` / `progressive` / `full`): quality per mask level, retention, N300 subsample agreement, progressive transitions, and none-vs-full latency by regime (resident/session gate; strict-cold reported separately), plus `resident_release_check` usage |
| [observed-landmark-inputs.md](observed-landmark-inputs.md) | Observed landmark-input helper contract (`compute_observed_landmark_inputs_default`) and zero-filled unobserved slot semantics |
| [math.md](math.md) | The program, the metric, the transfer theory, where the bytes go, and the research roadmap |
| [metric.md](metric.md) | Transfer Description Length (TDL): definitions, gates, result states |
| [claims-policy.md](claims-policy.md) | What the public surface may claim, evidence labels, banned phrases |
| [benchmark.md](benchmark.md) | Public-table benchmark protocol, datasets, baselines, and auditors |
| [architecture.md](architecture.md) | Pure-Rust engine product, compatibility boundary, and contract inputs |
| [cli.md](cli.md) | Native command reference, including the table and `mlc-v3` workflows |
| [data-contract.md](data-contract.md) | The mandatory target-last table contract |
| [signed-engine-release.md](signed-engine-release.md) | Verified `scripts/install.sh` workflow, `SHA256SUMS` + `BUILDINFO.json` integrity, and Sigstore OIDC release attestations |
| [testing.md](testing.md) | Deterministic test commands, budgets, and repair evidence |
| [audit.md](audit.md), [audit-rubric.md](audit-rubric.md) | Jankurai gate, exceptions, and finding severity |
| [release.md](release.md) | Certification and publication gates |
| [decisions.md](decisions.md) | Decisions in force and pending owner decisions |
| [REPO_MAP.md](REPO_MAP.md) | Top-level directory map |

Every number in the white paper and README is emitted from a committed, SHA-pinned benchmark receipt (`bench/real-world/results/summary.json` and `bench/real-world/results/multilevel-convergence-snapshot.json`).
