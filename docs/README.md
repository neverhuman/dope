# Documentation

Product rules live in the Markdown notes. The white paper is the readable
argument for the generator comparison.

| Path | What it is |
| --- | --- |
| [architecture.md](architecture.md) | Rust product, V1 boundary, contract inputs |
| [testing.md](testing.md) | Commands, budgets, and repair evidence |
| [audit.md](audit.md) | Jankurai gate and exceptions |
| [audit-rubric.md](audit-rubric.md) | Finding severity |
| [release.md](release.md) | Certification and publication gates |
| [math.md](math.md) | What a program stores, why it transfers, the implementation audit, and the research plan |
| [cli.md](cli.md) | Command reference, including the table workflow |
| [../paper/](../paper/README.md) | The active manuscript and supplement |
| [whitepaper/](whitepaper/README.md) | The frozen historical paper, its supplement, and the scripts that regenerate every number |
| [internal/](internal/README.md) | Planning records, review inputs and archives kept for provenance |

Every number in the paper is a macro emitted from a committed, SHA-pinned
ledger under `research/benchmark/results/`; the analysis contract is
`research/benchmark/review_fixes/predeclare_v2.json`. A null release score
is printed as null.
