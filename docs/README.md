# Documentation

Start with the [repository map](REPO_MAP.md) if you are new, and with
[the math](math.md) if you want to know why DOPE works and what to build next.

| Path | What it is |
| --- | --- |
| [REPO_MAP.md](REPO_MAP.md) | Every top-level directory, the three paper trees, and the cleanup plan |
| [math.md](math.md) | The program, the metric, the transfer theory, where the bytes go, and the research roadmap |
| [metric.md](metric.md) | Transfer Description Length (TDL): definitions, gates, result states |
| [claims-policy.md](claims-policy.md) | What the public surface may claim, evidence labels, banned phrases |
| [decisions.md](decisions.md) | Decisions in force and pending owner decisions |
| [architecture.md](architecture.md) | Rust product, V1 boundary, contract inputs |
| [cli.md](cli.md) | Native command reference |
| [data-contract.md](data-contract.md) | The mandatory target-last table contract |
| [testing.md](testing.md) | Commands, budgets, and repair evidence |
| [audit.md](audit.md), [audit-rubric.md](audit-rubric.md) | Jankurai gate, exceptions and finding severity |
| [release.md](release.md) | Certification and publication gates |
| [governance/](governance/) | Python-to-Rust migration plan and the Python exceptions inventory |
| [whitepaper/](whitepaper/README.md) | The **frozen** historical v2 paper, its supplement, and the scripts that regenerate every number |
| [../paper/](../paper/README.md) | The **active** TDL manuscript |
| [archive/](archive/) | Retired contracts and histories, kept so receipts still verify |

Every number in either paper is a macro emitted from a committed, SHA-pinned
ledger under `research/benchmark/results/` or a committed run receipt; the
analysis contract is `research/benchmark/review_fixes/predeclare_v2.json`. A
null release score is printed as null.
