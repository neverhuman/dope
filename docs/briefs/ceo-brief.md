# D.O.P.E. v3.0.0 — Executive Brief

> **Download formatted 2-page PDF:** [**`dope-v3-ceo-brief.pdf`**](dope-v3-ceo-brief.pdf) *(publication layout with actual-size 2 KiB visual & Transfer Score frontier chart)*

**Downstream Objective-Preserving Encoding: Evaluating AI Models on Locked Enterprise Data Without Moving a Single Row**

> **Bottom Line / Key Takeaways**
> - **What DOPE does:** Compiles a table and its declared business target *inside your firewall* into a **≤ 2,048-byte file** — the size of a few paragraphs of text — containing fitted statistical summaries and schema, with **no complete input records**.
> - **Development-panel evidence:** Models trained on synthetic rows from DOPE's 1,787-byte median file achieve a **Transfer Score of 87.9** (100 = a second real sample; 0 = constant guess), versus **2.6** for stored real rows at the same 2,048-byte cap and **77.2** for uncapped ARF from a file **1,644× larger** (2.94 MB).
> - **Statistical nuance:** DOPE leads three SDV generators across all three development split sets; against ARF, the 95% paired interval includes zero, so DOPE is not shown to transfer better than ARF on this look.
> - **Governance boundary:** DOPE files are **derived data**. Release v3.0.0 is uncertified, provides no formal differential privacy mechanism, and is **not HIPAA de-identification**.

## 1. The Business Problem: Data Locked Behind Firewalls

High-value enterprise tables — healthcare records, banking and insurance ledgers, customer histories, and sensor logs — sit behind strict regulatory, contractual, and technical firewalls. Evaluating external AI/ML vendors or new models normally forces three bad choices:

- **Export records:** Months of legal review and an irreversible data transfer.
- **Trust public benchmarks:** Flying blind on generic datasets unlike your business data.
- **Build everything internally:** Slow, costly, and repeated for every new vendor.

## 2. What DOPE Does (Plain English)

DOPE changes what crosses the firewall — from a sensitive table to a tiny, inspectable generator built for one declared prediction target (such as churn, claims cost, or credit risk):

1. **Compile inside your firewall:** Run one signed CPU executable (`dope`, MIT licensed, zero cloud calls) where the table lives.
2. **Review a ≤ 2,048-byte artifact:** DOPE emits fitted statistical summaries and schema — no complete input records — with every stored category or bucket backed by at least 10 fitting rows.
3. **Let external vendors build outside:** Any number of vendors use the shared public `dope` tool to draw synthetic training rows from that tiny file, train and tune models normally, and return the trained models.
4. **Score inside on real held-out data:** Your team evaluates every returned model on real rows that never left your systems.

One 2 KiB file supports an entire multi-vendor bake-off without separate row-level access negotiations.

## 3. The Evidence at a Glance

**Transfer Score** measures downstream training value: models trained on synthetic rows are tested on real held-out targets, where **100** is a second real sample and **0** is a constant guess (higher is better; negative scores mislead models).

*Fresh-split DEV-SELECT development estimate (44 informative cells, 11 dataset families, 4× sampling; weakest of linear and boosted-tree learners scored per cell; failures kept at 0.0).*

| Generator / Reference | Transfer Score ↑ | Median File | Size vs. DOPE v3 |
| --- | ---: | ---: | ---: |
| **DOPE v3** (`auto-v8`, ≤ 2,048 B cap) | **87.9** | **1,787 B** | **1×** |
| Stored real rows (same 2,048 B cap) | 2.6 | 2,016 B *(28.5 rows)* | 1.1× |
| ARF (uncapped default) | 77.2† | 2.94 MB | 1,644× larger |
| SDV Gaussian copula (uncapped) | 40.2† | 141 KB | 79× larger |
| SDV TVAE (uncapped) | -3.9† | 475 KB | 266× larger |
| SDV CTGAN (uncapped) | -69.2† | 1.05 MB | 590× larger |
| *Scale: copy of fitting rows* | *96.1* | *—* | *—* |
| *Scale: bootstrap of fitting rows* | *91.2* | *—* | *—* |

*† Uncapped library defaults; every file exceeded 2,048 B (applying the 2,048 B cap penalty without refitting scores 0.0). Per-table sizes exclude the shared runtime binary or library used to sample rows.*

Across three development split sets, DOPE leads Gaussian copula (+47.8), TVAE (+91.9), and CTGAN (+157.1) with 95% intervals strictly above zero. Against ARF, the paired difference is **+10.8 points (95% CI -3.5 to +29.5)** — the interval includes zero, so DOPE is not shown to transfer better than ARF on this look. Under the benchmark's full decision rule, DOPE is **PROMOTE** against TVAE and **HOLD** against ARF, Gaussian copula, and CTGAN.

## 4. Governance, Risk & Honest Limits (Executive Decision Checklist)

> - **Derived data — legal/compliance decides:** Treat `.dope` files and synthetic CSVs as derived data. Schema and fitted summaries can still carry information about the source table; your governance team decides whether an artifact may cross a boundary.
> - **No formal differential privacy & not HIPAA de-identification:** v3.0.0 implements no formal differential privacy mechanism and is **not HIPAA de-identification** (neither Safe Harbor nor Expert Determination). Privacy checks are empirical audits on emitted rows (68.8% gate pass rate; zero exact-copy excess).
> - **Development estimates & uncertified release:** Results come from fresh **DEV-SELECT** splits; no confirmation, holdout, or sealed panel has been opened. The `v3.0.0` binary is uncertified, the paired interval against ARF includes zero, and cross-model feature-driver preservation is unmeasured.
> - **Built for training models, not realistic rows:** Designed for training and screening predictive models on a declared target. A boosted-tree discriminator still separates DOPE rows from real rows easily (median AUC 0.967).

## 5. How to Pilot & Next Steps

1. **Compile inside on one approved table:** Pick one non-sensitive or internally approved benchmark table with a clear target and compile a ≤ 2,048-byte artifact inside your firewall.
2. **Test models on synthetic rows:** Have an internal team or shortlisted vendor train and tune models using only synthetic rows drawn from that file.
3. **Score inside on held-out real rows:** Evaluate returned models on real data that never moved to validate vendor selection accuracy before expanding scope.

**Read next:** [CTO Technical Brief](cto-brief.md) · [Quick Start](../../README.md#quick-start) · [Real-Data Walkthrough](../../examples/v3-quickstart/README.md) · [Roadmap](../../ROADMAP.md) · [White Paper (PDF)](../whitepaper-v3/dope-v3.pdf) · [v3.0.0 Scoreboard Receipts](../../bench/real-world/results/scoreboard/v3.0.0/README.md)
