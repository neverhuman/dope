# DOPE engineering and validation status

The twelve engineering packets have implemented source or readiness controls. **The assignment is not fully qualified and ALL CI is not passing.** Full exact-head CI, native execution and the blocked qualification prerequisites remain open. The original task text is a historical planning baseline; this page and the [criterion matrix](tasks/inventory.json) describe execution separately.

## Latest user language policy: NOT_COMPLIANT

Python is permitted only for genuine deep-learning training and necessary training-coupled state. The retained v3 summary/Elo and evaluation/orchestration paths are nontraining migration debt, not approved Python end state. The current snapshot has 890 tracked Python files; 787 runtime-inventory paths exclude103 research/levels files. No inventory or historical exception is a waiver. [The Rust migration queue](language-policy/README.md) tracks seven external source-only lanes and separate LP-/RS-/rs- acceptance IDs. No Rust packet has been adopted or natively qualified by this documentation update. Preserve all compatibility/test/evidence history; the original 65 criteria below remain their historical scoped baseline, not new-policy completion.

## Source and evidence boundaries

- Planning base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`.
- Current code checkpoint: `409c84e3486fe820776aa781a37b373954653c94`, on the same `fast/next-version-dope-20261103` branch and isolated worktree. Later documentation-only commits do not constitute new source qualification.
- Main/default-branch README CI failed at the planning base because `test_lineage.py` had 505 lines while the unchanged guard requires strictly fewer than 500. Local commit `63c24d02974a14767eb9b96314831268be8ca762` reduces it to 499 with an identical Python AST.
- Four additional branch source-cap violations were repaired by private helper extraction in `35d2f7955312b7293a749d70c077992a7425ccf4` and the code checkpoint above. Numerical bodies, existing tests, public training classes/checkpoint schema and metric v1/v2 dispatch remain preserved. The new v3 summary helper enters future producer source hashes; historical receipts and identity records were not changed.
- The static tracked-source inventory covers 1,289 governed source files and reports zero cap violations. This is a source inspection, **not execution of the Rust guard**. Its threshold and exemptions remain unchanged.

A normal fast-forward source-preservation push was independently read back at `c74320ce032008b1bca841daab0ec8d2b4460756` on the existing fast branch. Final documentation-only successors and their exact remote-head proof are recorded in the stopped checkpoint receipt; no PR or workflow was launched.

## Named acceptance criteria

The preserved 65-criterion baseline is **30 scoped PASS / 31 NOT_RUN / 4 BLOCKED**. PASS describes only the specific evidence cited by that criterion; it never means full CI, scientific efficacy, privacy certification, release or serving approval.

| Task | Implemented scope | Execution disposition | PASS / NOT_RUN / BLOCKED |
| --- | --- | --- | --- |
| D00 | Contracts and lineage decisions | Source implemented; validation incomplete | 4 / 0 / 1 |
| D01 | Prospective common REAL-FIT evaluation | Source implemented; validation incomplete | 5 / 0 / 0 |
| D02 | Additive informative driver rank | Source implemented; validation incomplete | 1 / 4 / 0 |
| D10 | Explicit local product gate | Source implemented; validation incomplete | 5 / 0 / 0 |
| D03 | Minimal native SDK facade | Source implemented; validation incomplete | 1 / 4 / 0 |
| D04 | Opt-in High-required landmarks | Source implemented; validation incomplete | 0 / 5 / 0 |
| D05 | FIT-only NC1 normalization | Source implemented; validation incomplete | 1 / 5 / 0 |
| D06 | DVS receipt and Audit Card | Source implemented; validation incomplete | 0 / 6 / 0 |
| D07 | Closed DML4 wire and sampler | Source implemented; validation incomplete | 1 / 5 / 0 |
| D08 | Bounded CPU trainer and exact resume | Source implemented; validation incomplete | 5 / 0 / 0 |
| D09 | V4 model and served-graph objective | Source implemented; validation incomplete | 4 / 2 / 0 |
| D11 | Qualification readiness controls | Readiness HOLD | 3 / 0 / 3 |

Every success ID and evidence explanation remains in [the machine-readable task inventory](tasks/inventory.json). The counts above have not been promoted merely because source exists or helpers were reorganized. The shared [machine status](../STATUS.json) retains the same distinctions.

## Checks actually completed

At the current code checkpoint, bounded existing CPU checks passed: **50 next-version tests, zero skips; one selected synthetic v3 canonical-summary fixture; 129 Python legacy golden cases at tolerance 1e-12**. The v3 fixture uses synthetic rows; the separate Rust-parity/model-checkpoint cases in its module were not run. Source/registration syntax and diff checks passed.

The earlier Rust 1.88 offline locked engine workspace/all-targets compilation PASS belongs only to `84a55cf95c33b22964e976c49b41cde7144671f1`. It executed no tests and is not final-head proof. Prior native test attempts were stopped for capacity before any test executed, including the 34-second 08:37 attempt that yielded when xbabe2 ledger work restarted.

Evidence is retained under `/home/ubuntu/jope_plan/evidence/next-version-implementation-20261103/`, including `ci-assist/lineage-source-cap/`, `ci-assist/rust-source-caps/`, `ci-assist/python-source-caps/` and `ci-assist/cpu-after-caps-409c84e3/`. Their source pins and terminal results take precedence over broad status labels.

## Remaining gates and stop conditions

The 11:36:58 coordinator ACK admits only one focused source-limit test at nice19/-j1 through the existing semaphore **if the 85% allocation guard admits the existing target**. Quiet source-preservation pushes to the existing fast branch are user-authorized after verifying that no workflow or open PR will trigger CI. Full native gates, PR creation and dispatched CI still need separate admission. No alternate target/cache, guard bypass, cleanup of others' storage, or foreign-job signal is authorized. A refused or unstarted test is NOT_RUN, never PASS. Final frozen-head preflight and any refusal are recorded externally.

After capacity admission, execute the focused native guard and required source-bound engine/root checks, then the ordinary full CI routes. `just check-all` covers kernel plus engine; the shared-levels next-version Python/native shards require their own admitted workflow route because the quiet fast branch does not trigger that matrix automatically. Stop on a failing check, renewed reserved ledger activity, allocation refusal or changed source pin; preserve logs and fix forward within ownership.

D00-S4 and D11-S2/S3/S4 remain blocked on actual campaign roster/exposure/custody, frozen owner hypotheses/thresholds/resources and admission. The [readiness manifest](../../v4-qualification-readiness.md) is HOLD. The CPU trainer does not support CUDA optimizer training. No training campaign, protected outcome access, model promotion, privacy certification, merge, release or serving activation has been performed or inferred.
