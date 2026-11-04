> Current authority: latest user requires Python ONLY for deep-learning training. Read ../README.md and the migration manifest first. This is a policy task, distinct from lower-case remote lane IDs. Proposed final repository destinations below do not authorize current authors to edit repository/shared roots; each author writes only its external packet/reserved namespace. No native execution, caller switch or qualification is implied.

# RS00: Port canonical v3 summary reducers to Rust

## Objective

Add Rust equivalents of `research/levels/v3_canonical_summary.py` at source head `a84bba031240582921b00532b23cab6d56f150ac`. Preserve the summary's exact JSON structure, comparison semantics, deterministic bootstrap stream, and legacy golden outputs. This is a metric/reporting port; it is not DL training and must not retain a Python production bridge.

## Read first

1. `research/levels/v3_canonical_summary.py`
2. `research/levels/v3_canonical_runner.py` call sites and summary/receipt assembly
3. `research/levels/world_class_eval.py::compute_elo_ratings`
4. `research/levels/v3_canonical_roster_eval.py::{CANONICAL_6_AUDITORS,CORE_COMPARATOR_AUDITORS}` and `v3_ablation_and_eval.py::ALL_TIERS`
5. `engine/dope-core/src/eval/mod.rs`, `engine/dope-core/src/dvs/{mod.rs,receipt.rs}`, their existing tests and canonical summary fixtures

## Ownership and files

Own Rust-only changes in `engine/dope-core/src/eval/` (prefer a focused `canonical_summary.rs` plus `eval/mod.rs` export) and focused Rust tests/fixtures. No Python source, runner, CI/guard, indexes, evidence, or existing golden fixture edits. Ask the repository owner before changing test ownership/registration or compatibility fixture bytes.

## Required behavior

Implement a typed reducer over public canonical ledger records:

- completed-row filtering; role and level grouping; planned/completed/missing/unexecuted/failure denominators;
- per-auditor sender-scaled, receiver raw, diagnostic, driver-fidelity, failure/floor/partial counts and means;
- comparator complete-case pairing, missing/failure counts, wins/ties/losses at 0.01 and 0.10 cutoffs, specified v2 exclusions;
- Elo using the exact method list/order, cell keying, per-role summaries and overall summary;
- dataset-level bootstrap resampling and paired-difference bootstrap intervals with NumPy PCG64 `default_rng(20261024)` stream equivalence, NumPy percentile interpolation, and Python-compatible ties-to-even rounding.

Use typed serde records or a validated input layer. Preserve exact field names, arrays/order, optional/null behavior, and rounding precision. Do not use `dope_core::eval::prediction_loss` as a substitute: it is an auditor loss function, not this Elo/report reducer.

## Tests and acceptance

- Build synthetic rows in Rust fixtures matching the public ledger schema; do not read real datasets or private evidence.
- First compare output bytes/parsed JSON against existing legacy canonical summary golden files at this pinned source. Cover empty input, one role, missing comparator, inapplicable v2 comparator, incomplete auditor, failures, all levels, and zero bootstraps.
- Add fixed bootstrap vectors proving exact NumPy sample indices, Elo CIs, paired mean/CI/SE, plus repeated-call determinism. Same seed alone is insufficient.
- Test stable JSON field/order/null behavior and denominator edge cases.
- Run focused tests: `cargo test -p dope-core --test seeded_auditors --offline` plus the new focused test target; run the relevant CLI integration target only after core parity. Use the existing admitted Cargo target and semaphore per current project instructions. No full build or data/model run.

## Completion proof

Report exact preimage/head, source/test hashes, focused command/terminal exit, fixture names, and which legacy outputs were byte-identical. Until exact bootstrap and legacy golden parity pass, the new Rust reducer is not production-ready and the Python call remains active only as migration history—not as accepted end-state.

## Stop conditions

Stop without replacing/update expected values if the current legacy goldens are unavailable, RNG equivalence cannot be demonstrated, the schema cannot distinguish missing from null, or output bytes drift. Escalate the exact missing fixture or RNG seam; no scientific qualification claim.

## Tracked migration acceptance IDs

- RS00-S1: closed typed summary input/output and resource refusals. Current verdict: NOT_RUN.
- RS00-S2: historical complete-case/missing/structural-exclusion/denominator behavior. Current verdict: NOT_RUN.
- RS00-S3: exact PCG64 draw stream, percentile/std/rounding compatibility. Current verdict: NOT_RUN.
- RS00-S4: Elo/paired output and ordering parity, with missing legacy fixture proof explicit. Current verdict: NOT_RUN.
- RS00-S5: native fixture terminal exits and nonzero counts under admission. Current verdict: NOT_RUN.
- RS00-S6: native caller route with no nontraining Python bridge. Current verdict: NOT_RUN.

These migration IDs are separate from the original D00–D11 65-criterion baseline. New nontraining test/checker implementations must be Rust; a 'Python policy test' means a Rust check of Python source, not a new Python test implementation. Training gradient-query/autograd losses are training-coupled; standalone inference, benchmark validation, reports and release decisions are Rust.
