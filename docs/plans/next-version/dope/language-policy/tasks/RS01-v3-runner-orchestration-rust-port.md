> Current authority: latest user requires Python ONLY for deep-learning training. Read ../README.md and the migration manifest first. This is a policy task, distinct from lower-case remote lane IDs. Proposed final repository destinations below do not authorize current authors to edit repository/shared roots; each author writes only its external packet/reserved namespace. No native execution, caller switch or qualification is implied.

# RS01: Move canonical v3 evaluation and runner orchestration to Rust

## Objective and dependency

After RS00 reducer parity is accepted, replace the Python production path in `research/levels/v3_canonical_runner.py` with Rust-owned preparation, evaluation, scheduling, receipts, aggregation, and CLI. Python remains eligible only for a typed callback that performs actual DL model training; evaluation and report assembly stay Rust.

## Read first

1. `research/levels/v3_canonical_runner.py` functions `_load_v2_panel_scopes`, `_prepare_dataset_split`, `evaluate_dataset_across_tiers`, `evaluate_dataset_across_tiers_prospective`, `_worker_loop`, `_reap_worker_pgroups`, `_collect_source_sha256s`, `_build_receipt_common`, `run_bounded_canonical_eval`, `aggregate_canonical_runs`, `main`
2. `research/levels/v3_canonical_summary.py` and RS00 accepted reducer/tests
3. `research/levels/v3_ledger_replay.py`, `v3_canonical_roster_eval.py`, `v3_ablation_and_eval.py` for the consumed preparation/evaluation contract
4. Rust `engine/dope-core/src/{eval,mlc_v3,dvs}` and `engine/dope-cli/src/{args.rs,main.rs,mlc_v3.rs,levels.rs,dvs.rs}`
5. Existing `engine/dope-cli/tests/{next_version_cli.rs,cli.rs}` and v3 resident/evaluator tests

## Ownership / scope

Own new Rust evaluation/replay module(s) under `engine/dope-core/src/` and one CLI command module registered in `engine/dope-cli/src/{args.rs,main.rs}`. Coordinate exact file ownership with the Rust CLI owner before edits. No Python production wrapper/bridge in final path; retain Python DL training callback only behind a Rust-owned typed protocol if a training step is genuinely required. No datasets, private evidence, CI exclusions, source guards, indexes, or claim/receipt custody edits.

## Smallest safe implementation sequence

1. Specify versioned public input and output schemas from synthetic fixtures; pin physical row roles, schema/provenance IDs, split seeds/caps, native-report identities, and expected refusal codes.
2. Port manifest validation, content identity, role/split selection, fit-only transform, deterministic seeded subsampling, and canonical ordering to Rust. Reuse existing normalization/encoding APIs only if their exact transform matches frozen v3 behavior; otherwise add a dedicated compatibility path.
3. Port evaluator dispatch and per-cell execution. Reuse Rust `dope_core::eval` auditors only after protocol parity is shown. Evaluation metrics, feature importance, retention, wire transformations, and all scoring remain Rust. If the legacy exact auditor is a Python DL net, define a training-only callback boundary and validate its output in Rust; no Python `predict`, summary, or report callback.
4. Port bounded scheduling, timeout/interruption handling, safe worker lifecycle, receipt/source digest collection, atomic output writing, and run aggregation into Rust. Rust owns run state and terminal/partial status.
5. Register a dedicated `dope` command in `engine/dope-cli/src/args.rs` and dispatch from `main.rs`; preserve legacy argument names/defaults where users depend on them, but do not preserve Python implementation as the execution backend.

## Acceptance tests

Use synthetic public fixtures only:

- split/role disjointness; physical row-ID preservation; fit-only statistics; caps/subsample order and seed replay;
- missing/invalid bundle, wrong schema/source identity, outer TEST leakage, interrupted worker, timeout, duplicate cell, and partial-run resume refusal;
- exact per-cell predictions/metrics and serialized receipt parity against accepted legacy goldens for each task/level/role;
- aggregator consumes prior run IDs deterministically, verifies source/receipt identity, and reproduces RS00 exact summary bytes;
- CLI argument, refusal, output-path, and atomic-write coverage through `next_version_cli.rs` or a focused integration target.

Run only the focused offline tests using the existing admitted Cargo target/semaphore after capacity ACK. Terminal exit is required. No model/data execution campaign or full CI.

## Completion criteria

A Rust CLI invocation performs the entire production flow with no Python inference/evaluation/data-preparation/report process; summaries and receipts match legacy goldens; refusals preserve role/provenance rules; all focused tests pass. Document any intentionally unsupported protocol branch as BLOCKED rather than silently changing output.

## Stop conditions

Stop if exact current transform/evaluator/golden behavior cannot be established, if a Python callback would need to evaluate/query data, if role or native-provenance enforcement would weaken, or if source identity/receipt migration changes accepted history. Do not delete the old Python module or relax guards as part of migration; removal is a separate owner-approved cleanup after caller inventory and parity admission.

## Tracked migration acceptance IDs

- RS01-S1: Rust-owned FIT/role/coordinate preparation with exact identity. Current verdict: NOT_RUN.
- RS01-S2: qualified exact auditor identities or explicit unsupported blocker, no proxy rename. Current verdict: NOT_RUN.
- RS01-S3: bounded owned-child scheduling, partial/failed accounting and safe outputs. Current verdict: NOT_RUN.
- RS01-S4: complete source/model/recipe/executable identity closure. Current verdict: NOT_RUN.
- RS01-S5: native CLI and receipt/aggregation equivalence without Python evaluation. Current verdict: NOT_RUN.
- RS01-S6: complete mapped native coverage and actual admitted focused test receipts. Current verdict: NOT_RUN.

These migration IDs are separate from the original D00–D11 65-criterion baseline. New nontraining test/checker implementations must be Rust; a 'Python policy test' means a Rust check of Python source, not a new Python test implementation. Training gradient-query/autograd losses are training-coupled; standalone inference, benchmark validation, reports and release decisions are Rust.
