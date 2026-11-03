> Source-only classification evidence at a84. This is a partial function review, not whole-repository compliance or native qualification. Any Python callback must remain genuine DL training; standalone prediction/evaluation callbacks are outside the permitted boundary.

# DOPE Python boundary audit: training only

## Scope and evidence

Read-only source audit of DOPE at xbabe2 worktree `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`, branch `fast/next-version-dope-20261103`, exact HEAD `a84bba031240582921b00532b23cab6d56f150ac`. Worktree was clean at inspection. No dataset, model, evidence payload, build, test, or runtime execution was performed. This packet is a migration plan, not implementation or qualification evidence.

The latest user policy supersedes the prior v1 Python exception: Python is retained only for deep-learning training. Evaluation, inference, summary/Elo, reporting, qualification, and production data preparation must have Rust implementations at end state. Do not interpret existing Python source or historical CI packets as permission to retain those functions.

## Function-level classification

| Current Python path / functions | Classification | End-state disposition |
|---|---|---|
| `research/levels/v3_canonical_summary.py`: `_arr_mean`, `_compute_elo_and_paired_block`, `summarize_canonical_ledger_and_elo` | Metric aggregation, paired comparisons, bootstrap Elo, report assembly | Port to Rust. This is the narrow first slice. |
| `research/levels/v3_canonical_runner.py`: duplicate summary functions; `_load_v2_panel_scopes`, `_prepare_dataset_split`, `evaluate_dataset_across_tiers`, `evaluate_dataset_across_tiers_prospective`, `_worker_loop`, `_reap_worker_pgroups`, `_collect_source_sha256s`, `_build_receipt_common`, `run_bounded_canonical_eval`, `aggregate_canonical_runs`, `main` | Input/data preparation, model inference/evaluation, process scheduling, run/receipt writing and aggregation | Move the full production runner boundary to Rust. Keep only a thin training callback API in Python if it invokes actual DL training; do not leave an eval subprocess or Python report bridge. |
| `research/levels/v3_canonical_roster_eval.py`: `fit_and_score_canonical_auditor`, `evaluate_canonical_oracle`, `score_synthetic_against_oracle` and wire pack/unpack helpers | Auditor training/scoring, model evaluation, metrics, and wire transformation | Rust for production evaluation and wire processing. If a concrete auditor is a DL model, its optimizer/model-fit callback can remain Python; data selection, metric calculation, and output assembly remain Rust. |
| `research/levels/v3_ledger_replay.py`: `_compute_retention_stats`, `evaluate_multiclass_auditors`, `score_multiclass_syn_against_real`, `evaluate_reg_bin_dev_cell`, `_load_single_mc_cell`, `run_frozen_checkpoint_ledger_replay` | Evaluation, dataset preparation, metrics, and receipt generation | Rust. Preserve frozen ledger semantics and existing golden receipts. |
| `research/levels/v3_ablation_and_eval.py`: `_synthesize_report_from_train`, manifest and cell loaders, ablation masks, `run_dev_val_ablations_and_multiclass_eval` | Data preparation, evaluation, report synthesis | Rust. Synthesized report fallback remains forbidden where the current protocol requires native evidence. |
| `research/levels/train.py`: `make_optimizer`, `learning_rate`, `train` | Optimizer and shared neural model optimization are DL training; `prepared`, `validation`, and `append` are data/validation/output support | Keep only optimizer, schedule, model update and permitted checkpoint/resume training callbacks in Python. Move episode/input preparation, validation scoring and production record writing to Rust. Preserve caller API only for the training callback. |
| `research/levels/train_landmark_dropout.py`: `train_landmark_dropout_from_pool` | DL training | Keep only the optimization callback in Python. `_resolve_train_blob`, manifest/evidence readers, pool construction, feature/landmark conditioning and production data prep move to Rust; no weakening role/provenance enforcement. |
| `research/levels/v4_trainer.py` and `v4_training_helpers.py` | Batching, optimizer updates, finite-gradient checks, exact resume/checkpoint and prefetch are DL training support | Permitted Python training boundary. Keep the public training entrypoint/callback; do not add evaluation/reporting/data-prep behavior there. |
| Other production candidates such as `benchmark.py`, `comparison.py`, `transfer_evaluation.py`, `cli.py`, `dashboard.py`, `render_multilevel_evidence.py`, `strata_audit.py`, `rapid_validate.py`, `precalc/*`, `tools/optional_landmarks_qa/*` | From their names they appear to be evaluation, reporting, validation, CLI, or data preparation; names alone are not sufficient to assign every function | Inventory each public function before migration. Only actual DL model optimization may remain Python. Do not treat this audit as a review of their exact semantics. |

## Existing Rust seams

- `engine/dope-core/src/eval/mod.rs`: `evaluate`, `prediction_loss`, `EvalOptions`, `AuditorResult`, `EvalResult`, `schema_program`; generic auditors and retention metrics exist. These are not yet the v3 canonical ledger/Elo report contract.
- `engine/dope-core/src/eval/{gbdt.rs,linear.rs,nn.rs}`: existing auditor implementations. Reuse only where auditor behavior exactly matches frozen protocol.
- `engine/dope-core/src/dvs/{mod.rs,receipt.rs,accounting.rs,drivers.rs,selection.rs,permutation.rs}`: DVS cell/family accounting, privacy diagnostics, selection/drivers, serialized receipts, audit card. Reuse for matching DVS semantics; do not conflate these receipts with v3 canonical ledger schemas.
- `engine/dope-core/src/mlc_v3/*` and `engine/dope-cli/src/mlc_v3.rs`: Rust v3 model inference, wire sampling/inspection, encoding, weight checks. CLI currently exposes `FetchWeights`, `InspectWeights`, `InspectWire`, `Route`, `Sample`, and `Encode`.
- `engine/dope-cli/src/{levels.rs,dvs.rs,args.rs,main.rs}`: Rust CLI command registration and existing levels/DVS flows. Add the canonical replay/report command here after its core library API exists; avoid Python as the production CLI/orchestration bridge.
- Tests already present include `engine/dope-core/tests/seeded_auditors.rs`, `engine/dope-core/tests/dvs.rs`, `engine/dope-core/tests/mlc_v3_resident/mod.rs`, `engine/dope-cli/tests/next_version_cli.rs`, and `engine/dope-cli/tests/cli.rs`.

## Parity risks that must be closed

1. `v3_canonical_summary.py` uses `numpy.random.default_rng(20261024)` for dataset bootstrap draws and subsequent paired-difference bootstrap draws, and `numpy.percentile` for intervals. A Rust RNG with the same seed is not equivalent. Either implement and fixture-test NumPy PCG64/choice/percentile behavior or establish a frozen compatibility-vector protocol before switching producers. Do not claim parity from using the same seed.
2. Output ordering, missing comparator accounting, special v2 comparator exclusions, complete-case denominators, failure counts, and per-role versus overall Elo are part of the output contract. Lock JSON/golden fixtures from public synthetic rows before changing the writer.
3. Python's `round` behavior and NumPy's percentile interpolation need explicit matching. Keep exact field names, null-versus-omitted behavior, integer denominators and stable ordering.
4. Runner input prep currently uses dataset split manifests, CSVs, fit-only transforms, caps, and seeded subsampling. Porting only the reducer would leave prohibited Python evaluation/data prep active. Treat the reducer as an independently testable first packet, not as policy completion.
5. Evaluation helpers may train conventional auditors for evaluation; under this policy that is not the permitted DL-training exception. Rust must own the full production evaluation contract; a narrowly typed Python callback may only perform actual DL model training if needed, returning predictions/weights to Rust for evaluation.

## Ordered migration

1. Implement the Rust canonical ledger reducer and compatibility fixtures in `RS00-v3-summary-rust-port.md`; no production call-site switch until public fixture parity is exact.
2. Implement Rust canonical runner input/evaluation/receipt/aggregation orchestration in `RS01-v3-runner-orchestration-rust-port.md`. Keep the Python DL trainer as a callback only where necessary. Switch CLI and callers only once end-to-end tests cover public synthetic fixtures and frozen legacy goldens.
3. Audit all remaining `research/levels` functions using `RS02-python-dl-training-boundary.md`; migrate non-training Python production code by subsystem. CI must enforce the new policy without excluding legacy paths or reducing test coverage.

## Stop conditions

Stop a migration slice if required legacy golden outputs, exact RNG compatibility, expected role/receipt identities, or current CI tests are unavailable; report the missing proof instead of changing a fixture or relaxing its guard. Stop if a Rust API cannot preserve the frozen public JSON/receipt bytes or numerical protocol. No protected data, models, evidence rows, or science runs are needed for these source-level tasks.
