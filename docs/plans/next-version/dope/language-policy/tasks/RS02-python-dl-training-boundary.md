> Current authority: latest user requires Python ONLY for deep-learning training. Read ../README.md and the migration manifest first. This is a policy task, distinct from lower-case remote lane IDs. Proposed final repository destinations below do not authorize current authors to edit repository/shared roots; each author writes only its external packet/reserved namespace. No native execution, caller switch or qualification is implied.

# RS02: Enforce the Python-only-DL-training boundary

## Objective

Inventory `research/levels` production Python functions and reduce the allowed Python surface to deep-learning model optimization, optimizer/scheduler, finite-gradient/update logic, and exact training checkpoint/resume. All other production workflows—including data preparation, inference, evaluation, Elo/summary, reporting, qualification, and CLI orchestration—must be Rust.

## Read first

1. `AGENTS.md`, `docs/plans/next-version/dope/STATUS.md`, and current DOPE task/owner/test maps
2. `research/levels/{train.py,train_landmark_dropout.py,v4_trainer.py,v4_training_helpers.py}`
3. Their imported modules: `data.py`, `validation.py`, `checkpoints.py`, `contracts.py`, `losses.py`, `model.py`, `observed_conditioning.py`
4. Remaining production Python modules and all Rust APIs they call
5. Existing Python tests only to map behavior; do not delete or narrow them until Rust tests cover replacement behavior

## Required function inventory and boundary

- Preserve a stable, typed Python training callback for actual DL optimizer/model updates. Current likely eligible functions: `train.train` update loop and checkpoint interactions, `train_landmark_dropout.train_landmark_dropout_from_pool` update loop, and `V4Trainer` update/checkpoint/prefetch code.
- Split out and migrate all nontraining pieces: corpus/manifest/evidence/blob readers, role/split selection, episode/input generation, transforms/feature and landmark conditioning, validation scoring, prediction/inference, report/ledger writing, run scheduling, CLI and aggregation.
- Do not classify classical estimator fitting used by an evaluation benchmark as permitted Python DL training. It belongs in Rust evaluation.
- Keep Python training callback inputs authenticated and role-qualified by Rust; callback outputs are state/metrics limited to training state. Rust validates checkpoint identities and owns all evaluation and release/report values.
- Preserve exact checkpoint format/identity, RNG, optimizer, cursor, rollback and deterministic behavior. Do not change model parameter names or pickle/module identities without explicit compatibility proof.

## Edit ownership

Before implementation, owner maps each module and Rust target to one worker. No concurrent editing of `research/levels/v4_trainer.py`, extracted helpers, model/objective files, CI guard/exception lists, or historical receipts. Do not widen Python exception lists or exclude legacy paths to make the policy check pass. User policy supersedes the historical v1 exception, but old records and source guards remain immutable evidence.

## Acceptance criteria

- A checked-in allowlist is function-based or tightly path/function scoped and demonstrates only the DL training entrypoint/dependencies remain Python. No broad `research/levels/**` exemption.
- Static policy test fails when a production Python inference/evaluation/report/CLI/data-prep function is added or when nontraining code enters a training helper.
- Rust owns a direct API/CLI for every migrated production function; no `subprocess python`, Python RPC, or opaque production bridge for nontraining tasks.
- Every removed Python production responsibility has Rust focused tests covering synthetic inputs, refusal behavior, deterministic values and preexisting goldens.
- Training tests prove exact CPU resume/update behavior and identity/role validation; CUDA claims require separate owner-approved capacity and hardware evidence and are out of scope here.
- Legacy Python goldens/receipts remain byte-for-byte unchanged; no fixture updates to authorize migration drift.

## Validation and handoff

Use bounded focused Rust tests and source-level Python policy tests only after the owner authorizes implementation and capacity. Do not run protected-data/model workflows. Report exact head, changed paths, old/new hashes, focused test commands and terminal outcomes, static allowance inventory, and explicit NOT_RUN areas. Any unported evaluation/data-prep code means boundary completion is NOT_DONE.

## Stop conditions

Stop before changing a function if it mixes training and evaluation but no stable split boundary is defined; if an identity, role, checksum, checkpoint or receipt contract would change; if only an exception-list edit can satisfy policy; or if new Rust parity tests lack existing golden inputs. Ask the Rust module owner for the smallest typed seam and mark remaining Python scope unqualified.

## Tracked migration acceptance IDs

- RS02-S1: complete function review of 103 research/levels files; name-only labels are insufficient. Current verdict: NOT_RUN.
- RS02-S2: training-only callback preserving gradients/optimizer/RNG/checkpoint identities. Current verdict: NOT_RUN.
- RS02-S3: Rust-written policy tests/checker, no broad Python exception. Current verdict: NOT_RUN.
- RS02-S4: all nontraining inference/evaluation/preparation/report routes migrated. Current verdict: NOT_RUN.
- RS02-S5: legacy tests preserved and native coverage proven before retirement. Current verdict: NOT_RUN.
- RS02-S6: overall NOT_DONE while active nontraining Python or required coverage/backend gap remains. Current verdict: NOT_RUN.

These migration IDs are separate from the original D00–D11 65-criterion baseline. New nontraining test/checker implementations must be Rust; a 'Python policy test' means a Rust check of Python source, not a new Python test implementation. Training gradient-query/autograd losses are training-coupled; standalone inference, benchmark validation, reports and release decisions are Rust.
