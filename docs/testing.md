# Testing, evidence, and budgets

`just deployment-check` verifies the separate locked `deploy/` CPU graph, typed
SafeTensors inventory, nonzero TVAE heads, legacy draw replay, explicit corrected
activation, allocation bounds and unavailable-device refusals. `just check` includes
this gate; the dedicated deployment CI lane runs it with a 20-minute cap. The graph
check refuses Python/libtorch bindings. GPU execution, arbitrary checkpoint export,
raw/mixed envelope export and downstream transfer remain separately unmeasured.
The active manuscript build (`just paper-tdl`) still checks that the authenticated
hero is Figure 1 on page 1; inspect rendered pages after visual changes.

The required ARM64 portability lane runs native aarch64 product computation:
all Rust library tests (including numerical goldens and historical decoders),
plus codec, CLI, target-last raw intake, artifact-only, bundle isolation and
frozen-geometry integration fixtures. It uses pinned Rust 1.88.0, the locked
default CPU graph and two build/test threads in the existing four-vCPU guest.
Its 20-minute job quota is unchanged. The real-entry readiness fixture checks
missing-tool and wrong-architecture refusals without pretending to execute ARM.
Paper, README and figure verification belongs to the x64 paper/CPU lanes;
image hashes or raster differences between x64 and ARM do not gate ARM compute.

`just fast` runs formatting, locked Rust unit tests, and V1 Python tests.
`just focus codec::tests` runs a reusable narrow Rust check and test filter
while editing; it never replaces the full CPU PR lane.
The CPU workflow pins sccache and preserves the lane's Cargo target cache to
shorten repeated locked checks. Local builds reuse their worktree target.
CPU guest setup installs Rust, Just, TeX, Poppler, and ripgrep in its existing
early package transaction, retaining the 300-second package-lock wait and
120-minute CPU job quota. `cpu.sh` checks all required tools before tests and never
starts a late package transaction. `test-cpu-tool-readiness.sh` exercises each
missing-tool refusal against the real CPU entry, before any build or replay.
`just check` adds Clippy with warnings denied, the Rust CLI fixtures, and the
raw-schema, raw-workflow, artifact-only, installer, byte-allocation, native
successor, public-demo, and frozen-geometry binding Rust integration suites.
Hosted CPU retains full
locked Cargo test discovery, which also executes every one of these suites.
Local check also executes `vnext_bundle` and `vnext_bundle_isolation`, including
real source-free workers linked to the current mixed rlib/cdylib Cargo artifact.
It never selects a stale hashed kernel library from a reused target directory.
Local check and hosted Cargo discovery also retain `vnext_interactions`, the
seven development fixtures for the recovered equal-term-tariff interaction
helper. Its 12/24/48/adaptive comparison charges term fields, not complete
encoded artifacts, and does not replace the production compiler's encoded-byte
allocation or qualify generalization.
Raw-schema tests cover explicit categorical field limits through intake and
source-free transform reload, including UTF-8 byte boundaries. Schema parsing,
validation, and metadata projection retain their 4096-byte defaults. Callers
using custom `InputLimits.max_field_bytes` can use the matching
`TableSchemaV1::*_with_field_limit` methods; table ingestion revalidates every
declared category against that limit before reading CSV bytes. Retained states
use their stored limit, with 4096 as the historical v1 fallback. The structural
JSON schema supplies the metadata envelope; native semantic validation applies
the chosen field byte limit and aggregate metadata cap.
Fast/check and hosted CPU verify the shipped native legal-input hashes before
product checks. The full installer suite covers license delivery, fixed-member
uninstall validation and legacy two-file compatibility. The Rust notice generator's full unit suite runs in local fast/check and hosted
Cargo discovery. Local fast/check and hosted CPU also verify real offline
regeneration against the shipped notice and pins, using cached locked crate
archives and pinned Rust legal inputs. GNU tar and gzip are required by the CPU
entry, with real missing-tool refusal fixtures before any build.
Local fast/check and hosted CPU run the six bounded public-demo regressions in
`research.benchmark.tests.test_public_demo`: frozen source admission, disjoint
physical groups, query-label independence, integer producer/reader compatibility,
safe native diagnostics, and honest failed-step/report output.
The actual native raw-schema, raw-workflow, artifact-only, and installer suites
remain auto-discovered by hosted CPU's existing locked Cargo test invocation.
The native guard's required fast/check and CPU routes share
`ops/ci/prepare-merge-guard.sh`: explicit locked full-graph metadata preparation
for both manifests, at most 120 seconds each with no retries. Setup may fetch
current locked public crates; the subsequent feature-boundary proof stays
offline and unfiltered. Metadata JSON lives under ignored target output, and
Cargo failures retain bounded stderr without exposing it in GitHub receipts.
Hosted CPU also runs the complete offline `research/transfer/tests/test_*.py`
discovery once, using its provisioned V1 interpreter after all fixture packages
are installed. Temporary files stay in `target/transfer-test-tmp`, and native
learner fixtures use one CPU thread. These fixtures preserve identity, coverage,
statistics and adapter contracts; they do not qualify official vendor weights
or a protected scientific campaign. Local fast/check retain their full discovery.
Local fast/check also run the bounded Rust include-closure fixture against the
actual fitter. Old and split module layouts must produce the same architecture
macros; missing, escaping, symlinked or conflicting constants fail closed.
Hosted paper-script discovery retains that fixture, and the CPU review-fix
entry includes the unknown-method and duplicate-identity receipt regressions.
`just security` scans secrets, dependencies, workflow syntax, and writes an
SPDX SBOM. `just score` regenerates Python boundary evidence, then runs the
full pinned Jankurai audit.
The audit workflow caches uv downloads in its disposable LAN guest. Audit
dependencies use the pinned uv installer and V1 version constraints; cache
restoration never replaces fresh boundary tests or changes the job quota.
`just fast-audit` writes a changed-source advisory under `target/jankurai/`;
its partial scope can produce boundary or release false positives and never
supersedes the clean full gate.

`just paper` regenerates every generated paper input, figure, README benchmark
block, and PDF from committed, SHA-pinned ledgers; `just paper-check` verifies
macros against those ledgers, figure hashes, the README block, and the build.
Both recipes enter `paper_build.py` once after their existing legacy preparation
and drift checks. The direct `build_pdf.sh` entry creates its own fresh admission.
The independent unit gate is `just paper-tests`: it retains the entire former
paper-script discovery suite, including the expensive committed-input replay,
and adds the bounded pipeline/recipe regressions. Full paper validation requires
both `just paper-tests` and `just paper-check`; moving the suite does not waive it.
CPU CI runs that same required named gate with its provisioned V1 interpreter:
`target/ci-python/bin/python -B -m unittest discover -s docs/whitepaper/scripts -p 'test_*.py'`
followed by
`target/ci-python/bin/python -B -m unittest discover -s research/benchmark/tests -p test_paper_build_pipeline.py -v`.
The later paper-number and paper-log unittests also use that provisioned V1
interpreter, preserving their NumPy-dependent claim-registry imports.
These replace the selective paper-script unittest calls in `ops/ci/cpu.sh`; the
independent committed-input replay remains exactly once in that suite.
The recipe fixture executes Just itself. CPU CI provisions the repository's
existing Just tool with the other tools before the test stage.
The real recipe regression lives in `test_paper_packaging.py`, so that required
paper-script discovery invokes it exactly once after CPU provisions Just.
The benchmark boundary's full discovery retains the bounded driver tests;
it does not require Just to run the paper recipe regression. Moving that
regression preserves its actual Just execution, reducer replay, old-path
rejection, and four fixture PDF checks. The focused invocation is
`PYTHONPATH=docs/whitepaper/scripts python3 -B -m unittest test_paper_packaging -v`.
`bash ops/ci/paper-clean.sh` is the CI form: it deletes `docs/whitepaper/generated`,
runs `just paper`, and requires a zero diff. `bash ops/ci/check-review-fixes.sh`
runs the review-fix and predeclare_v2 statistics, binder, and sealed-gate fixtures.
`bash scripts/ci-local.sh cpu|security|audit` reproduces the hosted lanes locally.

Admission captures the verified lock bytes without rereading them as a new pin.
The driver replays reducers and renders expected bytes from captured code and
inputs in one private snapshot child; different reducer contracts cannot share
a completion merely because they use the same panel. TeX, anonymization, the
architecture sidecar, and final paper checks consume a captured input tree under
the lane's reused `target/paper-build`. Source identities are checked around
children, including restored writes. This is trusted-driver control flow with
one writer per private tree, not an OS sandbox against a principal able to modify
that tree or substitute the interpreter, TeX installation, or loaded libraries.

Fresh `--check` compares review/auxiliary bytes without rewriting them, then
compiles all four PDFs and checks drift including the architecture outputs.
A local receipt may report `status=passed` with `cold_budget=unmeasured`.
CPU CI still has two independent full replays: the committed-input unit test
and its standalone authenticated PDF entry. Broader unit coverage and stronger
snapshot admission may affect that lane's cost. Canonical-driver fixture proof
does not qualify the full CPU or paper job under their respective 120-minute
and 60-minute quotas; setup, unit tests, compilation and drift need separately admitted measured
job proof. The historical cancellation remains failed, without unchanged retry.

The CPU PR lane is required. The scheduled GPU lane uses CUDA 12.8, libtorch
2.7, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; it checks repeated training,
resource measurement, and sealed corpus validation. A failed or absent GPU
lane blocks promotion and claims for its affected backend.

The bounded compiler tries at most two micro-TVAE profiles for L3 and three
compact profiles for L2 after a symbolic baseline. A caller can set a shorter
deadline or lower byte cap. Standard neural training is capped at ten minutes
per explicit candidate; GPU admission rejects an estimated footprint above
16 GiB. Fitness diagnostics use at most 128 rows and 32 dimensions per sample.
External learner smoke tests use fixed seeds and one thread per learner; they
do not change Rust certification.

## Cost and time budgets

The full CPU job has a finite 120-minute quota. Paper-clean, security and
full-audit each retain a 60-minute quota. The ARM64 portability job has a 20-minute quota. Disposable guest package installs
wait at most 300 seconds for an existing dpkg lock, within those job quotas, then
fail if it remains held.
The GPU compute spend cap is one scheduled 120-minute job per
week with no automatic retries. A single neural
fit is limited to 10 minutes and 16 GiB of estimated GPU memory. Abort a run
when its timer, memory quota, or sealed-corpus authorization fails; the job
must retain the partial log and mark the affected backend ineligible. The
operator kill switch is cancellation of the GitHub job or interruption of the
local `ops/ci/gpu.sh` process. Do not retry a paid GPU job without recording
the previous failure and an explicit repair in the campaign ledger.
The stop condition is any exceeded timer, memory cap, or missing sealed input.
`agent/cost-budget.toml` records separate `cpu_pr_minutes` and `paper_clean_minutes` allowances. CPU job 119248919772 exhausted its former hour near the final test groups; only that job allowance was increased. This does not change scientific fit/query budgets or renew a cumulative local admission. The manifest is the machine-readable budget; `just check` and the
CPU CI lane run `ops/ci/check-budgets.sh` to compare it with workflow timeouts,
the 600-second compiler cap, GPU memory admission, and required sealed inputs.

## Launch gate evidence

Security evidence is the CI secret scan, dependency advisory scan, workflow lint,
and SPDX SBOM. Backup evidence is the signed prior passing manifest and its
hash-verified bundle. Monitoring evidence is the campaign ledger, certification
JSON, and scheduled GPU job result. Rollback evidence is the restore procedure
in [release.md](release.md#rollback). Abuse controls are the strict input
boundary, byte limits, privacy tests, and allowlisted release inventory. A
production promotion needs all five evidence classes plus the exact source
commit, contract digest, and sealed holdout receipt; missing evidence blocks
promotion. GitHub Dependency Review is unavailable on this repository, so the
independent Cargo advisory scan and strict security wrapper provide the required
dependency gate. A PR can establish the code and fixture gates without claiming a
sealed-corpus release.

Test reports and repair receipts live under `target/`. A failing gate should
record a typed reason and the narrow rerun command. Stop a GPU experiment at
its deadline or when it exceeds its frozen memory budget. No campaign job
opens sealed holdout data outside certification. Do not rerun expensive
campaign matrices merely to repair a local source or fixture failure.

## Local repair

Command errors print a JSON `repair_receipt` with `purpose`, `reason`,
`common_fixes`, `docs_url`, and `repair_hint`. The fields describe a repair
without copying the input value into the receipt; the error line supplies the
row and column when input validation fails.
For data failures, use the reported row and column to fix numeric input;
avoid copying a source field into an issue or report. For an artifact failure,
compare its recorded digest and regenerate it from the same source commit.
For a backend failure, check the build feature and the scheduled GPU report.
Rerun the narrow failing command before repeating a whole campaign.

## Unified-spec extraction and compatibility proof

Keep `rust/` exports stable while extracting one coherent section at a time.
Compare its expanded source bytes before and after the move, record source and
contract hashes, then replay the existing deterministic fixtures. A move of
production code must preserve the complete implementation-hash source closure.
The included interaction fitting section is development code; its existing
eight integration fixtures cover fold complements, widths, seeded proposals,
permutations, substitutions and code-only refusals. Its tariff is not encoded
artifact size or an independent generalization estimate.

Run these commands from the checkout, reusing its `target/` directory:

```sh
rtk cargo test --locked --test vnext_interactions
rtk cargo test --locked --lib codec::
rtk cargo test --locked --test rust_cli
PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_frontier research.transfer.tests.test_learning_density -v
rtk just fast
rtk just check
```

The codec suite retains DPK2/DPK3 minor-version decode and round-trip checks;
the CLI suite also samples historical v1 JSON artifacts. The frontier tests
replay all shared frontier goldens through the historical module's canonical
`evaluate` alias, while the existing family-coverage diagnostics retain their
own denominator and censoring tests. Full `fast`/`check` discovery keeps both
paths exercised. Rust/Python TDL parity additionally requires both commands:

```sh
python3 -B -m research.transfer.tdl.golden --check
rtk cargo test --locked --lib metrics::golden
```

Candidate protocol admission uses the unfrozen
`contracts/tdl-contract-v2-candidate.json`, its protocol/result schemas and the
shared fixtures under `tests/fixtures/tdl-v2/`. Run the Rust candidate tests and
`python3 -B -m unittest research.transfer.tests.test_tdl_result_schemas -v` for
those changes. These proofs do not resolve null calibration or multiplicity
decisions, qualify a Python worker/model, or authorize a release. TypeScript
protocol types must be schema-generated and must not implement another reducer.
Inventory custody assets by hashes and Git identities; do not clean them as
part of an extraction. Record local proof separately from exact-head CI.

## Explicit workplan packet qualification

Transfer discovery runs every source/fixture check once. A clean checkout has no
canonical private workplan packet: the packet entry emits JSON with
`status=not_run`, `CLEAN_CHECKOUT_PACKET_ABSENT_NOT_QUALIFIED`, and
`release_authorized=false`. The P04 unit case asserts that explicit unqualified result and emits its report;
source/fixture successes do not qualify the ledger. Unit refusal fixtures
exercise missing, malformed, FIFO, symlink, changing and forged packet inputs.
There is no automatic fallback to another checkout, binary or README.

To require the actual admitted captured packet, explicitly set `DOPE_PACKET_DIR`
to its absolute path and run `PYTHONPATH=. python3 -B -m
research.transfer.workplan_packet --require-packet`, followed by
`PYTHONPATH=. python3 -B -m unittest
research.transfer.tests.test_workplan_acceptance.TestP04WorkplanDashboard -v`.
`DOPE_PACKET_MODE=qualify` or `DOPE_REQUIRE_PACKET=1` also requires a packet;
missing/incomplete inputs refuse with a nonzero status and a value-free reason.
`DOPE_PACKET_MODE=unit` explicitly runs source/fixture coverage without reading
a packet and cannot qualify one. It conflicts with `DOPE_REQUIRE_PACKET=1`.

Qualification pins the exact admitted task, progress and validator bytes, the
reported/provisional 61/65 snapshot, four blocked E03-E06 tasks and existing W01
binding. Input-pin changes require a deliberate reviewed transition; matching
counters alone cannot qualify substituted metadata or validator code. It does not establish model SDK
parity, scientific acceptance, protected access, custody or release readiness.
Actual qualification receipts must record the command, loaded module path/hash,
packet input hashes, executed source/tree/environment and budget. Keep the actual
packet and validator output private; do not commit or upload them to public CI.
The required `just check` and all five remote gates remain; no test fixture is a
substitute for the real packet or independent review.

## Confined numeric control worker

Transfer discovery retains `research.transfer.tests.test_control_boundary`. Its
mandatory setup builds the own-checkout Rust cdylib with `cargo build --locked
--lib`, then tests real dedicated Linux x86_64 confine/read/seal execution and
source/FD/network/exec denial, finite exact output and redacted failure paths.
The focused command is `PYTHONPATH=. python3 -B -m unittest
research.transfer.tests.test_control_boundary -v`. This is public synthetic source
proof only, separate from runtime deployment, full learner or scientific admission.

## Development pilot learner execution

Transfer discovery retains `test_pilot_learning_execution` and
`test_pilot_constant_null`. The focused command is
`PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_pilot_learning_execution research.transfer.tests.test_pilot_constant_null -v`.
It exercises the full learner coordinate roster, a separate native SDK mean-null
reference, fit-only inputs, query-label separation, physical operation reuse,
actual permutation calls, failure retention, private evidence and native RSS
units. Fixture adapters and resource claims do not qualify vendor weights, full
E03 execution, independent reproduction or confirmation. Development execution
requires the existing stage reservation and trusted exact plan/registry/null/
runtime/rights/resource admission; no implicit verifier, protected access or
automatic retry is supplied.

## Materialized development pilot draws

Run `PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_pilot_materialized_draws research.transfer.tests.test_pilot_runner research.transfer.tests.test_pilot_learning_execution research.transfer.tests.test_pilot_constant_null -v`. Version 2 pilot cells encode all native seeds as explicit 16-digit uint64 hex strings and bind a distinct `sample_4n` stream. The DOPE sample stage executes both native binary draws under one existing journal reservation and cumulative wall allowance, then passes the private pinned manifest to the learner worker. The compatibility native fixture produces both streams and stops when no learner worker is supplied; the trusted parent worker fixture below continues through actual fixture learner and driver calls. Version 1 decoding and safe-integer Pool identities remain unchanged. Control receipt verification includes its real confined ABI and complete closure byte charge; a version 2 control cell requires further typed runtime wiring and currently refuses. These source fixtures do not qualify the full learner/control roster, eight-stage E03 results, confirmation, custody or release.

The private materialized-draws v2 manifest retains the original pinned schema bytes. Receipt columns are compared in order with the Rust public-column projection of that schema, including opaque IDs, roles, dtypes and mask states; matching column counts alone refuse. Source names/domains remain private. The focused tests also exceed the SAMPLE allowance during the second draw and persistence, assert the retained outputs and terminal over-budget journal state, and prevent learner start or replay. Earlier unmerged v1 draw manifests without schema bytes refuse this new format; historical cell plans, product artifacts and receipts are unchanged.

The pilot v2 private draw API currently supports DOPE only. Decode, direct construction, pair storage and reload refuse both control methods with `pilot_v2_control_runtime_unavailable`: the actual confined-control receipt has no authenticated PublicColumn schema. An optional caller-invented schema field does not qualify that route. Typed control runtime/schema wiring remains a required next increment.

## Prospective physical learner operations

`PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_pilot_physical_operations research.transfer.tests.test_campaign_runner research.transfer.tests.test_pilot_learning_execution research.transfer.tests.test_pilot_runner -v` exercises actual fixture calls against a prospective physical DAG. The graph binds the exact campaign spec, cell, learning plan, registry and actual operation admission; the learner environment and input must match the campaign before reservation. The existing parent-held `StageJournal` suballocates its already charged learner-stage wall budget, fsyncs the graph before issuing a single-use in-process guard, and settles completed, failed and unexecuted physical nodes. Shared functional keys carry logical aliases without charging the wall reservation again. Reload retains an interrupted reservation and cannot issue another guard. No child reopens the parent's lock, and no separate quota ledger or automatic retry/refund exists.

Development `LearningExecution` and direct prediction refuse without a matching live reservation. The planner covers the full learner roster and, in streaming mode, the existing permutation routine's actual baseline and perturbation keys. Every Mitra prediction prospectively reserves eight child fits and a conservative 400-update envelope (eight times the pinned 50-step cap), including all driver calls. The previously observed 329 updates are not execution evidence for a later call. Requested native attempts are retained; uninstrumented completed fit/update observations remain null. The guard checks finite call membership and wall observations; the admitted outer executor and native runtime still enforce process/resource limits. This is trusted in-process Python control flow, not an OS isolation boundary or a subprocess-worker deployment.

The pilot learner argv now includes the private captured fit CSV, schema and both exact byte pins, separately from selection/query inputs. The external argv route alone does not provide a full learner/driver worker. The trusted parent capsule route below connects those libraries without transporting or reconstructing the live guard. Neither route alone qualifies typed controls, eight-stage scientific E03 execution, model rights, independent reproduction, custody, confirmation or release. Existing historical stage records remain readable; all failures and cumulative reservations remain charged.

After a newly interrupted physical reservation, the trusted parent may call `record_interrupted_operations` with an actual interruption receipt and a required verifier. The existing journal fsyncs an interruption disposition: every physical outcome is **unknown**, including operations whose private child records were never durably reconciled. It retains the full reservation and permits a terminal interrupted stage after reopen, including another crash between disposition and finish. It grants no guard, retry, refund or reconstruction of old work. Unknown effects require actual owner reconciliation before re-execution. This API concerns the newly created development operation journal; historical custody HOLDs and missing records remain untouched.

The physical guard is wired to the in-process learning API and the optional trusted parent `PilotWorkerRuntime` route. The compatibility external learner command does not consume this graph. The 400-update number is a prospective reservation, not a directly instrumented optimizer-update runtime limit. Native update observations remain null. Actual packaged-worker transport, child/update instrumentation and independently accepted resource/cost evidence remain required before claiming E03 physical execution is qualified.

## Trusted parent pilot worker

`PilotWorkerRuntime` connects `execute_pilot_cell(..., worker_runtime=worker)`
to the existing learner execution, metric reducers and streamed permutation
drivers. For this route the signed cell's `worker_binary` names a private
configuration capsule, rather than an external executable. Write
`worker.capsule()` before freezing/signing the cell; its SHA binds repository
modules, the full learner/null registry, original driver stream, physical wall
allocations and `WorkerAuthority`. The authority's source commitment is
`identity({"source_commit": spec.source_commit, "source_tree": spec.source_tree})`;
its input pin is the cell's input digest, which excludes the capsule and avoids
a signature cycle. The parent checks current rights/resource bindings before
native work. Required executor verifiers authenticate the supplied source, role,
owner and resource evidence; constructing hashes or using fixture callbacks
never grants development or scientific admission.

Private physical membership records bind the exact fit/selection/query CSVs and
reject overlapping units even when their row IDs differ. Contexts may supply
exact ordered row IDs or the `dope-pilot-private-context-indices-v1` format. The
latter pins ordered row indices before draw identities exist; it performs no
search or selection after the draw. Actual resulting IDs and targets are kept
only in private operation evidence. Numeric regression is the supported input
shape; missing or nonfinite values refuse without echoing source text.

The existing journal reserves learners before assembly/model entry, then
suballocates the complete physical learning and permutation DAG. Drivers run
while each fitted model is retained, without refits; their physical costs are
charged to learners. The subsequent driver stage commits the retained evidence.
Receipts distinguish parent lifetime RSS upper bounds from unmeasured child or
GPU peaks. Native failures retain attempts, partial predictions and not-run
slots; a response or count never grants statistical or independent acceptance.
Capsule replay, missing role admission and insufficient parent allowance refuse.

Run `PYTHONPATH=. python3 -B -m unittest
research.transfer.tests.test_pilot_worker_runtime
research.transfer.tests.test_pilot_worker_inputs
research.transfer.tests.test_driver_seed_uint64 -v` (on one command line). The
end-to-end fixture executes the real Rust compile/reload/isolated n/4n sampler
and fixture learners across all sixty paths. It does not run vendor weights,
qualify either control producer, open a sealed holdout, or complete E03–E06.

Fixture execution keeps the development-runtime prerequisite and reports
`fixture_validation_complete` separately. Unexpected callback/runtime exceptions
produce static failed-stage records when the journal can durably settle. A failed
physical settlement/fsync remains pending and charged for actual owner disposition;
it cannot be disguised as a finished failure. Physical units may repeat within
one partition (repeated measurements); they may never cross fit/selection/query
roles. Row counts do not establish independent sample counts or statistical power.

The trusted parent pilot worker treats contract/value, shape/type/lookup, arithmetic,
callback/runtime and I/O failures as a single failed stage. The parent journal
records that terminal state only after physical operations have settled; failure
handling performs no auxiliary file write. Unclassified exceptions and interrupts
leave the original reservation pending for owner diagnosis. They do not receive a
retry, budget refund or successful fallback. The runtime fixtures exercise both
publication I/O failure and this unresolved-reservation behavior.

The signed parent-worker authority declares a canonical fulluint64 learner seed
independently of the cell compiler fit seed. Both streams remain traceable in
private assembly evidence. Development mode authenticates the existing owner
learner stream and fixed recipe; matching an arbitrary seed hash is insufficient.

## Combined typed-control and functional-driver successor fixtures

The combined 17-file control/driver source candidate adds the authenticated
`PilotControlRuntime` route beside the bare DOPE materialized-draw API described
above. It connects actual framed copula/compressed-row compile/reload and confined
n/4n draws to the existing learner/driver journal, with exact schema/order and
separate CSV/aggregate-input witnesses, distinct compiler/learner uint64 streams,
and the full artifact/runtime/invocation byte cap. A real cap negative must stop
before sampling or learning. Functional driver reuse retains every metric while
sharing only identical fit/query/predict-seed keys under the reserved DriverPlan;
each new Mitra prediction still reserves eight child fits and 400 updates.

Focused modules are `research.transfer.tests.test_pilot_control_materialized_draws`,
`research.transfer.tests.test_pilot_control_runtime`,
`research.transfer.tests.test_pilot_materialized_draws`,
`research.transfer.tests.test_pilot_runner`,
`research.transfer.tests.test_pilot_worker_runtime`,
`research.transfer.tests.test_pilot_worker_inputs`,
`research.transfer.tests.test_pilot_physical_operations`,
`research.transfer.tests.test_pilot_driver_prediction_aliases`, and
`research.transfer.tests.test_control_boundary`. Run them with `python3 -B -m
unittest` under the admitted local validation scope. A source-only export harness
may substitute an explicitly recorded, hash-verified prebuilt-library setup for
the boundary fixture's Cargo build; it must record actual module paths/hashes,
base commit/tree, candidate patch, interpreter/toolchain and native artifact pins.
Such an export is generated test input, has no Git identity, and is not execution
on the base commit or receiving main. Preserve private fixture provenance,
original failures, finite resource admission and one-shot output paths. Authored
source, AST checks, prior disjoint prototype results and fixture passes are not
full local gates, native SDK qualification, E03 completion or scientific evidence.

The native qualification runtime verifies the installed distribution's unique top-level
`.dist-info/RECORD`, ignoring vendored records and refusing ambiguity. Its pinned
component labels distinguish NVIDIA code (Apache-2.0) from Kumo weights
(OpenMDW-1.1); complete pinned license documents and rights admission remain
separate checks. The helper reports its actual loaded file. Focused protocol and
metadata regressions run with `PYTHONPATH=. python3 -B -m unittest
research.transfer.tests.test_native_qualification -v`; these tests run no SDK models.

Private pool-context v2 freezes a training-only rule before generation. Standard
uses the first requested rows of each pool; stratified_anchor selects equally
spaced training-target ranks with original input-index tie order in that same
pool. Real-data indices are never copied onto synthetic rows. Signed context
bytes and actual selected row/position hashes remain bound to the captured draws;
query and selection labels are absent from the selector. Legacy explicit row-ID
and index contexts retain their behavior.

Standalone control ABI2 qualification (Root candidate; not yet qualified)

Linux x86_64 requires the actual standalone binary and build receipt. Missing
binary/receipt is an error, not a skipped positive test. Other architectures
are explicitly unsupported for this route; the product v1 route remains intact.
Root must admit finite CPU/RAM/disk/wall resources before these commands:

```sh
python3 -B research/transfer/native_control_build.py --out target/native-control-build
python3 -B research/transfer/native_control_manifest.py --binary target/x86_64-unknown-linux-gnu/release/dope-native-control-runtime --build-receipt target/native-control-build/RESULT.json --out target/native-control-runtime.json
python3 -B -m unittest research.transfer.tests.test_native_control -v
```

The output directory is exclusive; reusing it refuses rather than resetting an
attempt. Reuse the lane's target directory for compilation. The typed-parent
fixtures also need the existing product kernel fixture; their existing setup
may build that kernel if missing. Account for this in Root's admitted validation.
Tests require exact f64 bytes against the existing literal sampler, including
copula libm rounding, and actual child denials. No tolerance relaxation or fake
binary is allowed. Synthetic fixture ceilings are not the original pilot caps.
Every actual draw pays full ELF + artifact + canonical request + 104-byte wire.
A measured failure at 4096 or 32768 is retained as a cost negative. No size,
parity, confinement or scientific success is established by parsing this source.

Mandatory target-last intake and consumers follow `docs/data-contract.md`. Run
`python3 -B -m unittest research.transfer.tests.test_target_last research.transfer.tests.test_pilot_worker_inputs -v`
for normalization, declared-target disagreement, strict consumer refusals, query-label
exclusion, unchanged source bytes and row identity. Native intake also runs
`cargo test --locked --lib typed_data::tests`; historical decoding stays unchanged.

Linux x86_64 local fast/check and hosted CPU build the standalone native control
runtime with `python3 -B research/transfer/native_control_prepare.py` before transfer discovery.
The build is offline and locked, preserves per-attempt source/build evidence, and
publishes a verified manifest. Missing or stale binaries fail; this requires the
current GNU libc 2.39 static math archive and admitted FMA/AVX features. Other
platforms explicitly mark this native route unsupported while retaining their
required product portability gates. The complete runtime, artifact and invocation
are charged at each control draw.


### Funded adapter lifetime fixtures

`python3 -B -m unittest research.transfer.tests.test_funded_adapter_lifetime -v`
reproduces the old cumulative-timeout failure with synthetic clocks. It also
checks actual fixture calls through the streamed physical DAG. These tests do
not execute or qualify official model weights or a protected campaign.

The default V1 adapter/graph/worker vocabulary remains unchanged. A trusted
parent may explicitly select `adapter_lifetime_policy="funded_per_fit_v1"` in
`PilotWorkerRuntime`; that choice is bound into its V2 configuration capsule and
the V2 operation graph before the existing StageJournal reserves the graph.
The per-prepared-model lifetime is the sum of that fit's already funded unique
prepare/predict nodes. Logical aliases add no calls or budget. One actual adapter
binds to each fit, and each physical node permits one backend attempt. Failures
retain elapsed cost. Reprepare, node replay, a closed or expired guard, mutated
funding, and an unreserved call refuse; none issues a fresh lifetime or refund.

`ModelIdentity`, `ResourceLimits`, and `AdapterAdmission` retain their V1 hashes.
In particular, Mitra's native `time_limit` remains the qualified identity's
600-second setting; the new lifetime does not change its training recipe. Every
node must still fund at least the native call limit, and the original parent
stage/job deadline and external process containment remain mandatory. This
trusted in-process graph is not isolation against arbitrary same-user code.
Fixture validation does not authorize a full pilot, sealed access, or release.

In V1, `ResourceLimits.wall_seconds` is the cumulative adapter cap. In the
explicit `funded_per_fit_v1` route it is the unchanged **native per-call cap**;
it must not be read alone as the prepared model's lifetime. Read the V2 adapter
receipt's `native_call_wall_seconds` together with `physical_budget.wall_ms`,
its graph/reservation hashes and operation keys. This semantic distinction is
part of the signed V2 capsule/graph opt-in. The standalone V1 route stays V1.

The trusted parent calls a guard sequentially. A guard and its adapter are not
an API for concurrent callers. Parallel cells must use separate processes,
private output directories and separately reserved journals; they may not share
or reconstruct one live guard. Internal SDK threads do not issue parent calls.

## Ordinary engineering PR review

The owner may implement, review the final diff, and merge an ordinary engineering PR using the same agent. External approval and last-push second-person approval are optional under the primary human decision of 21 October 2026. All five strict GitHub Actions checks, conversation resolution and normal PR merge remain required; a self-review must never be fabricated as another identity. Verify actual receiving-main checks separately. This changes engineering review policy only: independent scientific acceptance, sealed-custodian separation, rights and custody evidence retain their own requirements.
