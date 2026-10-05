# ARF fit and sample preparation for the matched S3 panel

This prepares an additional native baseline. It does not admit the full campaign
or add a real dataset result. Official tests remain evaluator-only. The adapter
uses the existing training-derived common-numeric worker projection and a CPU
runtime; it never requests a GPU.

## Author implementation and objective

The MIT author source is `bips-hb/arfpy` commit
`8b63c1b3999981125b4af2828ff52cba8e29169d`. Its source archive hash is
`0c9012778cb5ffe0a006fc32b33ae4ed3f85689c6742253d95f91c10c42bd2aa`.
The immutable source stays on scratch. Its ARF, FORDE and FORGE code is used
without modification. The first three-column probe used the study `arf_native.py` density evaluator
at hash `2e5f6373ca3b36019d82038712f16dce976d9f783f7dbc5444aec96b6c5e2f18`.

Select ARF configurations by training-derived validation mean log density of the
author's FORDE mixture, maximizing with the already frozen artifact-byte then
configuration-digest tie breaks. Values from this native objective are never
ranked against another method's native objective. Common retention or DOPE's KPI
cannot select an ARF configuration.

The central four original and four refinement configurations remain unchanged;
the combined per-cell cap is eight trials and twelve total hours, including
failed and timed-out attempts. Each new fit receives a deadline of at most
600 seconds and an outer hard timeout. Preserve the author default separately
from the native-selected configuration. The new entry points do not replace
historical pilot sources, manifests or receipts.

## Runtime and artifact contract

`arf_runtime_guard.py` checks a pinned version-2 preparation manifest before any
third-party import. Every declared interpreter, package, source and bytecode file
is hashed; exact directory inventories reject added namespaces and aliases.
The executing guard, adapter and native evaluator must also be those frozen
source files. It requires an isolated no-site interpreter, disabled bytecode writes, an empty
private cache, the frozen Python import chain and an empty CUDA visibility
environment. The launcher must rehash the interpreter and import roots before
invoking that interpreter. This is a declared import-provider contract, not
certification of the complete system dynamic-library closure.

`arf_adapter.py` verifies input hashes and worker partition names before fitting.
It rejects workers that contain `test.csv`, incompatible widths, malformed or
nonfinite numeric data, noncanonical seeds and excessive fit deadlines. Training
and validation are consumed from the exact verified buffers. Binary-category
typing is fitted from training only. Random forests use the frozen sixteen CPU
threads; FORDE uses truncated normals, `oob=False`, and the selected alpha.

The restricted artifact contains FORDE factors plus projection and adapter
metadata. Charge all six files. It stores neither training rows nor the fitted
classifier. Sampling verifies the complete artifact before imports and consumes
the verified factor buffers through the original FORGE class. Samples are clipped
to the common unit-interval projection. Runtime and artifact hashes and the
deadline are checked again before success is returned. Resource admission and
trial/job identity accounting remain the outer coordinator's responsibility.

## Measured fixture verification

The private preparation root is
`/mnt/fast-scratch/dope-benchmark/arf-s3-population-contract-preparation-v3`.
It reuses hash-verified owned providers for numpy 1.26.4, pandas 2.1.4,
scipy 1.16.3 and scikit-learn 1.7.2. No dependencies were installed for this probe.

The deterministic synthetic fixture has 128 training and 64 validation rows,
three numeric columns, fit seed 11, and one already frozen refinement
configuration (10 trees, minimum node size 20, one iteration, alpha 0.5).
Three sample seeds at 128 and 512 rows each have two exact repetitions.
All six cells match samples drawn directly from the fitted author's FORGE
implementation before factor serialization. The artifact charge is 11,756 bytes;
this fixture makes no L3 or quality claim. The successful bounded operation took
82.970 seconds including outer verification; the inner contract took 78.124
seconds. This is separately reported adapter research cost.

The first launch failed before dependency imports because its bootstrap lacked
`__file__`; the next admission wrapper rejected a reused output filename before
starting the probe. Both failures remain on scratch. The third wrapper used a
new reservation and an exact verified bootstrap buffer. A predecessor version had already passed one fixture fit before the executing
module binding was tightened (77.009 seconds including verification). Both
successes remain recorded: two actual fixture fits across source versions,
159.979 successful operation seconds, zero real dataset fits, and no official
test access. Failed launch overhead is retained separately in its receipts.

Immutable anchors:

| Evidence | SHA-256 |
| --- | --- |
| Runtime preparation | `746cf74907879f6bf1f25d799ea5123670a2292b16b993436c7d36d81192f2d2` |
| Fixture contract | `57caa2b52e5a8de31e921c92d21f75d03b7d25e833308994a1c9b0f482dbbb90` |
| Fixture worker | `ad8de9a9f01f09d9cf1237b86c1955697e31c0cd3104d99cbb357855dc8a0315` |
| Successful private log | `d32ff3da74e68842ee377a12a154f72c0aed9e882fbe5790640a98e92e9d8eb4` |

Run the hermetic pre-initialization controls with:

```sh
python3 -B -m unittest research.benchmark.tests.test_arf_contract -v
```

Before a real S3 round, freeze source/runtime/worker/grid/job/sample identities
and a capacity registry recognized by every overlapping queue. Preserve the
healthy DOPE refinement queue, admit only quiet xbabe1/2/3 CPU slots, keep the
200,000,000,000-byte scratch ceiling, and account for every failure and timeout.
No SDV rerun is authorized by this preparation. MFS-v2, PTF-v1, release and
superiority claims remain null.

## Next-round request and native selection boundary

`arf_fit_request.run` prepares the next research dispatcher boundary. A frozen
round and externally pinned request digest are required. Requests match frozen
canonical JSON job digests, so seed `11.0` cannot substitute for seed `11`.
CPU environment is checked before loading any source provider. The worker binds
adapter/native/guard sources and exact five-file worker partitions, rejects
unsupported projected widths, and writes only under its canonical job attempt.
The 600-second request and round deadlines are checked again after final rehashes.

`arf_native_selection.select` accepts one accounted receipt per frozen trial,
with at most eight trials and 12 hours of charged method–dataset time. It selects
only held-out FORDE mean log density, with artifact bytes and configuration digest
as the frozen tie breaks. Scheduling cutoffs stay distinct from timeouts,
infrastructure interruptions and method errors. A cutoff marks the search
incomplete; it cannot establish the best configuration over the whole grid.
Native values are never ranked against another method's native values.

These primitives do not freeze a round, verify a receipt-lock, grant capacity,
spawn a process, fold retries, sample a model, or open final evaluation. The outer
dispatcher must verify interpreter/source/runtime/transport before invocation,
reserve fresh aggregate capacity, enforce a hard process timeout and cell budget,
and preserve every attempt and hash-bound receipt before selecting or publishing.
No real S3 fit is admitted by this source preparation. The original 100-lineage proposal remains unfrozen; its 24 width-incompatible
planned cells retain their historical adapter-scope status, with no DOPE win.
The new single-column contract below can support a separately frozen successor
proposal; it does not rewrite those earlier cells or admit real fits. Official tests and all gated scores remain sealed/null.

## Categorical-only and single-column contract repair

The original author can fit a nonempty single column. Its FORDE implementation
returns an empty continuous-parameter DataFrame without column headers when all
variables are categorical. The study density evaluator previously indexed that
empty table as though continuous parameters were present. It now evaluates the
existing categorical probability factors in that case and rejects absent
continuous factors when a continuous variable exists. The density formula,
zero-probability floor, native direction, search grid and tie breaks are unchanged.
The numeric adapter now accepts one or more columns in both fit and sample.
The author code, runtime guard, providers and historical receipts are unchanged.

A separately frozen synthetic v2 operation declares four fits: original author
and guarded adapter on one continuous and one binary categorical column. Each
case uses 128 training and 64 validation rows, the same configuration described
above, fit seed 11, three sample seeds, and samples of 128 and 512 rows. All four
fits succeeded. Serialized author factor files and held-out density values match
exactly between direct and guarded fits. All 12 sample cells match direct FORGE,
two factor replays, and the guarded sample entry point byte for byte.

The operation took 91.252959 seconds with a 600-second
outer hard limit, fresh sixteen-core admission on xbabe2, and measured peak
resident memory 261275648 bytes. Artifact charges include all six
files; continuous and categorical fixtures charged 4402 and 1648 bytes respectively.
These are synthetic API checks, not real dataset quality or L3 certification.
Complete system dynamic-library closure and a real S3 round remain unadmitted.

Failures are retained: the original univariate v1 operation started two fits and
failed in the study density evaluator after 10.987937 seconds. Its continuous
case had already matched original FORGE; the categorical failure is not author
unavailability. The first v2 launcher selected an old helper path and stopped
before dependencies or fits (4.140890 seconds). The corrected launcher has a
separate attempt receipt and does not replace either failed receipt.

All private evidence stays under
`/mnt/fast-scratch/dope-benchmark/arf-univariate-author-contract-v2`.

| Evidence | SHA-256 |
| --- | --- |
| New runtime preparation | `c56f26732be1112d051b96d87c004898b7218b4c6b514b45964877db4c203a2a` |
| Synthetic contract | `538626e6d38a6c89b147a84dc1a5790b98565160b455906be4e5fc8a12073ea1` |
| Frozen operation source inventory | `130c65751764b28f5a46eafe6a039514542e3fb826772a2c504a66281a689f0f` |
| Successful four-fit proof | `d41d9268231281ff9bc545342029cfaa95922441f35e2478b64795b4f6709364` |
| Corrected outer actual exit | `5f35e032c61f980dd274cc1da10cfb379ea1d7cd9ebfd819a348e125d9983466` |

Focused analytic controls cover the one-variable categorical mass, the product
of categorical masses, absent continuous factors, continuous and categorical
one-column preprocessing, and empty-column rejection. Run them alongside the
pre-initialization contract controls:

```sh
python3 -B -m unittest research.benchmark.tests.test_arf_native research.benchmark.tests.test_arf_contract -v
```

A real successor round still requires new source/runtime/job/worker identities,
fresh aggregate capacity and its own attempt ledger. No real S3 or official test
rows were used; no SDV rerun was launched. All gated scores remain null.
