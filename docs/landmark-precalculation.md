# Reusable private landmark measurements

`research/landmark_cache/` schedules the merged Rust `landmarks-native-v1` panel
over the pinned 3,859-entry JopeDime catalog. It prepares data for the separate
multilevel training work. It does not train a shared encoder or change the native
panel, product defaults, frozen evidence, source rights or certification.

Only each entry's declared official TRAIN object is fetched. The orchestrator
never requests its official TEST or mixed BeyondArena Parquet. A source target
is resolved from the catalog's explicit declaration and placed last in a new
private CSV. Original source bytes and physical row order are preserved. The
native panel performs its existing fold-local preprocessing. No Python learner
or alternate landmark formula is introduced.

The catalog contains unrecorded source rights and unresolved source families.
These states are copied without upgrading them. Private owner-requested
precalculation is **not** admission of those sources to a training or evaluation
campaign. The consuming training PR must resolve family identity, deduplication,
rights and meta-training/development/final membership before consuming labels.
Alternative targets and related sources must not be split across those roles.
Do not interpret native within-TRAIN CV as source-family-held-out performance.

The separate 142-entry BeyondArena inventory has mixed official splits. Admit
dedicated authenticated TRAIN exports through the existing trusted-export path
before adding a new input adapter. This cache does not bypass that boundary.

## Frozen schedule and cache identity

The catalog digest is the existing `fetch_jope.CATALOG_SHA256`. Every entry gets
a scheduled record, including unsupported tasks and insufficient inputs. A cell
key binds the catalog entry, TRAIN SHA/bytes, explicit target, task/family/rights
metadata, native executable SHA, native source commit, preparation source hashes,
and **all** landmark options. Linux and macOS executables have different keys;
cross-platform numerical equivalence is not assumed. Partitioning uses catalog
dataset order, so hosts with different executables still receive disjoint datasets.

The default precalculation profile uses five folds, 8,192 ridge rows, 4,096 small
learner rows, 2,048 kNN references, 64 feature spectra, seed 0 and 20 billion
estimated work units. It is a bounded native recipe, not the proposed future
exact `landmarks-v1`. Reports retain all 23 slots and their complete,
uninformative, unsupported, failed or over-budget dispositions. A successfully
stored report does not imply that all its slots succeeded or were informative.
Native row caps and estimated-work pruning remain visible in each report.

```sh
python3 -B -m research.landmark_cache plan \
  --catalog /private/catalog.jsonl --bucket EXPLICIT_SOURCE_BUCKET \
  --engine-sha256 ACTUAL_BINARY_SHA256 --source-commit ACTUAL_NATIVE_COMMIT \
  --work-budget 2000000000000 \
  --out /private/new-plan.json

python3 -B -m research.landmark_cache run \
  --plan /private/new-plan.json --binary /absolute/dope \
  --out /private/new-run-host1 --profile EXPLICIT_PROFILE \
  --output-bucket EXPLICIT_RESULT_BUCKET --prefix derived/dope/landmarks/RUN_ID \
  --shard 0 --shards 3 --workers 8 --threads 2 --memory-gib 3 \
  --timeout 600 --wall-seconds 43200
```

Choose host allocations from fresh load/RAM/disk measurements. Reserve RAM for
existing users as well as every simultaneous worker. Native processes run at
nice 10 with bounded CPU time, wall time and Rayon threads. Linux enforces a
finite address-space limit. macOS rejects that limit, so its parent instead
samples the native PID's RSS every 250 ms and kills its process group above the
allocation; observation failure also stops it. This sampled resident-memory
watchdog is not an instantaneous address-space ceiling. Receipts distinguish
the policies and report a sampled peak, or null when no observation was made. AWS
credentials are absent from the native child's environment. No GPU is required.
The orchestration parent is not a hostile-code sandbox or sealed evaluator.

The example selects the higher-coverage two-trillion-unit profile. A real pilot
on three heavily pruned tables removed their budget-omitted cells within the
same row caps; native runs took about 20–99 seconds. This is a different complete
recipe identity from the default 20-billion-unit discovery profile. Do not mix
the two option sets in one coordinate space. Both retain unsupported cases;
larger budgets cannot remove the native 512-encoded-column shape limit.

## Publication and interruption

S3 objects use explicit server-side encryption, no public ACL, conditional
create and SHA-256 checksums. The native private report is written first at
`objects/sha256/HASH.json`; its bound `cells/CELL_KEY/manifest.json` is published
last. Reuse downloads and verifies both binding and report bytes. A conflicting
immutable object or failed transport refuses; neither is silently overwritten.
No raw CSV, header, target value, per-row loss or native error text is uploaded
as a public sidecar. Full reports are private measurements, not public embeddings.
Keep their S3 prefix under the owner's existing private access controls.

Each local attempt has a private receipt before native work, including actual
PID, host and resource limits. Terminal attempts and partial downloads remain
on disk. Ordinary resume reuses verified completed cache entries; it does not
silently retry failed, limited or interrupted jobs. Reconcile any still-running
PID and preserve the original attempt before supplying `--repair-reason` for an
actual diagnosed repair. An interrupted `.partial` source download must be
preserved/renamed explicitly before a replacement download; no broad cleanup.
Filesystem publication uses private permissions; it is not independent custody.

Hosts without S3 credentials can use `--offline-sources /private/train-blobs`
and `--local-store /private/cache` instead of S3 arguments. The blob directory
contains only scheduled TRAIN files named `TRAIN_SHA256.csv`. The same hashes,
roles, conditional object writes and manifest-last rules apply. A courier copies
only this cache back to an existing credentialed host, then runs:

```sh
python3 -B -m research.landmark_cache publish-local \
  --plan /private/host-plan.json --store /private/copied-cache \
  --profile EXPLICIT_PROFILE --output-bucket EXPLICIT_RESULT_BUCKET \
  --prefix derived/dope/landmarks/RUN_ID
```

This verifies every report against its original plan before publishing. It
shares no credentials with worker hosts and never requests another source role.

## Coverage dashboard and verification

Copy each host's `progress.json` to an owned local output directory and render:

```sh
python3 -B -m research.landmark_cache dashboard \
  --progress target/landmarks/host1.json target/landmarks/host3.json \
  --out target/landmarks/dashboard
python3 -m http.server 8765 --bind 127.0.0.1 --directory target/landmarks/dashboard
```

For live remote snapshots, pass explicit `--remote HOST:/absolute/progress.json`
values and `--interval 15` to `dashboard` instead of local progress files. The
collector uses existing SSH access, keeps the last complete copy if a pull
fails, and copies no source rows or report measurements. Host timestamps retain
the actual last worker update. No credentials are distributed to offline workers.

Refresh the snapshot while workers run. The page polls its aggregate JSON every
15 seconds and shows scheduled/terminal coverage, reusable reports, unsupported
inputs, failures, per-host allocations and native slot states. It does not expose
source headers, raw values or measured gains. It makes no training-convergence
claim: training is owned by the separate multilevel implementation.

Run `python3 -B -m unittest discover -s research/landmark_cache/tests -v`.
These offline tests cover catalog and cell mutations, target placement,
TRAIN-only access, private permissions, conditional publication, corrupt cache
rejection, interruption retention and dashboard redaction. A real native/S3
pilot is separate operational evidence. Root `just fast`/`just check`, security,
audit and receiving-main gates retain their existing roles and platform limits.
