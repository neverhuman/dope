# Private reusable training-partition CV landmark inputs

`research/levels/precalc` prepares model-independent small-model CV scores and validity masks. It does not train a shared model, admit source rights/families, or read official TEST objects. The human-authorized source is the checksum-pinned official TRAIN catalog; resolved consumer metadata remains a separate gate.

The finite registry contains `fit-iid-65-v1` and `outer5-iid-fold0-v1` through `outer5-iid-fold4-v1`. The latter each select the 80% outer training rows; landmarks use another five inner folds entirely inside those rows. Fold-local native transformations are learned within each inner fit. No outer query, tune, calibration, confirmation or TEST labels become encoder inputs. Group/time data require a qualified group/time producer; these IID cache entries do not authorize IID training for such data.

Physical IDs derive from the catalog dataset ID, independently of eventual family names. Row ordering binds physical ID, split seed and physical source indices; consumers must retain that ID through `physical_dataset_id`. The exact training CSV hash, schema, row scope, task, options, native binary, producer source, CPU/platform and allocation bind every result. Only explicitly declared targets are moved to the final canonical column. Predictor type inference sees selected training rows only. No source values or headers appear in error receipts.

The new namespace is `s3://veox-jopedime-datasets-use2/derived/dope/landmark-inputs/fit-cv-v1/precalc-20261023/`. The earlier whole-TRAIN diagnostic corpus remains immutable and cannot stand in for these split-bound input features.

- `datasets/<dataset>/physical-<id>/splits/<split>/seed-<seed>/train-<row-hash>/cv-<options-hash>/producer-<key>.json` indexes each measurement.
- `objects/sha256/<hash>.json` stores complete private native reports, 23 scores plus masks, and input manifests. The 2T native work budget is explicit; budget-refused/uninformative coordinates remain masked, never invented.
- `receipts/`, `dispositions/` and `runs/` retain actual costs, native exits, allocation failures and scheduled denominators. Failed/interrupted attempts require review; no automatic retries or relaxed budgets.
- `corpora/<index-hash>/index.json` records ready, failed, unsupported and pending cells across the whole finite schedule. A content-addressed index is a snapshot, never a mutable claim that unfinished work completed.

S3 writes use private account policy, AES256, SHA256 checksums, conditional create-only writes and exact readback. Raw CSVs are private local scratch only and are never uploaded into the derived namespace. Nodes without existing AWS credentials may use exact-TRAIN short-lived GET delegation and a local immutable store, followed by checksummed private courier publication; long-lived credentials are never copied.

```sh
python3 -m research.levels.precalc plan --catalog PRIVATE_CATALOG --source-commit FULL_GIT_COMMIT --out NEW_PLAN
python3 -m research.levels.precalc run --plan NEW_PLAN --catalog PRIVATE_CATALOG --binary PINNED_NATIVE --expected-binary-sha256 SHA256 --root NEW_RUN --workers 4 --threads 4 --shard 0 --shards 4
python3 -m research.levels.precalc index --plan NEW_PLAN --runs RUN0 RUN1 RUN2 RUN3
python3 -m research.levels.precalc download --entry-key SELECTED_ENTRY --expected-entry-sha256 ENTRY_SHA256 --out NEW_PRIVATE_DIRECTORY
python3 -B -m unittest research.levels.tests.test_precalc -v
```

Select allocations only after fresh CPU affinity/load, available RAM, disk and process-owner inspection. Workers run at nice10 with a conservative foreground reserve and per-child RSS/wall limits; replenishment pauses under contention. Dataset sharding keeps all six views together, downloads each exact TRAIN blob once and checks existing objects before computing. Cache keys are independent of model weights. Current input consumers require at least one valid coordinate; an all-masked measurement remains organized evidence but is not silently admitted. `download` produces a checksum-bound entry for `InputCache`; consumers still verify their own exact training CSV/schema/physical row scope.

This registry covers the named splits and seed, not every possible future random split. A new split policy or seed needs an explicit plan and separate lineage. No completed old pass is repeated, no held-out partition is opened, and no unqualified dataset is automatically promoted to shared-model training.

For IID reports, query evidence is checked against the exact native per-fold query caps: ridge and its quarter control use `ridge_rows` per fold; small learners and their quarter control use `ceil(small_rows/folds)`. Every query remains in its own held-out inner fold within the selected training partition. Larger inputs therefore legitimately score a capped subset, rather than requiring query counts equal to the entire input. Full native reports retain actual per-fold training counts and aggregate query cost.
