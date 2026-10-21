# Dedicated BeyondArena TRAIN export

This is an experimental trusted data-courier producer. It is not a model runner,
product route, evaluator or proof of independent sealed custody. The historical
training-screen helper and its family policy remain unchanged.

The operator supplies an externally pinned grant with role
`trusted_data_exporter`, explicit execution authorization, actual exporter UID,
a finite start/expiry, source revision and metadata hash, exact versioned raw
object identity, exact fold view, group/time roles, ordered membership hashes,
producer file hashes, private output directory and explicit S3 profile. A queued
or candidate grant with `execution_authorized: false` cannot fetch data.

The producer authenticates source/task/split bytes before fetching one exact S3
object. It verifies returned version, length and SHA before decoding. Original
S3 bytes remain unchanged. The operator must keep raw-source access out of the
optimizer environment and expose only the dedicated export directory.

**Physical decoding boundary:** a mixed Parquet row group can decode outer-test
values inside this trusted exporter. This implementation does not claim that
those bytes were never decoded. Only official TRAIN rows are exported; outer
TEST targets are not inspected by the selection or inner-split logic. Declared
outer group/time keys are checked for disjointness/order. It is unsuitable where
policy forbids even a trusted exporter from decoding mixed source row groups.

The target is resolved from bound metadata, never raw column position. Group and
time keys are retained as private identities, excluded from predictor columns,
and the declared target is placed last without changing its values or dtype.
Original physical row positions bind to the original source hash. Inner fitting
and validation preserve whole group/time blocks; IID feature aliases also stay
together, without using targets to choose a partition. Ambiguous time strings,
unsupported group/time combinations, missing role columns, duplicate columns,
invalid memberships and mismatched source versions are refused.

No transform is fitted here. Subsequent preprocessing must fit only the recorded
inner fitting partition, retain and charge its full state, and preserve query
and target semantics. Source MAPE and transfer-contract MSE are distinct; an
original-unit MAPE zero-target convention remains a caller qualification leaf.
Recorded license text is copied verbatim, not upgraded or treated as a blanket
rights, privacy or model-admission decision.

Install the bounded data dependency from `train-export-requirements.txt` into an
owned isolated environment. Tests need no S3 or credentials:

```sh
python3 -B -m unittest research.benchmark.tests.test_beyondarena_train_export -v
```

After a separate actual resource/data-export grant, the existing courier invokes:

```sh
python3 -B -m research.benchmark.beyondarena_train_export_io \
  --grant GRANT.json --grant-sha256 EXPECTED_SHA256 \
  --source SOURCE.json --task TASK.json --split SPLIT.json --profile EXPLICIT_PROFILE
```

The CLI limits address space and CPU time, binds a finite wall deadline, disables
SDK retries and emits fixed refusal codes rather than source values. Its host
owner still enforces the allocated CPU slot, aggregate RAM/output/cost budget and
raw-source isolation. A complete `manifest.json` plus verified output hashes is
required before consumption; interrupted directories remain incomplete evidence.
The output has no model or scientific acceptance status.
