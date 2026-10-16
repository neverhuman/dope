TDL conformance uses strict JSON files in transfer/, aggregate/, drivers/,
frontier/ and charge/. Every file has this envelope:

```json
{"schema_version":1,"case_id":"unique_id","component":"transfer","citable":false,"purpose":"software conformance only","input":{},"expected":{}}
```

Run `python3 -B -m research.transfer.tdl.golden --check`. Without `--check`, the
runner emits JSONL `{case_id, actual}` for an independent consumer. L0 owns
root `just golden-parity` and the Rust consumer; the proposed recipe stub is
`bench/metrics/golden-parity.just`. Python passing alone establishes no Rust
parity. Expected outputs are checked at absolute tolerance <=1e-12 for floats;
integers, booleans, states, keys, arrays and nulls match exactly.

JSON NaN/Infinity tokens, numeric exponent overflow, and duplicate object keys
are rejected by the loader. Invalid metric numerics are represented as strings
such as `"NaN"`, `"Infinity"` and `"1e309"` in valid JSON input, and must produce
stable failure codes. Receipts never echo input values.

The working protocol is not frozen. Transfer uses original-unit paired losses,
positive supplied real-gain LCB, and absolute/relative signal floors. Required
family means are aggregated by minimum, with the mean also reported. Cluster
bootstrap resamples source families, datasets and paired seed blocks; every
block contains every required family. Auditor observations are averaged inside
the block. Clusters, seeds and auditors are sorted for input-order invariance.

The portable RNG is unsigned xorshift64 with shifts 13, 7, 17, masking after
left shifts; zero seed maps to one. A draw selects `state % population_size`.
The lower quantile has sorted index `floor(alpha * (replicates - 1))`. Defaults
are 1000 replicates, alpha=.05, seed=1729. The cluster fixture uses 199 draws.
Arithmetic means accumulate `value / count` left to right to avoid sum overflow.

Driver inputs are already grouped by the evaluator on training-only data.
Mass agreement is 1-TV. Pairwise rank agreement gives matching ties full credit
and one-sided ties half credit (a single group scores 1). Sign agreement uses
only nonzero stable real directions with positive real importance. Interaction
agreement multiplies mass agreement by min(total strength)/max(total strength).
Each required component is divided by its real-real ceiling and clipped to
[0,1]; the final driver gate uses their minimum. Unstable ceilings and empty
real controls are uninformative; missing required components are incomplete.
The correlated-groups fixture's first group represents substitutable features.

Frontier cells retain raw measurements; cumulative transfer uses only artifacts
that pass the mandatory gates and driver threshold. The selected byte count is
the smallest *tested* artifact passing both thresholds. Missing required cells
propagate a typed state for the whole requested grid. No interpolation or
extrapolation is allowed.

The fixture container is `TDL1`, u32le section count, then lexically sorted
sections: u32le ASCII-name length, name, u64le content length, content. Payload
charges the entire container, including framing. Runtime/shared-prior files
must explicitly have `scope: runtime|shared_prior` and `dataset_specific:false`;
the executor must independently verify their frozen, data-independent identity.
Dataset-specific sidecars belong in payload. Sampling closure and cold bundle
include the full payload plus all declared fixed files. Amortization is
secondary; encoded scalar counts and source row counts cannot change charge.
Literal-storage cases retain every row and schema byte plus wrapper framing.
This reference format does not replace DPK or raw-artifact compatibility.

All cases are synthetic fixtures and `citable:false`. They are not measured
scientific, privacy, benchmark or release evidence. The standalone numeric toy
table and manifest live under `tests/fixtures/tdl/`.
