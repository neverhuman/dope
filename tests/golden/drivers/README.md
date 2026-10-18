# Driver conformance fixtures

The JSON files in this directory are historical protocol v1 cases. An omitted
`protocol_version` dispatches to v1; an explicit `"v1"` produces the same result.
Candidate D80 fixtures live in `v2/`, use `schema_version: 2` and explicitly set
`input.protocol_version: "v2"`. Historical golden replay rejects that envelope.
Both Rust and Python driver unit suites evaluate the shared candidate cases and
compare their expected states, shapes and numbers at tolerance `1e-12`.

Run the candidate corpus with:

```
cargo test --locked --lib transfer_metrics::drivers::
PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_drivers -v
```

`just golden-parity` continues to check the historical corpus. C05 owns extending
the top-level versioned golden runner to discover candidate driver subdirectories.
Every case is noncitable fixture evidence, with analytic expected values and
fixture bounds; none is a measured scientific calibration or release receipt.

V2 requires a predeclared `required_components` roster containing `mass`.
`rank`, `sign` and `interaction` are optional declared gates. Nominal and
non-monotone features do not acquire an ordinal or global sign; additive tasks
do not require nonexistent interactions. Query population, ordered groups,
perturbation draws and PreparedView are evaluator-owned SHA-256 pins shared by
all three arms. Their values and importance vectors remain private inputs.
The caller must align these arms before reduction; this numeric consumer does
not authenticate an evidence graph or infer applicability from results.

Negative noisy loss increases are clipped to zero before mass normalization.
The independent real-control arm uses `real_control_importances` (and matching
sign/interaction arrays when applicable). For each applicable component the
reducer calculates `h = a_RS - 0.8*a_RR`. The mass agreement is
`a_RS = 1 - 0.5*L1(p_R,p_S)`. The ratio `clip(a_RS/a_RR,0,1)` is descriptive.
Zero absolute real importance and unstable real controls abstain.

Draft C02 interface: `inference.status` must be `complete`, `bound_method` must
be `full_family_max_t`, and `bounds` must include `estimate` and `lower` for
each `<component>.<endpoint>` coordinate. Endpoints are `margin` (h),
`stability` (a_RR), `real_strength` (positive real mass), `control_strength`
(positive independent-real mass), and `synthetic_strength` (synthetic/real
positive mass). C02 must resample paired endpoints jointly over the entire
declared claim family, including budgets and families; additional bound
coordinates are allowed. Each supplied estimate must match the recomputed
endpoint within `1e-12`, and the lower bound cannot exceed it beyond that
arithmetic tolerance. The decision itself compares bounds strictly: margin
LCB >= 0, stability LCB >= `stability_floor`, both absolute strength LCBs >=
`importance_floor`, and synthetic-strength LCB >= `synthetic_strength_floor`.
Missing bounds produce an incomplete result. Terminal perturbation failure,
timeout and unsupported states stay explicit.

All floors are positive, explicit draft inputs. Fixtures use illustrative
floors only. D05/D19 remain held and the candidate contract stays unfrozen.
The D00 integration assumption is a common PreparedView with predeclared opaque
group order; no raw headers, feature values or extrema appear in output.
The strict Draft 2020-12 result schema is
`schemas/driver-result-v1.schema.json`; its format version is independent of
the driver protocol version.
