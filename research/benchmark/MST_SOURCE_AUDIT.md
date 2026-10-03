# MST author-source admission audit

MST remains `source_audit_pending`. This audit admits no runtime, model or
fit/sample job. Official tests remain sealed; MFS-v2/PTF-v1/release/superiority
remain null. No author modules were imported or executed.

The [author MBI repository](https://github.com/ryan112358/mbi) is pinned at
`e9d10d045247577db2d4742b15de325817b5bc59`, with a verified Apache-2.0 license.
[mst-source.audit.json](mst-source.audit.json) binds 32 selected source/configuration
files from the previously verified archive to their Git blobs and SHA256.
This is the current author's MBI 2.0.0 implementation, not a reproduction receipt
for an earlier paper revision. Selected source is not a complete executable or
third-party-rights inventory. Citation: `mckenna2021winning`.

## Author defaults and native outcome

The released MST function uses 2,500 estimation iterations to select a spanning
tree and 10,000 iterations for final estimation. CLI defaults are epsilon 1,
delta 1e-9, two-way marginal evaluation and a 10,000-cell workload limit.
These are source facts; the campaign representation and configuration remain
unfrozen. Protocol epsilon/delta cells must stay separate during selection.

The author CLI reports mean marginal total variation error, minimized: half the
L1 distance of normalized real and synthetic counts, averaged over the selected
workload. Bind this author metric to training-derived validation before tuning.
The scalar implementation, search grid and ties remain unfrozen; native values
are not ranked across methods and our shared KPI is not a tuning substitute.

## Fit/sample and runtime gates

The author function returns a synthetic table and does not expose reusable model
and domain-compression state. Freeze a minimal original-source adaptation that
exposes those states, then verify a safe numeric artifact codec and artifact-only
reload. Storing the fitted output table as a generator is not admitted. Charge
all model, support, configuration and representation files; reject original rows
in inference artifacts and verify finite requested sizes, seeds and replay.

The core uses discrete domains. Native categorical and projected common-numeric
representations need separate bindings. Fit and ordinary model sampling use
NumPy global randomness; the separate JAX extension has a different seed API.
Pin the correct API and initialize fit/sample seeds explicitly.

The package requires Python >=3.11 and JAX >=0.9.0. The mechanism imports pandas,
which is not a direct package dependency. Interpreter/import roots, bytecode,
libraries and source rights still need full custody. CPU/RAM/scratch and timer
admission must respect the live study-owner and predecessor locks.

Formal DP additionally requires accounting for preprocessing and selection and
an audited mechanism/runtime. No formal DP claim or study reproduction is made.
