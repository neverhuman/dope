These tables are deterministic synthetic software fixtures, not benchmark
results. `manifest.json` marks the entire directory `citable:false`.

The header-bearing numeric toy table is shared by copied-rows, gzip-original and
native DOPE standalone adapter tests. It is constructed arithmetically and has
no downloaded or protected rows. Native table schemas cannot accept arbitrary
evidence fields, so the directory manifest carries the noncitable designation.

Golden input/output cases are in `tests/golden/`; they are executable examples
of an unfrozen protocol, never measured transfer or privacy evidence. Renderers
must call `bench.metrics.qualification.require_citable` (or enforce the same
strict rule) before admitting result records.
