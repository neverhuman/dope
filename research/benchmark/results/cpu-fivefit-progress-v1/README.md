# CPU five-fit replication: first completed batch

This is an immutable fit-ledger checkpoint, not a completed benchmark panel.
The publication contains 24 successful ARF fits and eight successful Forest-Flow
fits. Each record identifies its frozen round, exact job, successful process
closure, artifact hashes, full model-plus-projection bytes and elapsed time.
No weights, generated rows, source headers or extrema are published.

ARF replicates the existing seed-11 default and native-selected configurations
on the original training-derived splits. The completed prefix uses seed 23
and covers 12 lineages; the frozen queue has 796 distinct new fits representing
800 logical configurations at seeds 23/37/53/71. Native selection remains the
seed-11 held-out FORDE mean log density. Reported native values are per-fit
measurements, not values to rank across datasets or methods. Worker threads
are explicitly reduced from 16 to four; original author code and scientific
parameters are unchanged. Historical runtime receipts are not upgraded.

Forest-Flow has four additional default fit seeds on two original lineages.
Its CPU implementation uses the original flow process with `gpu_hist=False`
and `n_jobs=1`. The queue contains 400 new default fits; native-selected
replications are not admitted by this round. The retained seed-11 native
selection evidence must bind each later native configuration. New artifacts
remain in the owner-authorized local scratch root, with a 200 GB ceiling;
nothing is deleted. Runtime path permission is extended to this new root.

Both queues use four-core affinity, nice 19, idle I/O priority, an 8 GiB
cgroup memory cap with swap disabled, a 15% available-RAM floor and a
600-second fit deadline. Earlier source/runtime controls remain separate
from these new scientific receipts. The parent interpreter/OS and every
possible dynamic library load are not certified by this ledger.

The current published artifact totals are 9,604,706 bytes for ARF and
1,442,562,203 bytes for Forest-Flow. These are unconstrained research models;
a generator's complete artifact bytes must still meet the 10,240-byte cap
before any release-safe L3 comparison. Shared n/4n utility, fidelity,
detection and privacy measurements for these new fits are pending.
Official tests remain sealed. MFS-v2, PTF-v1, release-safe and superiority
are null; missing coverage is not a DOPE win.

## Reproduction

`manifest.json` pins every public data file and the producer. On the receipt's
original host, invoke `python3 -B -m research.benchmark.publish_cpu_fit_batch`
with `--manifest`, its `--manifest-sha256` and `--output`. The two input
manifest paths and hashes are recorded in `manifest.json`. The producer
verifies receipts before metric decoding and rehashes source, runtime
manifest, model and projection files. Every CSV also regenerates from the
committed JSON using `render()`; the contract tests verify exact CSV bytes,
schemas, public hashes and incomplete-matrix claims without reading scratch.
