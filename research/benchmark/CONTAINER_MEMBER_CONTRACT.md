# Compressed research members

The population byte probe retained sixteen original over-cap fits and produced
32 candidate containers, one each at compression levels 1 and 9. The complete
80-byte header, native model and learned projection are charged. Those byte-only
results do not establish generation quality or a release-safe artifact.

`research_container_members.decode_verified_members` prepares the next sampling
contract. It receives immutable encoded bytes plus three externally frozen
SHA256 values: the full container, original kernel and original projection.
All three digest arguments require the exact builtin `str` type, preventing
caller-defined comparison behavior. It checks the declared charge against the
entire encoded length and the unchanged
10,240-byte ceiling before bounded decoding. Internal header digests cannot
replace the independently pinned original member identities. The result contains
immutable owned model/projection bytes and the full encoded charge; it never
rereads a caller path or imports a model library. Errors contain no member values
or digests. The existing codec, probe sources and parent receipts remain unchanged.

This function establishes byte custody only. It has no fit/sample dispatcher,
runtime admission, production wire format or generator-validity certification.
No model or projection is omitted from the artifact charge. Decoded temporary
members are derived from the charged container; a later sampler must consume
these verified bytes, with no uncharged learned files or reread of original
training data. Native-format/model validity and projection applicability still
require separate checks.

Before any sampling round, freeze and verify the wrapper, existing codec, native
binary/library/runtime, metric implementations, parent job/fit receipts, worker
partitions, complete sample schedule and retries. Preserve all candidates and
the original sixteen failures. Finish immutable SDV predecessor closure, honor
queued density reservations and admit fresh disjoint CPU/RAM/scratch capacity
under the 200 GB ceiling on xbabe1/2/3 only. No new fit or sample is launched by
this change; the prepared 32-batch/192-cell validation preview remains unadmitted.
Actual generation must replay the original kernels and repeated sample seeds.
Official tests remain sealed. MFS-v2/PTF-v1/release/superiority remain null until
their full evidence and admission gates pass; no win or production claim follows
from successful decoding.
