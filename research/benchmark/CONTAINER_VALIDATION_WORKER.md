# Compressed validation worker preparation

`container_validation_worker.py` implements the proposed CPU sampling path for
the 32 compressed candidates from sixteen original artifact-cap failures. This
source change launches no real fit, sample, metric, transport or host probe. The
32-batch/192-cell training-derived validation preview remains unadmitted. All
original failures and both compression levels remain in their original lineage.

A future, separately frozen round must stage this worker as `source/entry.py`
alongside the frozen common, shared-runtime and native-runtime guards, unchanged
research codec and verified-member decoder. Its isolated bootstrap must verify
the interpreter/import roots and all source hashes before invoking the worker;
use `-I -S -B` with the frozen empty bytecode-cache path. The worker rejects
bytecode writes, added source files and directory aliases. No executable closure
or dispatch admission follows from copying these sources.

Before any source helper or dependency initializer, the worker checks the
externally supplied round digest, exact source inventory, sealed-test and null
claim flags, frozen parent receipts, training/validation worker inventory and
job identity. Admission, execution-lock freeze, clean SDV predecessor closure
and queued density priority must each be the JSON boolean `true`. These facts
must be proved by the coordinator and bound to the round; setting flags alone
is not evidence. The request must live under its canonical complete job digest
and distinct numbered attempt. Output directories require ownership by the
worker user and mode 0700. Hosts are limited to xbabe1/2/3.
Request membership uses the canonical complete job digest; equal-valued JSON
numbers cannot substitute another retry identity. CPU environment checks precede
all source helper loading and runtime initialization.

The decoder checks externally frozen container, kernel and projection digests
and charges the complete encoded artifact, including its header. Only verified
owned members are materialized in a new attempt-local directory, using exclusive
creation and read-only files. The decoded projection must match the frozen
worker projection and regression task. Primary sampling uses the materialized
kernel. The original kernel is read separately only for equivalence checks;
it never substitutes for the compressed input in the primary sampling path.

The proposed schedule retains fit seed 11, sample seeds 101/211/307 and `n`/`4n`.
All six samples must match original-kernel replay, the first sample must repeat
exactly, and one `4n` metric cell must repeat excluding measured elapsed time.
Metric receipts remain separate. A successful batch records exact job, round,
request, metric and container identities with the entire encoded charge. A
replay mismatch prevents a successful batch receipt; partial evidence remains.
The worker checks the native binary/library closure before each subprocess and
uses an explicit sampler environment with the checked PATH/locale/library path,
empty CUDA visibility and single-thread numeric-library limits. Unfrozen loader
inputs such as LD_PRELOAD, LD_AUDIT and GLIBC_TUNABLES are not inherited. The
future coordinator must also freeze a restricted interpreter/transport environment
before process initialization. It passes its remaining 600-second batch budget
to sampling. The outer coordinator
must enforce that same deadline across bootstrap, transport, all guards,
sampling and metrics, including partial or failed work.
Expiry during final integrity checks prevents the successful batch write and
raises a fixed timeout error, preserving the completed partial metric evidence.

Before dispatch, complete immutable SDV closure, honor queued density priority,
freeze the real runtime/binary/library/metric inventories, worker and receipt
bindings, exact matrix and retries, and admit fresh disjoint CPU/RAM/scratch
reservations under the 200 GB ceiling. No frozen live source is modified by this
change. Model-format validity, raw projection utility cost, privacy controls,
coverage and full fit/sample replication still need measured evidence. This
worker is a research wrapper, not a product wire format or release certificate.
Official tests stay sealed; MFS-v2/PTF-v1/release/superiority remain null. The
hermetic tests use opaque bytes and fake initializers/samplers only.

`container_python_custody.verify_import_inputs` supplies a separate check for an
already trusted coordinator before it starts the worker interpreter. External
manifest digests bind the interpreter, its declared `/usr/bin/python3` alias,
and the exact owned manifest bytes consumed by JSON parsing. Both manifests are
read once into bytes, hashed and parsed from those buffers; a separate final
rehash detects later path changes without substituting for that byte binding.
The declared inputs include the
absent ZIP import root, both package trees, standard library including bytecode
and extension files, exact declared file aliases, empty cache and metric source.
Added directory aliases and altered files fail with a fixed error. The function
compares interpreter and standard-library alias targets as literal text. The
interpreter alias must resolve through the filesystem to a regular file; trailing
slash or dot suffixes that require a directory reject, even when frozen literally.
The checker starts no process and imports no candidate dependency. It checks declared import
inputs only: system ELF/library closure, bootstrap source and controller custody,
restricted startup environment, capacity and deadline enforcement still require
separate evidence. Its successful receipt explicitly leaves full runtime closure
and execution admission false, and all gated scores null. Existing frozen runtime
manifests and live rounds remain unchanged. Stage any future bootstrap outside
the worker's exact six-file source directory.

```bash
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_python_custody.py -v
```

## Bootstrap invocation proposal

`container_bootstrap_invocation.prepare_invocation` consumes owned proposal bytes
bound by an external digest and prepares immutable interpreter arguments and an
explicit environment. Proposals remain unadmitted. They retain the exact 600-second
cap, sealed-test/null-claim flags, xbabe1/2/3 host scope, complete job digest and
numbered request path. Arguments use the pinned Python path with `-I -S -B` and
the declared cache prefix. The environment contains only the fixed PATH/locale,
declared single library directory, empty CUDA visibility and one-thread limits;
inherited loader, Python startup and credential environment is not copied.
The library directory rejects both `:` and `;` separators and `$` dynamic
tokens, retaining one literal directory. The loader recognizes both separators
and expands dynamic tokens; an empty list entry selects the current directory
([ld.so(8)](https://man7.org/linux/man-pages/man8/ld.so.8.html)).

The coordinator supplies the original outer-attempt monotonic start, before
guards or transport. Preparation checks the remaining budget after parsing.
`remaining_seconds()` retains that original start and must be called again
immediately before a future bounded operation; expiry at 600 seconds rejects.
This contract starts no process, enforces no transport timeout itself, checks no
filesystem ownership or ELF resolution, and supplies no predecessor or capacity
proof. Those properties remain false on the plan. Actual dispatch still needs
all separately frozen bootstrap/source/runtime/library/transport identities,
SDV closure, density priority, fresh aggregate admission and whole-batch deadline
enforcement. The worker independently verifies the real round and request.
No new fit, sample, tuning trial or gated score follows from a proposal.

```bash
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_bootstrap_invocation.py -v
```

## ELF declaration inspection

`container_elf_inputs.inspect_elf_inputs` parses owned bytes only after checking
their external digest. Its supported profile is ELF64, little-endian x86-64
executable/shared objects with fewer than 1,024 program headers and at most
4,096 dynamic records. Malformed headers, truncated or ambiguous mappings,
duplicate interpreter/dynamic segments and missing/unterminated strings reject
with a fixed error. The dynamic table's virtual mapping must match its file
offset. It records the declared interpreter, dependencies, loader
paths, audits and filter/auxiliary libraries without opening or executing them.

The inspector does not validate executable semantics or determine the libraries
actually loaded. Dynamic token expansion, RPATH precedence, cache/default paths,
preloads, hardware-capability directories and runtime `dlopen` inputs still need
separately frozen resolution and file-custody evidence. No filesystem, system
ELF closure, bootstrap/controller, transport, resource or job admission follows
from extracting declarations. Success leaves loader resolution, full runtime
closure and execution admission false and all gated scores null. The opaque
controls initialize no candidate process or library; live rounds remain intact.

```bash
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_elf_inputs.py -v
```

## Declared ELF file custody

`container_system_custody.verify_declared_elf_inputs` binds the private declaration
inventory to externally selected runtime and auxiliary manifest digests. The
owned manifest bytes are hashed before parsing. It checks complete Python import
trees and aliases, then derives the exact selected ELF members: the interpreter,
package files ending in `.so` or containing `.so.` in the filename, and matching
standard-library files. Omitted, duplicate or added inventory members reject.
Each selected file is read into owned bytes, checked against its parent digest
and replayed through the frozen ELF inspector. Canonical declaration comparison
keeps JSON integer, float and boolean identities distinct. Complete import trees,
aliases, manifests and inspector identity are checked again before success.

The caller must already trust the coordinator, interpreter and all guard source
before importing them. The returned receipt verifies declared files only; this
filename selection does not establish all native executable inputs. It starts no
candidate and neither resolves dependencies nor reads the loader cache, default
search paths or runtime `dlopen` inputs. Actual system-library resolution and
file inventories, bootstrap/transport/deadline/capacity proof, clean SDV closure,
queued density priority and cross-host checks remain required. Runtime closure
and execution admission stay false; all gated scores stay null. Existing locks,
live rounds, native objectives and artifact/compute caps are unchanged.

```bash
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_system_custody.py -v
```

## Selected loader input snapshots

`container_loader_custody.verify_loader_inputs` hashes the externally selected
private manifest before parsing it, then verifies its frozen parent manifests
and cache-print transcripts. It rechecks each selected regular file's bytes,
mode, owner and component-by-component alias chain. Candidate provider ELF
declarations are replayed through the pinned inspector without loading them.
Alias targets retain their literal text, including dot components and separators;
a trailing slash or dot requires a directory as it does during filesystem lookup.
Directory snapshots cover entry names, types and literal alias targets; they do
not hash the contents of every child. Frozen absence records retain the first
missing component and any aliases encountered before it. Added or changed
entries, retargeted aliases and changes to selected files reject. File hashes,
snapshots, manifests, transcripts and inspector identity are rechecked before
success. Errors contain no input paths, names or values.

Call this from an already trusted coordinator after the separate Python and
declared ELF file checks. All guard source and the coordinator's own import and
interpreter closure must be trusted before importing these modules. The private
proposal enumerates candidate files using package basenames, cache metadata and
direct absolute dependency names. A cache listing records candidates, not the
provider that the loader will choose. This checker verifies the selected
snapshots; it does not establish that the proposed set includes every input.
Actual RPATH/RUNPATH, cache/default-directory and hardware precedence, preload,
audit/filter and runtime `dlopen` behavior remain unresolved. See
[ldconfig(8)](https://man7.org/linux/man-pages/man8/ldconfig.8.html) and
[ld.so(8)](https://man7.org/linux/man-pages/man8/ld.so.8.html).

Success is a point-in-time custody check, not an atomic execution lease or a
complete loader-resolution proof. No process, library, fit or sample is started.
Cross-host identity, bootstrap/transport/deadline/capacity proof, clean SDV
closure and density priority remain required before any dispatch. Actual loader
selection, full runtime closure and execution admission stay false; all gated
scores stay null. Existing locks, rounds, native objectives and caps are unchanged.

```bash
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_loader_custody.py -v
```

## Ordered candidate catalog replay

`container_loader_catalog.verify_candidate_catalog` checks the selected loader
snapshots, hashes root ELF bytes and replays their declarations through the pinned
inspector. It derives the ordered root dependency, interpreter and search-path
records and compares them with the bound unresolved catalog. Cache-print bytes
are hashed before decoding. The fixed enumeration policy combines package
basenames, the transcript's x86-64 cache entries and direct absolute names.
It preserves first occurrence order, requires every candidate file binding,
seeds interpreter dependencies and replays recursive `DT_NEEDED` edges once per
new provider path. Repeated paths
terminate dependency cycles; an explicit edge ceiling bounds the traversal.
Interpreter paths are marked visited before seeding their dependencies. Repeated
interpreter records stay in the catalog, but a direct dependency or self-cycle
cannot expand an interpreter's dependencies again.
Canonical JSON comparisons preserve integer, float and boolean distinctions.
Omitted, added, reordered or changed edges reject, as do unsupported audit,
filter and auxiliary declarations, absent candidates and unbound provider files.
Root bytes, manifests, transcripts and selected snapshots are rechecked before
success. Errors contain no input paths, names or values.

Call from a trusted coordinator after the separate Python and root ELF inventory
checks; all guard source and the coordinator's interpreter/import closure must
already be trusted. Catalog order is the frozen enumeration policy, not glibc
search order. The receipt counts edges with distinct resolved file identities,
including identical bytes at different paths, without
selecting a provider. RPATH/RUNPATH and hardware precedence, complete default
search, preload, runtime `dlopen` and actual provider selection remain unresolved.
This starts no library, process, fit or sample and grants no execution lease.
Cross-host, bootstrap, transport, deadline, capacity, clean SDV closure and density
priority proofs remain required. Gated scores remain null and existing rounds,
native objectives and resource caps are unchanged.

```bash
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_loader_catalog.py -v
```

## Selected configuration declarations and directory snapshots

`container_loader_configuration.inspect_configuration` checks builtin byte and
digest identities before decoding a bounded ASCII profile: absolute canonical
directory declarations and fixed-parent includes with literal/star filename
patterns. Comments, blank lines, source line numbers and repeated declarations
are preserved appropriately. Unknown directives, multiple path arguments,
dynamic tokens, relative/noncanonical paths, escaping, unsupported glob syntax,
non-ASCII input and NUL reject. The profile accepts at most 1 MiB and 4,096
records. It is deliberately a declaration inspector, not a complete `ldconfig`
configuration parser, and opens no declared path.

`verify_configuration_inputs` first owns an externally selected private manifest,
pins the inspector file and verifies its frozen selected-loader parent. Every
selected configuration body is hashed before inspection, and its exact records
must agree with the manifest. It follows the supported includes against selected
directory entry snapshots, respecting leading-dot filename matching. Every
matching body must be selected; every selected body must be reachable from the
declared configuration root. Each configuration path is expanded once for
membership, with a bounded queue; this does not reproduce cache construction or
its handling of repeated includes. Additional directory/absence records must
cover exactly the declared directories absent from the parent snapshots, and
their names, types, literal aliases, mode and owner are checked. Records, source,
parents and snapshots are rechecked before success. Errors reveal no input values.

As with the other custody guards, call from an already trusted coordinator after
the separate Python/root ELF checks and trust all guard source before importing
it. Directory snapshots cover names/types/alias targets, not every child's
bytes. Configuration records do not prove that the loader cache was constructed
from these inputs or establish actual RPATH/RUNPATH, cache/default-directory,
hardware, preload or `dlopen` behavior. Actual selection, complete search/runtime
closure, cross-host identity, bootstrap/transport/deadline/capacity, predecessor
and density priority remain separate requirements. This starts no candidate,
library, fit or sample and grants no atomic execution lease. Official tests stay
sealed; gated scores remain null; existing rounds, native objectives and caps
are unchanged.

```bash
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_loader_configuration.py -v
```
