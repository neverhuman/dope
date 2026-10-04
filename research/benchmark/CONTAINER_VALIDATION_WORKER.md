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

## Declared search-path projections

`container_search_path_custody.verify_declared_search_paths` owns the external
projection manifest before parsing it, checks its source identity and binds the
exact configuration, selected-loader and root-declaration parents. It runs the
configuration snapshot and ordered candidate catalog guards, then reconstructs
every root and selected-provider RPATH/RUNPATH component in declaration order.
The supported projection substitutes literal `$ORIGIN` (followed by slash or
end) and `${ORIGIN}` tokens with the parent of the **declared pathname**, then
normalizes that pathname. Braced tokens can have literal suffixes. It performs
no alias resolution to determine origin. Empty, relative, unsupported dynamic
or NUL components reject. Each projection must have a verified directory or
absence snapshot in the selected/configuration parents; no unbound component
can pass. Canonical comparisons preserve ordinal, boolean and numeric identities,
including repeated components and their order. The component ceiling matches
the bounded candidate catalog. Bodies, snapshots, parents and source are
rechecked before returning counts. Errors reveal no input paths or values.

This is a literal declaration projection rule, not actual glibc origin, library
search order or provider selection. Call from an already trusted coordinator
after Python and root ELF custody, with all guard source and interpreter/import
closure trusted before import. Neither directory entry snapshots nor a declared
path prove child library contents or complete default/hardware/preload/`dlopen`
behavior. Point-in-time checks grant no atomic lease. Full runtime, cross-host,
bootstrap, transport, deadline, capacity, clean SDV closure and density priority
remain separate execution prerequisites. No library, process, fit or sample
starts; official tests stay sealed, all gated scores stay null, and the frozen
rounds, native objectives and caps remain unchanged.

```bash
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_search_path_custody.py -v
```

## Proposed loader environment snapshots

`container_loader_environment.verify_proposed_environment` owns the external
environment proposal and the bootstrap/search helper identities before deriving
the restricted invocation. Its exact seven-entry environment must match the
frozen declaration before loader files are inspected. This supplies the planned
empty CUDA, fixed PATH/locale/thread values and single library directory without
copying inherited startup or credential variables. The declared library directory
must be a verified directory snapshot and `/etc/ld.so.preload` must have a verified
absence record. The frozen declared-search-path guard checks the complete parent
chain before and after these bindings; source, proposal and parent bytes are
rechecked before success. The caller's original outer timer is used throughout,
including a final deadline check after the integrity reads; it is never reset.
Deadline exhaustion remains a typed `TimeoutError`. Other errors contain no
input values. Only the two bootstrap clock calls classify deadline exhaustion;
I/O timeouts reject with the fixed generic error and never publish their original
text. Clock timeout text is normalized as well. Opaque controls route preload
to a fake missing file and use no real loader, candidate interpreter or library.

The receipt verifies a proposed environment and selected snapshots. It does not
prove that a future process actually uses this environment, which libraries it
loads, child library contents, full hardware/default/cache/`dlopen` behavior, or
runtime closure. The caller must trust all guard source and its interpreter/import
closure before import and separately verify Python/root ELF custody. Bootstrap
entry/request/cache filesystem identities, transport, cross-host identity, actual
outer deadline enforcement, capacity, clean SDV closure and density priority
remain execution requirements. Point-in-time checks grant no atomic lease and
start no candidate process, library, fit or sample. Official tests stay sealed,
all gated scores stay null, and existing rounds, native objectives and caps are
unchanged.

```bash
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_loader_environment.py -v
```

## Proposed bootstrap filesystem custody

`container_bootstrap_filesystem.verify_proposed_filesystem` owns an externally
pinned filesystem manifest and its selected environment parent. It verifies the
restricted invocation and selected loader snapshots before inspecting proposed
startup files. Exactly the six frozen regular source files must exist; added
files, directories and aliases fail. The request is a regular file below its
canonical complete job digest and numbered attempt, in a caller-owned 0700
directory. Its externally frozen bytes are hashed before JSON is decoded. The
request's canonical job digest must match the owned unadmitted descriptor and
invocation, so integer and floating-point seed identities cannot be substituted.
The declared bytecode cache must exist, have no aliases and be empty.
Each inspection rechecks all cache path aliases, including after final parent
verification. Proposal I/O occurs outside the pure timing-call wrapper, so an
I/O timeout cannot become deadline exhaustion.

The guard repeats the environment check, all proposed file identities, descriptor
and manifest hashes, and the original caller deadline before returning a receipt.
Only trusted timing calls preserve a fixed typed deadline error; I/O failures
produce a fixed rejection without input text. The caller must already trust all
guard/coordinator sources and their complete interpreter/import closure before
importing this module, and separately verify Python and root ELF custody.

A staged `round-proposal.json` is metadata, not the final `round.lock.json` the
worker requires. This guard neither executes sources nor admits a final round,
request, atomic lease or process. The file snapshots do not prove actual loader
provider search, runtime closure, cross-host transport, predecessor closure,
density priority or resource capacity. Revalidation at dispatch and an atomic
execution lease remain necessary. Existing locked rounds and native objectives
are unchanged; official tests remain sealed and all gated scores null.

Focused controls use only opaque files and validated parent fixtures:

```sh
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_bootstrap_filesystem.py -v
```

## Native sampler executable declarations

`container_native_sampler_custody.verify_sampler_declarations` adds the separate
native sampler executable root omitted by the Python interpreter/package/stdlib
inventory. It owns bounded regular metadata files before decoding them, pins its
inspector/bootstrap/guard helpers, and derives the restricted proposal on the
original caller timer. The owned request's canonical job digest must match the
proposal and external manifest. Its original generator binary and recorded
product-source map must match the owned original GPU fit round.

The selected executable must be a regular unaliased file with the frozen exact
byte count and hash. Only owned bytes reach the ELF declaration inspector, whose
result must exactly reproduce the frozen declaration record. Parents, executable
bytes and helpers are checked again before the original final deadline and
receipt. Final metadata checks repeat regular-file and size admission before
hashing; final helper checks repeat unaliased regular-file admission, including
all parent components. Late aliases or nonregular replacements are rejected
before reading them. Proposal I/O is outside timing-only exception handling; unsupported
metadata, I/O and integrity failures have fixed no-value errors.

This proves selected native file/declaration custody and recorded metadata
identity. It does not rebuild product sources, independently replay parent fits,
verify interpreter/provider/library resolution or certify runtime closure. The
caller must already trust all guard/coordinator sources and their complete
interpreter/import closure before import, and separately establish proposed
filesystem, Python/root ELF and child library/environment custody. Metadata
files are limited to 16 MiB; this is separate from the generator artifact cap.

Both native guards admit initial helper paths as unaliased regular files before
hashing, rejecting FIFOs and directories without opening them. This repeats the
final helper admission and does not establish an atomic file lease.

The sampler needs its actual frozen native library directory; the earlier staged
planning directory is no substitute for that binding. Final execution locks,
transport, atomic leases, predecessor/density priority and capacity remain
unadmitted. No native executable/library, fit or sample is started. Official
tests remain sealed, original failures stay visible and all gated scores null.

```sh
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_native_sampler_custody.py -v
```

## Original native Torch library directory

`container_native_library_custody.verify_native_library_directory` binds one
selected original native-library directory to the owned sampler metadata,
original validation round and that round's host-runtime inventory. The host
label and binary identity must agree with the restricted proposed request.
All selected directory children must be recorded files or file aliases; each
recorded path, resolved path, exact byte count and file hash must agree with the
historical inventory. A separate owned snapshot records current literal alias
targets and directory membership, including ancestor aliases and ownership.
It does not infer original literal alias text from normalized legacy records.

The caller must establish all guard/coordinator sources and complete
interpreter/import closure before importing this helper. On the same original
whole-batch timer it must separately verify Python/root ELF, proposed filesystem
and native executable custody. This helper owns bounded regular parent metadata,
admits its own/loader/sampler helpers as unaliased regular files before their
initial hashes, derives a pure restricted plan, and checks
the directory and selected file identities twice. It rechecks bounded regular
parents, unaliased regular helpers and the original deadline before its receipt.
I/O errors have fixed no-value rejection text; only trusted timing calls retain
the fixed typed deadline error. No loader, executable or candidate dependency
is initialized.

This is point-in-time selected-directory/file custody. It does not establish
complete native dependency inventories, actual loader/provider selection, the
child's executable environment, cross-host identity, runtime closure, transport,
atomic leases, resource capacity or predecessor/density priority. The historical
runtime inventory's other files and observed `ldd` transcript are not replayed
by this helper. Final execution locks remain required; official tests stay
sealed and all gated scores null. Native objectives, old rounds/failures and
artifact-byte accounting stay unchanged.

```sh
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_native_library_custody.py -v
```

## Proposed native child environment

`container_native_child_environment.verify_native_child_environment` binds a
separate, restricted child proposal to the original selected native directory.
The predecessor proposal and its metadata remain unchanged. Every proposal field
except `native_library_directory` must match the original canonical JSON identity,
and the replacement directory must match the externally owned original library
inventory. Canonical request/job/round identity is checked with the frozen native
metadata; numeric or boolean changes do not match by Python value equality.

The pure invocation supplies empty CUDA visibility, fixed PATH/locale/thread
limits and the single declared directory. Its environment entries must exactly
match the owned declaration. A separately owned selected-loader absence record
binds `/etc/ld.so.preload`; its current identity is checked before and after the
selected original-library guard, and again after final metadata/source hashes.
All helper sources and metadata paths require unaliased regular-file admission
before hashing; metadata is bounded to 16 MiB. The original whole-batch timer is
never reset and is checked after final integrity work.

Call only after the trusted coordinator establishes all sources and complete
interpreter/import closure before imports. The caller must separately establish
Python/root ELF, the proposed filesystem and original native executable custody
on that same timer. This helper rechecks selected native-library files itself;
it does not perform those separate prerequisites or start a process. Other
native inventory files, actual loader/provider selection, an executed child's
environment, cross-host identity, transport, atomic leases, resource capacity,
predecessor/density priority and final campaign locks remain separate gates.

The receipt describes a proposed environment and point-in-time snapshots, not an
execution lease or measured generation. Official tests stay sealed; all gated
scores remain null. Native tuning objectives, artifact accounting and historical
rounds/receipts stay unchanged.

```sh
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_native_child_environment.py -v
```

## Original recorded native runtime files

`container_native_runtime_inventory.verify_native_runtime_inventory` rechecks
all file paths in one owned original validation host-runtime inventory, including
files outside the selected Torch directory. Its exact path set, resolved paths,
integer byte counts and SHA-256 digests must match that original inventory and
the host/binary identity of the separate restricted child proposal. A separate
owned snapshot binds current literal aliases, ancestor identities and ownership;
these are new point-in-time snapshots, not inferred original literal alias text.

The helper owns bounded regular metadata and admits unaliased regular helper
sources before hashing. It rechecks the proposed child environment and selected
native directory, then checks every recorded native file twice. Final metadata
and helper admission/hashes precede the final original whole-batch deadline.
I/O errors are fixed no-value rejections; only trusted timing failures retain the
fixed typed deadline error. No `ldd`, native executable, library or ML dependency
is started or loaded. The recorded linker transcript is not replayed.

Call only after a trusted coordinator owns all sources and the complete
interpreter/import closure before imports. The caller must separately establish
Python/root ELF, proposed filesystem and original native executable custody on
that same original timer. This is recorded file custody, not proof of all dynamic
loads, provider selection, an executed child's environment or system closure.
Snapshots and stat/hash checks are not an atomic execution lease; later changes
still require execution admission checks. Cross-host identity, transport, leases,
capacity, predecessor/density priority and final campaign locks remain separate
gates. Official tests stay sealed and all gated scores remain null.

```sh
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_native_runtime_inventory.py -v
```

## Combined proposed startup custody

`container_startup.verify_proposed_startup` binds the owned proposed filesystem,
original native sampler and full recorded native inventory into one startup
preparation. The selected inventory must reference that exact sampler metadata;
the filesystem's predecessor proposal must match the sampler's original proposal.
The separate native child proposal still changes only the library directory,
under the child guard's canonical request and proposal checks.

All three parent manifests are externally owned, bounded regular files. The
helper owns their linked Python/root declarations and replays the Python/root
ELF, proposed filesystem, original sampler and recorded native-file guards on
one original whole-batch timer. It checks time before the first stage, after the
Python/root stage, through the timed guards and after final metadata/source
hashes. Initial/final helper paths must be unaliased regular files before hashing;
final parent metadata repeats bounded regular admission. Fixed I/O rejection
text never includes private input values. Proposal file I/O is outside timing-only
exception handling; an I/O timeout is an integrity rejection even while the
original batch budget remains available.

The trusted caller must own every source and its complete interpreter/import
closure before importing these helpers. Replaying recorded preparation stages
does not admit the staged worker or prove actual loader/provider selection,
all possible dynamic loads, an executed child's environment or system closure.
Point-in-time checks are not atomic leases. Cross-host identity, transport,
capacity, predecessor/density priority and the five final campaign locks remain
separate gates. No fit, sample, native executable/library or dependency initializer
is started. Official tests remain sealed and all gated scores null.

Opaque orchestration controls explicitly stub only the independently tested
Python/root stage; the proposed filesystem, original native executable, child
proposal and recorded native files run their real guards against fake files.
The private preparation replay uses the actual recorded Python/root custody
inputs, with no candidate imports or execution; it is separate preparation cost,
not a benchmark outcome or independently reviewed private-data replay.

```sh
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_container_startup.py -v
```
