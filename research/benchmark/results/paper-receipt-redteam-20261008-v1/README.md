# Paper receipt red-team: 2026-10-08

Snapshot: origin/main `26801e4b45cd6e4867d33bf8a4094e2c01d1c02b`, including PR187.
Only repository metadata and existing publication extracts were read. No original
row, model, synthetic sample, or official test was opened. This is an evidence
review, not a new scientific evaluation or a certification.

## Trace inventory

`report.json` pins each of the 27 TeX inputs and six numerical figure PDFs,
its generator and reduction inputs. The descriptive KPI file has 2,648 values
with publication hashes and JSON pointers; exact pointer/reduction tests are
required alongside manuscript macro, regeneration, PDF layout and leak checks.
Algorithm diagrams, budget constants and invented arithmetic examples are not
measured KPI claims. This inventory does not certify that every original
scratch receipt was rehashed. Current model weights were not read.

## Findings for the sole paper owner

### TRACE-01 (evidence_gap)

Loss-curve points/final hidden-basis loss, replay elapsed summaries, BeyondArena per-family elapsed/loss/byte extracts and provenance file counts cannot all be resolved to individually hash-bound original files from the aggregate extracts alone.

These extracts contain measurements/counts and source descriptions or redacted source labels, but not a complete per-file path+SHA-256+field mapping. Exact regeneration from an extract is not original-receipt verification.

Paper owner: retain the measured values and distinguish aggregate-only traceability. Compute lane: a later receipt-index publication should bind original receipt/report/loss files; no original values or receipts may be reconstructed by guessing.

### SCOPE-01 (scope_ambiguity)

Unqualified fit seed 11 only / no five-fit evidence wording can be read as applying to the new ARF and Forest-Flow appendices.

ARF now has100 lineages with all5 fit seeds and Forest-Flow has13 published five-fit lineages; primary DOPE/legacy density-neural headline still uses fit11.

Sole paper owner should qualify those passages as the primary displayed DOPE/legacy headline. ARF validation completion must not imply official-test, release or all-baseline five-fit completion.

## Scientific conclusions

No new superiority, equivalence, formal privacy or production claim is supported.
ARF has five-fit common-validation evidence; TabDDPM and TabSyn do not have
complete native/default five-fit population coverage. Native KPI values from
different methods are not ranked. Partial groups have no aggregate mean.
Official tests remain sealed; MFS-v2, PTF-v1, release-safe L3 and superiority
remain null. Missing baseline slots are pending rather than failed methods.

Passing regeneration checks verifies reductions of frozen publications. It
does not resolve TRACE-01 by itself. Original-receipt mapping must be preserved
when available, without guessing hashes, editing original receipts, or claiming
closed-loop production evidence. The paper owner owns any prose correction.

The completed focused replay/schema/source-pointer checks and full `just paper-check`
returned exit0; actual receipts are hash-bound in `report.json#/checks_completed`.
The latter completed in180.341 seconds. Its launch requested two CPU cores,
4GiB maximum, nice19, idle I/O and the15% RAM floor. A later rebase audit
observed Snap moving `just` into a separate scope; the original child scope
ceiling was not verified. The current owned audit scope now has independently
checked two-core,4GiB,zero-swap limits. These are verification resources, not
scientific fit-cost or admission claims. No original row/model/sample/test was
opened. Repository required gates and hosted CI remain separate from this
red-team proof.

## Historical paper bytes

`historical-paper.json` indexes deterministic gzip snapshots of the exact
paper inputs read at `reviewed_main_commit`. Each snapshot was obtained from
that Git commit and verified against the original report's byte length and
SHA-256 before compression. The report and findings are unchanged. This
lets later corrections update the manuscript while the historical audit
continues to verify its original evidence, rather than pinning future
manuscripts to an obsolete text. Immutable benchmark publications remain
bound to their original committed paths.
