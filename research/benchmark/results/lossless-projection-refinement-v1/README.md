# Seed11 lossless byte-cap diagnostics

Two retained `features12_steps2048` seed11 model/projection pairs were encoded in the research-only `DOPECR01` wrapper using zlib level9. The measured charge includes the complete 80-byte header and payload.

| Opaque lineage | Historical raw charge | Complete research wrapper | Historical parent result |
| --- | ---: | ---: | --- |
| `654566a1f91408aa` |22,233B |7,644B | `charged_artifact_cap`, exit1 |
| `a32a69e7c701eb14` |19,019B |5,275B | `charged_artifact_cap`, exit1 |

Both wrappers fit the 10,240-byte research cap. The admitted byte probe reported exact recovery of the original member bytes, exact repeated encoding, and a passing existing member verifier. Historical raw-charge failures and parent exit1 receipts remain unchanged.

The byte probe exited0 after 0.156343s internally (0.203225s including its wrapper), with 26,173,440B peakRSS and one declared CPU core. Separately, 27 generated opaque controls passed with actual exit0, 0.058121s elapsed and 16,777,216B peakRSS. These timings cover byte diagnostics and controls only.

No new fits, samples or GPU operations were performed. Sample parity and generator validity were not measured. These two byte diagnostics do not count as complete generators, DOPE wins or L3 production certification. `DOPECR01` is a research wrapper; production DPK3.2/v1 formats are unchanged. MFS, PTF, release and superiority gates remain null.

[Report](seed11-byte-cap-diagnostics.json), [strict version1 schema](seed11-byte-cap-diagnostics.schema.json) and [source proof](source-proof.json) bind the existing measured receipts and exact producer source hashes. Private absolute references remain in the ignored proof manifest. Publication preparation replayed pinned metadata and source hashes; it did not reread model, projection, container or CSV bodies or rerun the byte probe.
