# CTGAN / TVAE complete native research ledger

Complete bounded native research ledger; training-derived validation only; no shared test-quality comparison

100 matched S3 lineages, 800 logical trials (704 new attempts, 96 historical reuses).
All trials are accounted for; this does not mean all fits succeeded.

| Method | Native selected / 100 | Unavailable / 100 | Paired default–selected | Default median native R² | Selected median native R² |
|---|---:|---:|---:|---:|---:|
| CTGAN | 35 | 65 | 30 | -0.411065 | -0.0832399 |
| TVAE | 34 | 66 | 31 | 0.0664437 | 0.160284 |

Medians use the same paired cells within each method; native KPIs are not cross-method ranks.

New attempt terminal statuses: deadline_unstarted: 524, foreign_gpu_owner_appeared: 10, ok: 154, transport_or_prelaunch_failure: 16.

Failed/unstarted attempts and two earlier failed trials remain in JSON and cost accounting.
Official tests remain sealed. Shared validation quality is pending; MFS-v2, PTF-v1,
release-safe status and superiority are null. No DOPE win or production certification is claimed.

Regenerate using `python3 -B -m research.benchmark.publish_sdv_population_native` with
the two external SHA-256 anchors in the committed publication manifest.
