# Complete additional ARF fit ledger

This publication records all **796 physical / 800 logical** additional ARF
fits on the frozen 100 S3 regression lineages, at fit seeds 23/37/53/71.
Each seed has 100 author-default and 100 native-selected logical cells.
One lineage has identical default/native configurations at each seed;
those four aliases share their real fit receipts and add no independent fits.

Native-selected configurations are the winners already frozen at seed 11
using **held-out FORDE mean log density**, maximized. This round repeats those
fixed configurations and author defaults; it starts zero tuning trials.
No DOPE or common validation KPI selects an ARF configuration. Native density
values are listed for each checkpoint and are not ranked against other methods.

`fits.csv` contains each physical fit's native validation KPI, artifact bytes,
fit seed and immutable job digest. `panel.json` adds model/projection/adapter
inventories, hashes, split hashes, configuration labels and original
fit/job/closure/round receipt paths. The existing fit-ledger publisher verifies
source files, runtime inventory JSON, all model artifacts and receipt hashes
before producing those rows; it never loads the serialized models.
This is artifact/source verification, not a new recursive verification of all
runtime dependency binaries.

## Coverage and clocks

The original seed-11 fits remain in their separate frozen native round;
this publication does not fit them again or silently count them among the 796.
All four new seed cohorts are accounted for. Complete **common-metric five-fit
coverage** still requires the running sampler/evaluator and the original
seed-11 measurements. This fit ledger does not declare that metric matrix
complete. Earlier 24-fit/144-evaluation and first-prefix publications stay
unchanged.

The manifest reports summed worker fit/validation time and the full
coordinator wall time as separate overlapping clocks. The coordinator wall
includes capacity checks, launch and runtime setup; the worker clock starts
at the adapter call. They are not additive. Costs charge each physical fit's
complete artifact inventory once, including the model, projection and adapter.
The actual complete round is reported; no partial-cohort rate is extrapolated.

## Resource and scientific limits

The frozen CPU round uses four cores, nice 19, idle I/O priority, an 8 GiB RAM
cap with zero swap, a 15% available-RAM floor, and a 600-second worker cap.
Models remain on the original scratch root under its 200 GB admission ceiling.
No existing artifact is deleted or moved. No GPU is used in this round.

Official tests remain sealed. These are training-derived validation observations.
MFS-v2, PTF-v1, release-safe L3, superiority and DOPE win counts remain null or
false. Full production certification and formal privacy are not inferred from
this fit ledger. The separate shared evaluations will report fidelity, utility,
detection and empirical privacy when their complete frozen cells close.

## Reproduction

The manifest pins the immutable input ledger, original round and complete
coordinator receipt, a read-only systemd exit observation, physical-to-logical
coverage and the actual publisher replay. The unmodified
`research/benchmark/publish_cpu_fit_batch.py` regenerates the panel and CSV
from that input ledger. All committed JSON is schema-checked; the regression
suite regenerates the CSV and checks the 796 physical / 800 logical coverage,
four aliases, fixed author/native labels, artifact totals and null gates.
