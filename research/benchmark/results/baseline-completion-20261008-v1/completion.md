# Baseline completion checkpoint

Frozen 2026-10-08T08:02:05.490098+00:00. Official tests sealed; gated scores null.

Metric target: 100 lineages × default/native × five fit seeds × n/4n × three sample seeds = 6,000 logical cells.

| Method | Published logical metric cells | Fit coverage / disposition | Failures / timeouts | ETA (MDT) |
| --- | --- | --- | --- | --- |
| ARF | 6000/6000 | 100 lineages, all five seeds; 5,892 physical metric receipts +108 aliases | No missing current matrix; historical failed attempts not recounted | DONE: actual common-metric close Oct7 16:46 MDT; PR187 merged Oct7 20:36 MDT |
| TabSyn | 143/6000 | 100/100 scaled-default fit11 OK; 35 earlier unmeasured allocations; 8 complete n/4n lineages; native winner unavailable | 0 non-OK current default fits; 68 earlier failed native-audit attempts; audit unavailable/input-unavailable remain distinct from fit failures | 272.67 eligible GPU-slot-hour allowance: conditional Oct19 16:40 MDT for Oct8 08:00 start; no slot granted; native objective readiness required |
| TabDDPM | 132/6000 | 21 retained models, fit11; other fit seeds pending; historical failures not recounted here | Historical failures/timeouts not recounted; 5868 pending slots are not failures | 287.67 eligible GPU-slot-hour allowance: conditional Oct20 07:40 MDT for Oct8 08:00 start; no slot granted; original history reconciliation required |
| Forest-Flow | 390/6000 | 343/400 additional default fit dispositions: 268 OK, 75 timeouts; published 13 five-fit lineages | 75 timeout; 0 other failures in this frozen additional-default round | 2026-10-08T10:11:01.787909-06:00 |

Counts and input paths/digests: `panel.json` and `inputs.lock.json` in this directory.
Timeouts remain missing and contribute no DOPE win. Forest default-only target is 3,000 metric cells.
Native-objective unavailability is not a generator failure. Pending cells are not failures.
No current bulk model hash, scientific auditor, or official test was reopened.
