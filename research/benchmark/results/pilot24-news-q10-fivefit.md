# News q10 DOPE: five GPU fit seeds on pilot validation

The q10 generator was trained independently at fit seeds 11, 23, 37, 53, and 71. Lossless packed artifacts each fit within 10,240 bytes. All 60 n/2n/4n/8n sample identities matched their deterministic reference hashes, and all 30 n/4n validation metric payloads replayed exactly.

| Fit seed | n median CatBoost retention | 4n median | Packed bytes |
|---:|---:|---:|---:|
| 11 | 0.9782 | 0.9987 | 9,201 |
| 23 | 0.3287 | 0.3109 | 9,205 |
| 37 | 0.5723 | 0.5478 | 9,204 |
| 53 | 0.7933 | 0.8170 | 9,195 |
| 71 | 0.4424 | 0.4084 | 9,201 |

Median across the five fit-seed medians: 0.5723 at n and 0.5478 at 4n. This is training-derived validation evidence. The one-fit-seed CTGAN/TVAE comparison is a separate paired panel. Official tests remain sealed; MFS-v2, PTF-v1 and production certification are null.
