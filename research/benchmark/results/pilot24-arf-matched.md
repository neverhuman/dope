# DOPE versus ARF: paired pilot validation

ARF tuning used its frozen held-out FORDE mean log density within each dataset. The shared outcome is CatBoost TSTR/TRTR retention at fit seed 23 and paired sample seeds on training-derived validation rows.

| Dataset | ARF config | Size | Paired seeds | Native density | ARF retention | DOPE retention | Paired DOPE − ARF | ARF bytes | Copy-gate failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adult | tuned | 1n | 3/3 | -3.508621 | 0.8322 | 0.8275 | -0.0047 | 18,665,222 | 3/3 |
| Adult | tuned | 4n | 2/3 | -3.508621 | 0.8337 | 0.8294 | -0.0043 | 18,665,222 | 2/2 |
| California | default | 1n | 3/3 | 22.322232 | 0.8961 | 0.7008 | -0.1953 | 51,000,334 | 1/3 |
| California | default | 4n | 3/3 | 22.322232 | 0.8998 | 0.6948 | -0.2071 | 51,000,334 | 0/3 |
| California | tuned | 1n | 3/3 | 22.547445 | 0.8993 | 0.7008 | -0.1968 | 83,169,286 | 0/3 |
| California | tuned | 4n | 3/3 | 22.547445 | 0.8990 | 0.6948 | -0.2034 | 83,169,286 | 0/3 |
| News | tuned | 1n | 3/3 | 103.601409 | 0.5743 | 0.3287 | -0.2538 | 54,748,166 | 0/3 |
| News | tuned | 4n | 3/3 | 103.601409 | 0.6088 | 0.3109 | -0.3164 | 54,748,166 | 0/3 |

Adult and News original-default fits timed out and are explicitly unavailable. One Adult tuned 4n sample timed out; its ARF metric is absent and it contributes no DOPE win. All ARF fitted artifacts exceed L3. JSON retains 23 measured cells, the failed sample, unavailable defaults, row-copy counts, native selections, and immutable receipt hashes. Official tests remain sealed; MFS-v2, PTF-v1, and production certification are null. This pilot is not a paper superiority result.
