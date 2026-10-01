# ARF complete native-validation pilot

Four original and four bounded-refinement trials were charged to each dataset. Selection used only ARF's held-out FORDE mean log density on training-derived validation rows. The likelihood evaluator is a labeled study implementation using the author's fitted factors. Values compare trials within a dataset, not methods or datasets.

| Dataset | Default | Selected round / trial | Native log density | Artifact bytes | L3 bytes | Successful trials |
|---|---|---|---:|---:|---:|---:|
| Adult | timeout | short / 1 | -3.508621 | 19,187,442 | no | 4/8 |
| California | ok | original / 3 | 22.547445 | 85,471,471 | no | 8/8 |
| News | timeout | short / 2 | 103.601409 | 55,316,440 | no | 4/8 |

All eight original Adult/News attempts timed out under their frozen 900-second caps; their null outcomes remain in the JSON. All twelve short refinements succeeded, but the final native selections are over the 10,240-byte L3 limit. Restricted artifacts and detailed logs stay on scratch. Official tests remain sealed; shared validation metrics, MFS-v2, PTF-v1, and production certification are unavailable.
