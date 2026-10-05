# Completed DOPE GPU refinement discovery

Six stratified discovery lineages; four bounded GPU refinements; one fit seed, three sample seeds at n/4n. Discovery validation only; confirmation and global family selection remain pending.

Baselines retain their own frozen held-out likelihood selections. Sample seeds are reduced within each lineage before taking dataset medians. Negative retention is retained. This selected discovery subset is descriptive; no superiority or production certification is inferred.

| Method | Configuration | Size | Measured / planned lineages | Charged bytes, min–max | CatBoost median retention (informative N) |
| --- | --- | ---: | ---: | ---: | ---: |
| Chow-Liu | default | 1n | 6/6 | 3342–64200 | 0.614644 (6) |
| Chow-Liu | default | 4n | 6/6 | 3342–64200 | 0.544618 (6) |
| Chow-Liu | native_selected | 1n | 6/6 | 5161–1004240 | 0.420751 (6) |
| Chow-Liu | native_selected | 4n | 6/6 | 5161–1004240 | 0.347685 (6) |
| DOPE | features12_steps2048 | 1n | 6/6 | 1014–10171 | 0.971314 (6) |
| DOPE | features12_steps2048 | 4n | 6/6 | 1014–10171 | 0.985233 (6) |
| DOPE | features12_steps512 | 1n | 6/6 | 1016–10167 | 0.890048 (6) |
| DOPE | features12_steps512 | 4n | 6/6 | 1016–10167 | 0.952813 (6) |
| DOPE | features12_steps8192 | 1n | 6/6 | 1014–10183 | 0.945034 (6) |
| DOPE | features12_steps8192 | 4n | 6/6 | 1014–10183 | 0.973880 (6) |
| DOPE | features12_width8_steps8192 | 1n | 6/6 | 911–9852 | 0.905186 (6) |
| DOPE | features12_width8_steps8192 | 4n | 6/6 | 911–9852 | 0.945199 (6) |
| DOPE | features16_steps8192 | 1n | 5/6 | 1014–10381 | 0.956071 (5) |
| DOPE | features16_steps8192 | 4n | 5/6 | 1014–10381 | 0.948145 (5) |
| DOPE | features24_steps2048 | 1n | 5/6 | 1014–10787 | 0.978144 (5) |
| DOPE | features24_steps2048 | 4n | 5/6 | 1014–10787 | 0.977180 (5) |
| DOPE | features24_steps512 | 1n | 5/6 | 1016–10781 | 0.954247 (5) |
| DOPE | features24_steps512 | 4n | 5/6 | 1016–10781 | 0.938967 (5) |
| DOPE | features24_width8_steps8192 | 1n | 6/6 | 911–10190 | 0.883318 (6) |
| DOPE | features24_width8_steps8192 | 4n | 6/6 | 911–10190 | 0.931453 (6) |
| GaussianCopula | default | 1n | 6/6 | 3294–179086 | 0.696463 (6) |
| GaussianCopula | default | 4n | 6/6 | 3294–179086 | 0.901430 (6) |
| GaussianCopula | native_selected | 1n | 6/6 | 1216–179086 | 0.715331 (6) |
| GaussianCopula | native_selected | 4n | 6/6 | 1216–179086 | 0.922868 (6) |
| independent_marginals | default | 1n | 6/6 | 972–24795 | -0.125621 (6) |
| independent_marginals | default | 4n | 6/6 | 972–24795 | -0.005668 (6) |
| independent_marginals | native_selected | 1n | 6/6 | 1202–16518 | -0.071849 (6) |
| independent_marginals | native_selected | 4n | 6/6 | 1202–16518 | -0.022901 (6) |

24/24 fit cells accounted: 23 successful, one charged artifact rejection (10,381 bytes > 10,240). All 25 attempt receipts are retained, including the earlier infrastructure interruption and its successful retry. Validation: 15/15 physical batches, 138 measured / 6 fit-unavailable logical cells.

Fit operations: 557.698916s; validation operations: 301.542255s. Scheduler wall includes admission waits and is reported separately. Earlier architecture research costs remain separate.

The six disjoint confirmation lineages have not been evaluated in this panel. Official tests remain sealed; MFS-v2/PTF-v1/release/superiority stay null.
