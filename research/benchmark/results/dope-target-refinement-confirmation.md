# Completed disjoint DOPE GPU refinement confirmation

Six stratified confirmation lineages disjoint from discovery; two bounded GPU refinements; one fit seed and three sample seeds at n/4n. Training-derived validation only; no global family selection.

Baselines retain their own frozen held-out likelihood selections. Sample seeds are reduced within each lineage before taking dataset medians. Negative retention is retained. This small confirmation subset is descriptive; no superiority or production certification is inferred.

| Method | Configuration | Size | Measured / planned lineages | Charged bytes, min–max | CatBoost median retention (informative N) |
| --- | --- | ---: | ---: | ---: | ---: |
| Chow-Liu | default | 1n | 6/6 | 3248–34036 | 0.653910 (6) |
| Chow-Liu | default | 4n | 6/6 | 3248–34036 | 0.712820 (6) |
| Chow-Liu | native_selected | 1n | 6/6 | 3348–102696 | 0.613496 (6) |
| Chow-Liu | native_selected | 4n | 6/6 | 3348–102696 | 0.705764 (6) |
| DOPE | features12_steps2048 | 1n | 6/6 | 984–5670 | 0.859083 (6) |
| DOPE | features12_steps2048 | 4n | 6/6 | 984–5670 | 0.939543 (6) |
| DOPE | features12_steps512 | 1n | 6/6 | 991–5654 | 0.668413 (6) |
| DOPE | features12_steps512 | 4n | 6/6 | 991–5654 | 0.694485 (6) |
| DOPE | features12_steps8192 | 1n | 6/6 | 991–5676 | 0.832749 (6) |
| DOPE | features12_steps8192 | 4n | 6/6 | 991–5676 | 0.937693 (6) |
| DOPE | features12_width8_steps8192 | 1n | 6/6 | 889–5344 | 0.843964 (6) |
| DOPE | features12_width8_steps8192 | 4n | 6/6 | 889–5344 | 0.949384 (6) |
| DOPE | features24_steps2048 | 1n | 6/6 | 984–6307 | 0.859083 (6) |
| DOPE | features24_steps2048 | 4n | 6/6 | 984–6307 | 0.888782 (6) |
| DOPE | features24_steps512 | 1n | 6/6 | 991–6286 | 0.645565 (6) |
| DOPE | features24_steps512 | 4n | 6/6 | 991–6286 | 0.664807 (6) |
| GaussianCopula | default | 1n | 6/6 | 5037–153153 | 0.562677 (6) |
| GaussianCopula | default | 4n | 6/6 | 5037–153153 | 0.657688 (6) |
| GaussianCopula | native_selected | 1n | 6/6 | 1257–71945 | 0.618586 (6) |
| GaussianCopula | native_selected | 4n | 6/6 | 1257–71945 | 0.668390 (6) |
| independent_marginals | default | 1n | 6/6 | 1083–12784 | -0.093886 (6) |
| independent_marginals | default | 4n | 6/6 | 1083–12784 | -0.023003 (6) |
| independent_marginals | native_selected | 1n | 6/6 | 1083–8541 | -0.074444 (6) |
| independent_marginals | native_selected | 4n | 6/6 | 1083–8541 | -0.003286 (6) |

12/12 GPU fit cells and 72/72 logical validation cells succeeded. Twelve physical batches closed with actual coordinator exit zero. All charges include model and projection.

Confirmation fit operations: 302.088672s; validation operations: 237.627321s. Combined discovery/confirmation fit operations: 859.787589s within the original 21,600-second ceiling. Scheduler waiting wall and earlier architecture research costs are separate.

Sampler replay is exact for n/seed101 (12 cells) and metric replay for 4n/seed101 (12 different cells). Each kind has 60 cells not repeated; 48 cells have neither repetition. The two profiles were frozen after discovery and before confirmation metrics. Prior DOPE and native/default density references are unchanged.

Confirmation is disjoint from discovery. This is not the public-core headline, final five-fit matrix, global candidate-family selection, or production certification. Official tests remain sealed; MFS-v2/PTF-v1/release/superiority stay null.
