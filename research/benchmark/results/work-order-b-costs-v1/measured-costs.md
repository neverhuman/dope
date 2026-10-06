# Measured research operation costs

Phase receipts are deduplicated physical operations. Wall seconds include wrapper overhead. Combined sample/evaluator timing stays combined. These entries do not establish equal research spend.

| Method | Phase | Receipts | Operation seconds |
|---|---|---:|---:|
| ARF | fit_and_native_wrapper | 800 | 14412.990776 |
| CTGAN | fit | 352 | 19194.933444 |
| CTGAN | historical_trial | 48 | 2485.769219 |
| CTGAN | native | 82 | 15005.237298 |
| CTGAN | sample | 78 | 14556.921795 |
| Chow-Liu | historical_fit_native | 600 | 116.907228 |
| Chow-Liu | sample_and_evaluate_combined | 175 | 3445.465798 |
| Forest-Flow | fit | 12 | 2543.915411 |
| Forest-Flow | native | 12 | 1925.978670 |
| Forest-Flow | sample | 12 | 4331.598717 |
| GaussianCopula | historical_fit_native | 200 | 697.582553 |
| GaussianCopula | sample_and_evaluate_combined | 162 | 8266.579277 |
| TVAE | fit | 352 | 17819.189345 |
| TVAE | historical_trial | 48 | 1901.893978 |
| TVAE | native | 83 | 15227.403737 |
| TVAE | sample | 78 | 14457.682308 |
| independent_marginals | historical_fit_native | 300 | 51.770245 |
| independent_marginals | sample_and_evaluate_combined | 165 | 3274.596052 |

| Run | Method | Phase | Operation seconds | Accounting |
|---|---|---|---:|---|
| dope_population_research_v2 | DOPE | fit_wrapper | 3121.035606 | separate_run_aggregate |
| dope_population_validation_v1 | DOPE | sample_and_evaluate_combined | 3826.696542 | separate_run_aggregate |
| dope_refinement_expansion_v2 | DOPE | fit_wrapper | 3427.700352 | separate_run_aggregate |
| dope_refinement_expansion_validation_v2 | DOPE | sample_and_evaluate_combined | 5213.128865 | separate_run_aggregate |
| sdv_shared_validation_v1 | CTGAN+TVAE | sample_and_evaluate_combined | 856.923647 | separate_run_aggregate |
| density_shared_validation_v1 | density_methods_combined | sample_and_evaluate_combined | 14986.641128 | alias_of_density_phase_rows_do_not_add |
| forest_shared_validation_v1 | Forest-Flow | shared_evaluator_combined | 228.732251 | separate_run_aggregate |

| Historical interruption | Method | Operation seconds | Accounting |
|---|---|---:|---|
| 25e42657582eef8fa5e8bf7df210eeb8a5be0bc18c6a8d3da9f99b1fbac45ac6 | CTGAN | 195.733249 | separate prior receipt |
| a6ff74eda71dfceedf11cee6ed1986203cacaf95cf751717e395acd2c1cd220c | CTGAN | 427.965393 | separate prior receipt |

Prior pilot costs remain separate; no historical grand total is reported. Scheduling cutoffs retain their native outcome class and are not method failures. Whole-device VRAM observations are not process VRAM or attributed energy.

Historical co-tenancy is unknown. Future shared GPU claims require clock-bound receipt observations; historical omissions cannot establish exclusive use.

Missing evidence:
- Exact hardware model fields absent from these public historical receipts
- No separate sample/evaluator timing for combined legacy operations
- No job-attributed energy; VRAM values are observed whole-device usage
- Historical co_tenant flags absent; unknown is not exclusive GPU use
- Forest phase RAM/VRAM monitors and density historical host/RAM fields await incorporation
- Earlier architecture research may overlap old pilot costs; no summed historical total
