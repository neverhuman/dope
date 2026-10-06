# MFS component evidence gaps

Compactness has 138 measured per-fit values: 136 successful charges and two over-cap charges (compactness zero). Another 62 fits lack a measured charge. Model and projection bytes are fully charged.

| Component | Known fit values | Aggregate score | Missing evidence |
|---|---:|---|---|
| utility_transfer | 0 | null | Missing the minimum one-sided 95% lower retention bound across required dataset profiles, five fit seeds and sample schedules. Research medians do not provide this component. |
| driver_fidelity | 0 | null | Missing feature importance Spearman rank agreement and top-k Jaccard agreement for the required auditors. |
| distribution_fidelity | 0 | null | Missing marginal W1, sliced Wasserstein and multibandwidth MMD results. KS alone does not provide this component. |
| structure_fidelity | 0 | null | Missing the production dependence fidelity calculation and required repetitions. Pairwise correlation alone is insufficient. |
| coverage_realism | 0 | null | Missing the production PRDC precision, recall, density and coverage calculations for n and 4n and required repetitions. |
| compactness | 138 | null | 62 fits have no measured artifact charge. Per-fit compactness values lack the required profile and repeat coverage for an aggregate. |

All hard gate outcomes, MFS-v2, PTF-v1, release-safe and superiority remain null. Production contract bytes and the scalar formula are unchanged. No official test data was opened and no metric job was started.

Missing global gates:
- Five matching final admission locks
- Required dataset and product profile coverage
- Full five-fit-seed x n/2n/4n/8n matrix
- Membership and attribute inference attacks
- Required real-versus-real and replay controls
- Authorized sealed official-test evaluator closure
- Preregistered paired analysis
