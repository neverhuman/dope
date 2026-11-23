# V8 evidence crosswalk

The approved plan governs implementation and admission. The proposal references below identify overlapping statistical families. An alias identifies an experiment or an existing primitive to check and carries no claim of coordinate or estimator equivalence.

| Tip | SHA-256 | Release interpretation |
| --- | --- | --- |
| Catalog · `DOPE_Dataset_Features_and_Transfer_Engineering_Spec.md` | `a8075975264260c90b24345e2938b87152d038f64d8c6cc131c7f0f3ae082daf` | F001–F144 identity, FIT/fold provenance, controlled ablations. |
| RichStats · `DOPE_Dataset_Features_and_Transfer_Engineering_Spec (1).md` | `cb0a2c2e6e4c674a48bd3884a247fa7d7dba64edaf139d87688247dc744be69c` | 128-column budget, reliability and group/class routing. Pair/global widths follow the approved plan. |
| DFP1 · `DOPE_FEATURE_PACK_V1_SPEC.md` | `b3fdfdb36bbc57c6c945b42cd9316f381813fad516644e7cdac2a4d3c33e7081` | T0–T3, typed support, native landmark reuse, conditional witnesses and closed loop. |
| Evidence · `DOPE_V8_EVIDENCE_SPEC.md` | `b3e984efefb96dc7a45ed90d5d83d11d3c1f122aab9caa37321abfc644be4c53` | Keep the public ladder as experiment ordering. Native replay sets promotion evidence. |
| Runbook · `dope_v8_engineering_agent_execution_spec.md` | `7cda5da0bbb2e8a71291c95919f921f23479261873d6bad85da5c68e41861826` | Reuse row-local and hard-parity checks. The approved little-endian DSE8 v1 contract supersedes its wire v3 proposal. |
| Compiler · `dope_v8_engineering_specification.md` | `cb92fc541e82f0d0a016cd409f192a7f1cdccaec9c5b3982b57f63ad1b132bcb` | Keep two normalized views and owner-side compilation. Flows, receiver weights and Soft-NODE need separate admission. |

## Layout and evidence

- The approved layout has 128 column, 16 pair, 16 class, 8 group and 160 global coordinates, plus a separate class-column block. The Antigravity draft of SHA-256 `7eeb2dde4c7d048a6250344c5f9f9bac2fd0d4f47ee5fc4be61036ca9bfe4fcd` proposes four class-column coordinates. Stable IDs in the table refer to this draft and need reconciliation at contract freeze.
- Candidate `8f4b4b92f8866be297409c55bcdb65d3706d6740` retains a four-coordinate class-column block in layout SHA-256 `e99647eb75cd078e0951ea93e12f48980475cab17f9fae028e05e4eb0d38999c`. The contract defines presence masks for column, edge and global blocks. Per-value effective support and uncertainty validity, and presence semantics for class, class-column and group outputs, still need frozen fields.
- The source catalog provides F001 through F144. DFP1, RichStats, Evidence, Runbook and Compiler describe overlapping families at different widths and estimator granularities. Freeze just one meaning per coordinate, with a tier, estimator version, mask, effective support and an independent uncertainty-valid flag. A packet-dependent statistic needs the analytic packet identity in its cache key. Record empty slots explicitly.
- Retain the 23 identities from [landmarks-native-v1](../../../../engine/dope-core/src/landmark/layout-native-v1.json) when using exact native landmarks. The evidence tip measured a 389.6 ms one-thread median for the full fine set. Keep that evaluation off the fast per-episode path. Distinct micro-landmark recipes need their own IDs.
- The first byte-spending experiments are typed support, rank dependence, the conditional base, supported effects and signed pairs, small trees, residual noise and regimes. The owner evidence arms must beat the equal-capacity packet-only control. Admission requires an identifiable native or training consumer, real serialized cost and guarded lift.
- The launch envelope is fixture-sized local testing. DEV and protected evaluation require the corresponding owner or custodian handoff. Analytic V8, learned V8 and automatic format selection each have their own admission decision.

### Dispositions

- **Gated experiment (G)** is an approved candidate awaiting native implementation or an admission receipt. No new recipe has an independently verified V8 consumer on this source baseline.
- **Reused primitive (R)** points to an existing source primitive for reuse, with V8 semantic parity pending. F018/F025 use `marginal/{atoms,lattice}.rs`. F045 uses DSE1 numerical dependence primitives, including rank/tie handling, where parity holds.
- **Deferred (D1)** covers additional shape and association tests alongside the cheaper core and tree witness. The evidence tip’s public ladder supports prioritizing the tree witness. Revisit when native ablations show incremental signal.
- **Deferred (D2)** covers detailed typed interaction, predictability and costlier influence alternatives. Admission requires incremental conditional/structural evidence with the core type map, masked curves and grouped witnesses in place.
- **Deferred (D3)** covers deeper geometry, topology, regime diagnostics and higher-order alternatives. Bounded reuse of simpler summaries comes first. Require measured section utility and extraction latency before expanding.
- **Deferred (D4)** depends on declared environment, entity or time metadata. The public route qualifies independent tabular draws. Group-aware splits remain required.
- **Deferred (D5)** is a measured-gap optimizer experiment outside the initial objective. Require gradient diagnostics and a matched-compute comparison.

### Catalog

| ID | Recipe | Draft native alias | RichStats §9 | DFP1 §4 | Evidence §4 | Runbook | Compiler §3 | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F001 | Sample-size profile | G_META_TASK | Global support / target | G.META, G.Y | N, T | §3.4 4–15 | C1 | G |
| F002 | Dimensionality profile | G_META_TASK | Global support / target | G.META, G.Y | N, T | §3.4 4–15 | C1 | G |
| F003 | Constant and unavailable fractions | S_SUPPORT | Support/type | C.TYPE | S01, S03 | §3.2 support | A4 | G |
| F004 | Duplicate-group concentration | G_META_TASK | Global support / reliability | G.META | G, Xg | §3.4 26 | C1 | G |
| F005 | Missing-pattern diversity | G_META_TASK | Global support / reliability | G.META | G, Xg | §3.4 26 | C1 | G |
| F006 | Target balance | G_META_TASK | Global support / target | G.META, G.Y | N, T | §3.4 4–15 | C1 | G |
| F007 | Covariance effective rank | G_SPEC_MP | Global spectrum | G.SPEC | D10 | §3.4 16–20 | C3 | G |
| F008 | Spectral energy curve | G_SPEC_MP | Global spectrum | G.SPEC | D10 | §3.4 16–20 | C3 | G |
| F009 | Regularized conditioning | D_DEP_PREC | Local/redundancy | C.DEP | D02 | §3.3 ch 1 | A5, C5 | G |
| F010 | Constraint inventory | same_nc_group | Structural metadata | C.RULE, edge 12 | S02, E08 | §3.3 ch 14 | A4 | G |
| F011 | Supported target gain | G_TARGET_RESID_T03 | Global target/difficulty | G.LIFT | L | §3.4 28 | C5 | G |
| F012 | Preparation evidence coverage | G_META_TASK | Global support / reliability | G.META | G, Xg | §3.4 26 | C1 | G |
| F013 | Multiresolution quantiles | M_QF_RESID | Marginal quantiles | C.QF | M07 context | §3.2 quantiles | A1 | G |
| F014 | Robust scale ratios | M_SHAPE_TAIL | Robust shape | C.SHAPE, C.TAIL, C.BOUND | M01–M06, S05 | §3.2 shape | A2–A4 | G |
| F015 | Quantile asymmetry | M_SHAPE_TAIL | Robust shape | C.SHAPE, C.TAIL, C.BOUND | M01–M06, S05 | §3.2 shape | A2–A4 | G |
| F016 | L-moment shape | M_SHAPE_TAIL | Robust shape | C.SHAPE, C.TAIL, C.BOUND | M01–M06, S05 | §3.2 shape | A2–A4 | G |
| F017 | Clipped-moment shape | M_SHAPE_TAIL | Robust shape | C.SHAPE, C.TAIL, C.BOUND | M01–M06, S05 | §3.2 shape | A2–A4 | G |
| F018 | Modal/atom concentration | S_SUPPORT | Support/type | C.TYPE, C.BOUND | S04–S07 | §3.2 support | A3–A4 | R |
| F019 | Entropy / effective support | M_SHAPE_TAIL | Robust shape | C.SHAPE, C.TAIL, C.BOUND | M01–M06, S05 | §3.2 shape | A2–A4 | G |
| F020 | Tail mass curve | M_SHAPE_TAIL | Robust shape | C.SHAPE, C.TAIL, C.BOUND | M01–M06, S05 | §3.2 shape | A2–A4 | G |
| F021 | Tail quantile spacing | M_SHAPE_TAIL | Robust shape | C.SHAPE, C.TAIL, C.BOUND | M01–M06, S05 | §3.2 shape | A2–A4 | G |
| F022 | Multimodality statistic | M_SHAPE_TAIL | Robust shape | C.SHAPE, C.TAIL, C.BOUND | M01–M06, S05 | §3.2 shape | A2–A4 | G |
| F023 | Shape-model distances | M_SHAPE_TAIL | Robust shape | C.SHAPE | M04–M06 | §3.2 shape | A3 | D1 |
| F024 | EDF goodness-of-fit profile | M_SHAPE_TAIL | Robust shape | C.SHAPE | M04–M06 | §3.2 shape | A3 | D1 |
| F025 | Discreteness / spacing profile | S_SUPPORT | Support/type | C.TYPE, C.BOUND | S04–S07 | §3.2 support | A3–A4 | R |
| F026 | Hurdle and zero-inflation profile | M_SHAPE_TAIL | Robust shape | C.SHAPE, C.TAIL, C.BOUND | M01–M06, S05 | §3.2 shape | A2–A4 | G |
| F027 | Boundary pile-up / censoring evidence | M_SHAPE_TAIL | Robust shape | C.SHAPE, C.TAIL, C.BOUND | M01–M06, S05 | §3.2 shape | A2–A4 | G |
| F028 | Quantile-estimation stability | G_ALIGN_TWIN_STAB | Reliability / codec sensitivity | C.UNC, G.ALIGN | B, K | — | — | G |
| F029 | Observed/missing support | M_MISS_UNC_PIT | Support/type | C.MISS | X01 | §3.4 26 | A4 | G |
| F030 | Co-missingness dependence | co_missing_corr | Missingness edges | C.MISS | D08 | §3.3 ch 13 | A4 | G |
| F031 | Missingness–target effect | R_UNIV_TARGET | Univariate target / local | C.MISS, C.Y | X02–X03 | §3.4 27 | A4 | G |
| F032 | Predictable missingness | R_UNIV_TARGET | Univariate target / local | C.MISS, C.Y | X02–X03 | §3.4 27 | A4 | G |
| F033 | Missingness–value association | R_UNIV_TARGET | Univariate target / local | C.MISS, C.Y | X02–X03 | §3.4 27 | A4 | G |
| F034 | Frequent missingness regimes | group / G_META_TASK | Structural metadata | C.TYPE, C.MISS, C.RULE | S02, Xg | §3.2 support | A4 | D2 |
| F035 | Categorical support spectrum | group / G_META_TASK | Structural metadata | C.TYPE, C.MISS, C.RULE | S02, Xg | §3.2 support | A4 | D2 |
| F036 | Category uncertainty | group / G_META_TASK | Structural metadata | C.TYPE, C.MISS, C.RULE | S02, Xg | §3.2 support | A4 | D2 |
| F037 | Categorical pair association | group / D_DEP_PREC | Nominal edges | C.RULE, edge 10–12 | D07, E08 | §3.3 ch 14 | B4 | D2 |
| F038 | Category–numeric association | group / D_DEP_PREC | Nominal edges | C.RULE, edge 10–12 | D07, E08 | §3.3 ch 14 | B4 | D2 |
| F039 | Class–category target table | class_col / R_UNIV_TARGET | Class-column | C.YF, §4.5 | R16, class | §3.2 target | A5–A6 | G |
| F040 | Regression–category target table | class_col / R_UNIV_TARGET | Class-column | C.YF, §4.5 | R16, class | §3.2 target | A5–A6 | G |
| F041 | Category combination support | group / D_DEP_PREC | Nominal edges | C.RULE, edge 10–12 | D07, E08 | §3.3 ch 14 | B4 | D2 |
| F042 | One-hot validity / parent map | same_nc_group | Structural metadata | C.RULE, edge 12 | S02, E08 | §3.3 ch 14 | A4 | G |
| F043 | Unknown-level stress profile | group / G_META_TASK | Structural metadata | C.TYPE, C.MISS, C.RULE | S02, Xg | §3.2 support | A4 | D2 |
| F044 | Categorical stability | group / G_META_TASK | Structural metadata | C.TYPE, C.MISS, C.RULE | S02, Xg | §3.2 support | A4 | D2 |
| F045 | Signed Pearson / rank correlation | r_lat / spearman_minus_pearson | Dependence edges | §4.4 ch 1–2 | E01 | §3.3 ch 0–3 | A1, B1 | R |
| F046 | Chatterjee dependence, both directions | chatterjee_xi_dir | Dependence edges | §4.4 ch 3 | E03–E04 | — | A6 | G |
| F047 | Distance correlation | nongauss_copula_excess | Dependence edges | §4.4 ch 4,7 | E05 | §3.3 ch 4 | B3 | D1 |
| F048 | Kernel dependence | nongauss_copula_excess | Dependence edges | §4.4 ch 4,7 | E05 | §3.3 ch 4 | B3 | D1 |
| F049 | Mutual-information estimate | nongauss_copula_excess | Dependence edges | §4.4 ch 4,7 | E05 | §3.3 ch 4 | B3 | D1 |
| F050 | Shrinkage partial correlations | partial_corr | Dependence edges | §4.4 ch 8 | D02 | §3.3 ch 1 | B4 | G |
| F051 | Copula checkerboard | nongauss_copula_excess / lambda_upper_excess / lambda_lower_excess | Pair grids / tails | §4.4 ch 5–7 | E05–E06 | §3.3 ch 5–8 | B2–B3 | G |
| F052 | Finite-threshold tail association | nongauss_copula_excess / lambda_upper_excess / lambda_lower_excess | Pair grids / tails | §4.4 ch 5–7 | E05–E06 | §3.3 ch 5–8 | B2–B3 | G |
| F053 | Asymmetric quadrant dependence | nongauss_copula_excess / lambda_upper_excess / lambda_lower_excess | Pair grids / tails | §4.4 ch 5–7 | E05–E06 | §3.3 ch 5–8 | B2–B3 | G |
| F054 | Dependence by regime | G_SPEC_MP / factor_resid_corr | Spectrum / geometry | C.DEP, G.SPEC | D04, D11 | §3.3 ch 12 | C4 | D3 |
| F055 | Nonlinear reconstruction gain | G_SPEC_MP / factor_resid_corr | Spectrum / geometry | C.DEP, G.SPEC | D04, D11 | §3.3 ch 12 | C4 | D3 |
| F056 | Dependence graph summaries | G_SPEC_MP / D_DEP_PREC | Local/redundancy / spectrum | C.DEP, G.SPEC | D02–D04, D10 | §3.3 ch 15; §3.4 16–20 | B4, C3 | G |
| F057 | Copula eigenstructure | G_SPEC_MP / D_DEP_PREC | Local/redundancy / spectrum | C.DEP, G.SPEC | D02–D04, D10 | §3.3 ch 15; §3.4 16–20 | B4, C3 | G |
| F058 | Conditional mutual-information proxy | nongauss_copula_excess | Dependence edges | §4.4 ch 4,7 | E05 | §3.3 ch 4 | B3 | D1 |
| F059 | Empirical characteristic-function sketch | R_UNIV_TARGET / G_SPEC_MP | Optional sketches / pair grids | C.Y, C.SPEC, G.SPEC | R11, D10 | §3.4 35 | B5 | D3 |
| F060 | Higher-order dependence sketch | R_UNIV_TARGET / G_SPEC_MP | Optional sketches / pair grids | C.Y, C.SPEC, G.SPEC | R11, D10 | §3.4 35 | B5 | D3 |
| F061 | Signed numeric target association | R_UNIV_TARGET / R_OOF_RESID_T03 | Univariate target | C.Y, C.YF | R01–R11 | §3.2 target | A5–A7 | G |
| F062 | Class-conditional feature profile | class_col / R_UNIV_TARGET | Class-column | C.YF, §4.5 | R16, class | §3.2 target | A5–A6 | G |
| F063 | Univariate OOF response curve | R_UNIV_TARGET / R_OOF_RESID_T03 | Univariate target | C.Y, C.YF | R01–R11 | §3.2 target | A5–A7 | G |
| F064 | Response curvature | R_UNIV_TARGET / R_OOF_RESID_T03 | Univariate target | C.Y, C.YF | R01–R11 | §3.2 target | A5–A7 | G |
| F065 | Monotonicity consistency | R_UNIV_TARGET / R_OOF_RESID_T03 | Univariate target | C.Y, C.YF | R01–R11 | §3.2 target | A5–A7 | G |
| F066 | Threshold / discontinuity profile | R_UNIV_TARGET / R_OOF_RESID_T03 | Univariate target | C.Y, C.YF | R01–R11 | §3.2 target | A5–A7 | G |
| F067 | Saturation and plateau profile | R_UNIV_TARGET / R_OOF_RESID_T03 | Univariate target | C.Y, C.YF | R01–R11 | §3.2 target | A5–A7 | G |
| F068 | Periodic target evidence | R_UNIV_TARGET / G_SPEC_MP | Optional sketches / pair grids | C.Y, C.SPEC, G.SPEC | R11, D10 | §3.4 35 | B5 | D3 |
| F069 | Nonmonotone association gap | R_UNIV_TARGET / R_OOF_RESID_T03 | Univariate target | C.Y, C.YF | R01–R11 | §3.2 target | A5–A7 | G |
| F070 | OOF drop-column gain | W_GBDT_WITNESS_CL | Driver/learner response | C.DRV | R14–R15 | §3.3 ch 11 | A7, C5 | G |
| F071 | Grouped permutation importance | W_GBDT_WITNESS_CL | Driver/learner response | C.DRV | R14–R15 | §3.3 ch 11 | A7, C5 | G |
| F072 | Conditional permutation importance | W_GBDT_WITNESS_CL | Driver/learner response | C.DRV | R14 | §3.3 ch 11 | C5 | D2 |
| F073 | Importance stability | W_GBDT_WITNESS_CL | Driver/learner response | C.DRV | R14–R15 | §3.3 ch 11 | A7, C5 | G |
| F074 | ALE main-effect curves | W_GBDT_WITNESS_CL | Driver/learner response | C.DRV | R14–R15 | §3.3 ch 11 | A7, C5 | G |
| F075 | Redundant driver groups | G_SPEC_MP / D_DEP_PREC | Local/redundancy / spectrum | C.DEP, G.SPEC | D02–D04, D10 | §3.3 ch 15; §3.4 16–20 | B4, C3 | G |
| F076 | Target conditional quantiles | R_OOF_RESID_T03 / G_TARGET_RESID_T03 | Residual response | C.RES, G.RESID | R11–R12, Y03–Y04 | §3.4 29–34 | A7, C5 | G |
| F077 | Class separation profile | class_col / R_UNIV_TARGET | Class-column | C.YF, §4.5 | R16, class | §3.2 target | A5–A6 | G |
| F078 | Calibration profile | G_WITNESS_FANOVA_CAL_LC | Global target / class-column | G.CAL, C.RES | Y05–Y06 | §3.4 28–29 | C5 | G |
| F079 | Target–feature nonlinear dependence | R_UNIV_TARGET / R_OOF_RESID_T03 | Univariate target | C.Y, C.YF | R01–R11 | §3.2 target | A5–A7 | G |
| F080 | Signal concentration | W_GBDT_WITNESS_CL | Driver/learner response | C.DRV | R14–R15 | §3.3 ch 11 | A7, C5 | G |
| F081 | Signed product-target moments | s10_prod_interaction_signed | Signed residual products | §4.4 target channels | R13, E07 | §3.3 ch 9 | B5 | G |
| F082 | Two-way target checkerboard | xor_quadrant_interaction_signed / gbdt_fanova_pair_energy | Pair-grid coefficients | §4.4 target channels | R13, E07 | §3.3 ch 10–11 | B5 | G |
| F083 | Interaction beyond additive effects | xor_quadrant_interaction_signed / gbdt_fanova_pair_energy | Pair-grid coefficients | §4.4 target channels | R13, E07 | §3.3 ch 10–11 | B5 | G |
| F084 | Depth-two tree gain | xor_quadrant_interaction_signed / gbdt_fanova_pair_energy | Pair-grid coefficients | §4.4 target channels | R13, E07 | §3.3 ch 10–11 | B5 | G |
| F085 | Pair residual-basis coefficients | s10_prod_interaction_signed | Signed residual products | §4.4 target channels | R13, E07 | §3.3 ch 9 | B5 | G |
| F086 | Conditional sign reversal | s10_prod_interaction_signed | Signed residual products | §4.4 target channels | R13, E07 | §3.3 ch 9 | B5 | G |
| F087 | Category–category interaction | gbdt_fanova_pair_energy / co_missing_corr | Typed interactions / stability | §4.4, C.DRV | E07–E08, B | §3.3 ch 11,13–14 | B5 | D2 |
| F088 | Category–numeric interaction | gbdt_fanova_pair_energy / co_missing_corr | Typed interactions / stability | §4.4, C.DRV | E07–E08, B | §3.3 ch 11,13–14 | B5 | D2 |
| F089 | Missingness interaction | gbdt_fanova_pair_energy / co_missing_corr | Typed interactions / stability | §4.4, C.DRV | E07–E08, B | §3.3 ch 11,13–14 | B5 | D2 |
| F090 | Interaction graph stability | gbdt_fanova_pair_energy / co_missing_corr | Typed interactions / stability | §4.4, C.DRV | E07–E08, B | §3.3 ch 11,13–14 | B5 | D2 |
| F091 | Triple/parity probe | R_UNIV_TARGET / G_SPEC_MP | Optional sketches / pair grids | C.Y, C.SPEC, G.SPEC | R11, D10 | §3.4 35 | B5 | D3 |
| F092 | Low-rank interaction spectrum | R_UNIV_TARGET / G_SPEC_MP | Optional sketches / pair grids | C.Y, C.SPEC, G.SPEC | R11, D10 | §3.4 35 | B5 | D3 |
| F093 | OOF residual distribution | R_OOF_RESID_T03 / G_TARGET_RESID_T03 | Residual response | C.RES, G.RESID | R11–R12, Y03–Y04 | §3.4 29–34 | A7, C5 | G |
| F094 | Residual dependence on columns | R_OOF_RESID_T03 / G_TARGET_RESID_T03 | Residual response | C.RES, G.RESID | R11–R12, Y03–Y04 | §3.4 29–34 | A7, C5 | G |
| F095 | Residual scale curve | R_OOF_RESID_T03 / G_TARGET_RESID_T03 | Residual response | C.RES, G.RESID | R11–R12, Y03–Y04 | §3.4 29–34 | A7, C5 | G |
| F096 | Heteroskedasticity score | R_OOF_RESID_T03 / G_TARGET_RESID_T03 | Residual response | C.RES, G.RESID | R11–R12, Y03–Y04 | §3.4 29–34 | A7, C5 | G |
| F097 | Landmark residual covariance | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F098 | Landmark disagreement | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F099 | Error overlap by region | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F100 | Bias–variance proxy | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F101 | Local noise proxy | R_OOF_RESID_T03 / G_TARGET_RESID_T03 | Residual response | C.RES, G.RESID | R11–R12, Y03–Y04 | §3.4 29–34 | A7, C5 | G |
| F102 | Standardized residual tail shape | R_OOF_RESID_T03 / G_TARGET_RESID_T03 | Residual response | C.RES, G.RESID | R11–R12, Y03–Y04 | §3.4 29–34 | A7, C5 | G |
| F103 | Residual class-probability vector | G_WITNESS_FANOVA_CAL_LC | Global target / class-column | G.CAL, C.RES | Y05–Y06 | §3.4 28–29 | C5 | G |
| F104 | Class confusion structure | G_WITNESS_FANOVA_CAL_LC | Global target / class-column | G.CAL, C.RES | Y05–Y06 | §3.4 28–29 | C5 | G |
| F105 | Label-conflict evidence | G_WITNESS_FANOVA_CAL_LC | Global target / class-column | G.CAL, C.RES | Y05–Y06 | §3.4 28–29 | C5 | G |
| F106 | Residual temporal structure | — | Optional group/time context | — | — | — | — | D4 |
| F107 | Residual environment structure | — | Optional group/time context | — | — | — | — | D4 |
| F108 | Analytic-base correction reliability | G_CLOSED_LOOP | Codec sensitivity | C.CODEC, G.LIFT, G.TWIN | M07, C, K | §4.2 | §5 | G |
| F109 | Extended landmark gain profile | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F110 | Sample-size learning curves | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F111 | Learning-curve slopes | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F112 | Regularization response | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F113 | Depth / interaction response | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F114 | Neighborhood-scale response | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F115 | Learner rank stability | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F116 | Optimism gap | G_WITNESS_FANOVA_CAL_LC | Landmark tokens / response | G.LM, G.LM2, G.LC, G.RES | L, B | §3.4 36–58 | C6 | G |
| F117 | Gradient mean sketch | — | Optional gradient sketches | §6.5 | §7.6 | §6 | §6 | D5 |
| F118 | Gradient second-moment sketch | — | Optional gradient sketches | §6.5 | §7.6 | §6 | §6 | D5 |
| F119 | Curvature / effective-dimension proxy | — | Optional gradient sketches | §6.5 | §7.6 | §6 | §6 | D5 |
| F120 | Influence concentration | W_GBDT_WITNESS_CL | Driver/learner response | C.DRV | R14 | §3.3 ch 11 | C5 | D2 |
| F121 | Neighbor-distance distribution | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F122 | Distance concentration | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F123 | Intrinsic-dimension estimate | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F124 | Local label disagreement | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F125 | Class-boundary fraction | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F126 | Margin distribution | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F127 | Neighborhood class entropy | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F128 | Class overlap | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F129 | Local linearity | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F130 | Clusterability / regime evidence | G_SPEC_MP / factor_resid_corr | Spectrum / geometry | C.DEP, G.SPEC | D04, D11 | §3.3 ch 12 | C4 | D3 |
| F131 | Coverage and density imbalance | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F132 | Constraint-respecting local connectivity | G_GEOM_PRIV | Local/redundancy / geometry | G.GEOM, G.CPLX | G, Y06–Y07 | §3.4 25 | C3–C4 | D3 |
| F133 | Feature-statistic bootstrap stability | G_ALIGN_TWIN_STAB | Reliability / codec sensitivity | C.UNC, G.ALIGN | B, K | — | — | G |
| F134 | Split sensitivity | G_ALIGN_TWIN_STAB | Reliability / codec sensitivity | C.UNC, G.ALIGN | B, K | — | — | G |
| F135 | Sample-size sensitivity of descriptors | G_ALIGN_TWIN_STAB | Reliability / codec sensitivity | C.UNC, G.ALIGN | B, K | — | — | G |
| F136 | Subgroup predictive structure | — | Optional group/time context | — | — | — | — | D4 |
| F137 | Subgroup driver structure | — | Optional group/time context | — | — | — | — | D4 |
| F138 | Environment distribution shift | — | Optional group/time context | — | — | — | — | D4 |
| F139 | Time trend and seasonality | — | Optional group/time context | — | — | — | — | D4 |
| F140 | Effective entity balance | — | Optional group/time context | — | — | — | — | D4 |
| F141 | Normalization sensitivity | G_ALIGN_TWIN_STAB | Reliability / codec sensitivity | C.UNC, G.ALIGN | B, K | — | — | G |
| F142 | Quantization sensitivity | G_ALIGN_TWIN_STAB | Reliability / codec sensitivity | C.UNC, G.ALIGN | B, K | — | — | G |
| F143 | Synthetic-size response | G_CLOSED_LOOP | Codec sensitivity | C.CODEC, G.LIFT, G.TWIN | M07, C, K | §4.2 | §5 | G |
| F144 | Marginal value of an extra probe | G_CLOSED_LOOP | Codec sensitivity | C.CODEC, G.LIFT, G.TWIN | M07, C, K | §4.2 | §5 | G |
