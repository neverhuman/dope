# DOPE versus compact study references: paired pilot validation

Independent marginals and Chow-Liu use study-owned implementations. Their tuned configurations maximize each method's frozen held-out mean log density. Native KPI values are comparable within a method and dataset only. The shared outcome is CatBoost TSTR/TRTR retention at fit seed 23 and three paired sample seeds.

| Dataset | Method | Config | Size | Native log density | Reference retention | DOPE retention | Paired DOPE − reference | Reference bytes | Copy-gate failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Adult | independent_marginals | default | 1n | 272.129579 | 0.0005 | 0.8275 | +0.8332 | 18,988 | 0/3 |
| Adult | independent_marginals | default | 4n | 272.129579 | -0.0002 | 0.8293 | +0.8295 | 18,988 | 0/3 |
| Adult | independent_marginals | tuned | 1n | 344.510452 | -0.0972 | 0.8275 | +0.9171 | 26,884 | 0/3 |
| Adult | independent_marginals | tuned | 4n | 344.510452 | -0.0283 | 0.8293 | +0.8567 | 26,884 | 0/3 |
| Adult | Chow-Liu | default | 1n | -3.562567 | 0.7253 | 0.8275 | +0.0997 | 26,641 | 0/3 |
| Adult | Chow-Liu | default | 4n | -3.562567 | 0.7275 | 0.8293 | +0.1017 | 26,641 | 0/3 |
| Adult | Chow-Liu | tuned | 1n | 1.758837 | 0.6585 | 0.8275 | +0.1690 | 108,760 | 1/3 |
| Adult | Chow-Liu | tuned | 4n | 1.758837 | 0.6646 | 0.8293 | +0.1638 | 108,760 | 3/3 |
| California | independent_marginals | default | 1n | 12.368613 | 0.0161 | 0.7008 | +0.6864 | 4,296 | 0/3 |
| California | independent_marginals | default | 4n | 12.368613 | -0.0095 | 0.6948 | +0.7043 | 4,296 | 0/3 |
| California | independent_marginals | tuned | 1n | 13.539174 | 0.0081 | 0.7008 | +0.6927 | 6,590 | 0/3 |
| California | independent_marginals | tuned | 4n | 13.539174 | 0.0069 | 0.6948 | +0.6858 | 6,590 | 0/3 |
| California | Chow-Liu | default | 1n | 11.360567 | 0.6464 | 0.7008 | +0.0533 | 11,783 | 0/3 |
| California | Chow-Liu | default | 4n | 11.360567 | 0.6376 | 0.6948 | +0.0552 | 11,783 | 0/3 |
| California | Chow-Liu | tuned | 1n | 16.309525 | 0.7104 | 0.7008 | -0.0136 | 164,445 | 0/3 |
| California | Chow-Liu | tuned | 4n | 16.309525 | 0.7167 | 0.6948 | -0.0220 | 164,445 | 0/3 |
| News | independent_marginals | default | 1n | 103.644561 | -109.8954 | 0.3287 | +110.2158 | 25,937 | 0/3 |
| News | independent_marginals | default | 4n | 103.644561 | -166.4777 | 0.3109 | +166.8138 | 25,937 | 0/3 |
| News | independent_marginals | tuned | 1n | 125.794232 | -21.8700 | 0.3287 | +22.1905 | 37,996 | 0/3 |
| News | independent_marginals | tuned | 4n | 125.794232 | -65.6836 | 0.3109 | +65.9945 | 37,996 | 0/3 |
| News | Chow-Liu | default | 1n | 60.817854 | -541.1017 | 0.3287 | +541.4222 | 71,671 | 0/3 |
| News | Chow-Liu | default | 4n | 60.817854 | -530.3163 | 0.3109 | +530.6272 | 71,671 | 0/3 |
| News | Chow-Liu | tuned | 1n | 99.114433 | -98.4410 | 0.3287 | +98.7770 | 867,505 | 0/3 |
| News | Chow-Liu | tuned | 4n | 99.114433 | -87.0985 | 0.3109 | +87.4346 | 867,505 | 0/3 |

News retention has a small real-over-null utility denominator; large negative values are measured and are not clipped. JSON retains every seed, artifact charge, row-copy count, native selection, and receipt hash. Official tests remain sealed; MFS-v2, PTF-v1, and production certification are null. This three-dataset, one-fit-seed pilot is not a paper superiority result.
