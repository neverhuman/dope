# Adult / California / News validation interim

Training-derived validation only. The official test partitions remain sealed. PTF-v1 and MFS-v2 are null because the product and privacy gates are incomplete.

Median CatBoost null-normalized TSTR/TRTR retention over two fit and two sample seeds per dataset. Native tuning used each baseline's own held-out mean log density. The DOPE q8 rows are symbolic compiler results; the separate GPU-trained neural round is still a research candidate.

| Dataset | Method | Configuration | n | 4n | Charged bytes |
| --- | --- | --- | ---: | ---: | ---: |
| Adult | dope | q8_symbolic | 0.8306 | 0.8298 | 6337–6341 raw; 4378–4382 restricted packed |
| Adult | Chow-Liu | default | 0.7267 | 0.7250 | 26641 |
| Adult | Chow-Liu | tuned | 0.6639 | 0.6680 | 108760 |
| Adult | independent_marginals | default | 0.0270 | -0.0113 | 18988 |
| Adult | independent_marginals | tuned | -0.0085 | -0.0184 | 26884 |
| California | dope | q8_symbolic | 0.7057 | 0.7002 | 2524–2525 raw; 1891–1892 restricted packed |
| California | Chow-Liu | default | 0.6481 | 0.6401 | 11783 |
| California | Chow-Liu | tuned | 0.7129 | 0.7161 | 164445 |
| California | independent_marginals | default | 0.0032 | -0.0132 | 4296 |
| California | independent_marginals | tuned | 0.0034 | 0.0018 | 6590 |
| News | dope | q8_symbolic | -6.0209 | -6.1417 | 13594–13603 raw; 9086–9095 restricted packed |
| News | Chow-Liu | default | -522.2033 | -510.2339 | 71671 |
| News | Chow-Liu | tuned | -98.7743 | -90.5594 | 867505 |
| News | independent_marginals | default | -100.9688 | -186.7531 | 25937 |
| News | independent_marginals | tuned | -21.8260 | -111.3700 | 37996 |

The packed DOPE projection maps remain restricted because they contain source names and extrema. Most baseline artifacts exceed the 10,240-byte L3 cap. News has a small TRTR-to-null improvement, so its retention ratio is unstable.

## Baseline native validation KPI

Each value is comparable only with the same method and dataset. Values from different methods are not ranked.

| Dataset | Method | Default mean log density | Tuned mean log density |
| --- | --- | ---: | ---: |
| Adult | Chow-Liu | -3.5626 | 1.7588 |
| Adult | independent_marginals | 272.1296 | 344.5105 |
| California | Chow-Liu | 11.3606 | 16.3095 |
| California | independent_marginals | 12.3686 | 13.5392 |
| News | Chow-Liu | 60.8179 | 99.1144 |
| News | independent_marginals | 103.6446 | 125.7942 |

## One-seed architecture research

These are validation measurements at fit seed 11 and sample seed 101. They are not the paired four-replicate comparison above.

| Dataset | Family | Candidate | Target weight | n | 4n | Restricted packed bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Adult | symbolic_refinement | compact_neural_residual_symbolic | 4 | 0.8348 | 0.8309 | 4382 |
| Adult | symbolic_refinement | symbolic_autoregressive_residual | 4 | 0.8306 | 0.8324 | 4420 |
| California | symbolic_refinement | compact_neural_residual_symbolic | 4 | 0.7096 | 0.7044 | 1891 |
| California | gpu_neural | micro_tvae_4_16 | 2 | 0.0602 | 0.0640 | 5832 |
| California | gpu_neural | micro_tvae_4_16 | 4 | 0.1895 | 0.1913 | 5832 |
| California | gpu_neural | micro_tvae_8_24 | 2 | -0.2808 | -0.2806 | 7082 |
| California | gpu_neural | micro_tvae_8_24 | 4 | -0.2841 | -0.2870 | 7086 |
| California | symbolic_refinement | symbolic_autoregressive_residual | 4 | 0.7107 | 0.6979 | 1892 |
| News | symbolic_refinement | compact_neural_residual_symbolic | 4 | -9.8032 | -9.6060 | 9086 |
| News | symbolic_refinement | symbolic_autoregressive_residual | 4 | 1.0724 | 1.0638 | 9201 |

Eight additional GPU-trained neural fits on Adult and News failed the 10,240-byte cap before sampling. Their encoded candidate sizes and receipt hashes are in the JSON. California's best GPU-trained neural candidate trails the symbolic candidates on validation utility. Changing target weight in the q8 symbolic family left its model artifact unchanged; those refits count as research cost without a selection gain.

The machine-readable JSON contains all 120 cell metrics, 24 paired comparisons, 20 architecture metrics, eight neural failures, byte charges, and hashes of immutable scratch receipts.
