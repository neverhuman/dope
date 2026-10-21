| Table | Encoding · backend | Worst-auditor retention | catboost | linear | mlp | State bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| abalone | one-hot · robust · chow_liu | 0.854 ± 0.057 | 0.903 | 0.962 | 0.972 | 791 |
| abalone | target-ordered · robust · chow_liu | 0.854 ± 0.022 | 0.879 | 0.957 | 0.973 | 793 |
| abalone | auto (one-hot if at most 4 levels) · robust · chow_liu | 0.854 ± 0.057 | 0.903 | 0.962 | 0.972 | 791 |
| abalone | target-ordered · min-max · chow_liu | 0.833 ± 0.018 | 0.855 | 0.967 | 0.947 | 797 |
| abalone | target-ordered · quantile · chow_liu | 0.846 ± 0.029 | 0.846 | 0.994 | 0.983 | 7,951 |
| abalone | target-ordered · robust · default search | 0.850 ± 0.116 | 0.850 | 0.936 | 1.006 | 859 |
| abalone | target-ordered · robust · sparse_gaussian_copula | 0.545 ± 0.132 | 0.638 | 0.592 | 0.715 | 1,388 |
| abalone | target-ordered · robust · compact_neural_residual | 0.897 ± 0.020 | 0.913 | 0.988 | 1.029 | 1,139 |
| abalone | target-ordered · robust · transfer-in-the-loop (K=3) | 0.885 ± 0.024 | 0.902 | 0.981 | 1.018 | 998 |
| adult | one-hot · robust · chow_liu | 0.762 ± 0.011 | 0.762 | 0.871 | 0.844 | 2,051 |
| adult | target-ordered · robust · chow_liu | 0.794 ± 0.008 | 0.794 | 0.888 | 0.887 | 2,016 |
| adult | auto (one-hot if at most 4 levels) · robust · chow_liu | 0.794 ± 0.008 | 0.794 | 0.888 | 0.887 | 2,016 |
| adult | target-ordered · min-max · chow_liu | 0.800 ± 0.008 | 0.800 | 0.949 | 0.902 | 1,722 |
| adult | target-ordered · quantile · chow_liu | 0.749 ± 0.010 | 0.749 | 0.842 | 0.835 | 7,854 |
| adult | target-ordered · robust · default search | 0.826 ± 0.024 | 0.826 | 0.873 | 0.903 | 2,335 |
| adult | target-ordered · robust · sparse_gaussian_copula | 0.832 ± 0.026 | 0.843 | 0.887 | 0.878 | 2,834 |
| adult | target-ordered · robust · compact_neural_residual | 0.803 ± 0.007 | 0.803 | 0.881 | 0.890 | 2,694 |
| adult | target-ordered · robust · transfer-in-the-loop (K=3) | 0.827 ± 0.028 | 0.838 | 0.878 | 0.873 | 2,672 |
