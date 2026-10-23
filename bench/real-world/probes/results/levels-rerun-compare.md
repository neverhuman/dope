| Table | identical programs (low/med/high/ultra) | robust retention, first run | robust retention, second run | largest change |
|---|---|---|---|---:|
| 197_cpu_act | yes/yes/yes/yes | 0.960 / 0.650 / 0.646 / 0.642 | 0.950 / 0.650 / 0.646 / 0.642 | 0.010 |
| 218_house_8L | no/no/no/no | 0.986 / 0.943 / 0.994 / 0.977 | 0.993 / 0.971 / 0.994 / 0.977 | 0.028 |
| 225_puma8NH | yes/yes/no/yes | 0.984 / 0.985 / 0.962 / 0.978 | 0.994 / 0.994 / 0.990 / 0.964 | 0.028 |
| 344_mv | no/no/no/no | 0.999 / 0.998 / 0.999 / 0.999 | 0.998 / 0.999 / 0.998 / 0.999 | 0.000 |
| 574_house_16H | yes/yes/no/no | 0.897 / 0.955 / 0.738 / 0.849 | 0.897 / 0.852 / 0.562 / 0.914 | 0.176 |
| adult | yes/yes/yes/yes | 0.866 / 0.868 / 0.857 / 0.874 | 0.866 / 0.868 / 0.857 / 0.874 | 0.000 |
| car_evaluation | yes/yes/yes/yes | 0.978 / 0.978 / 0.975 / 0.974 | 0.963 / 0.978 / 0.975 / 0.974 | 0.015 |
| churn | yes/yes/yes/yes | 0.801 / 0.854 / 0.859 / 0.845 | 0.801 / 0.854 / 0.859 / 0.845 | 0.000 |
| mushroom | yes/yes/yes/yes | 0.726 / 0.729 / 0.729 / 0.730 | 0.726 / 0.729 / 0.720 / 0.730 | 0.008 |

identical programs: 25/36; largest per-cell change 0.176
over the 9 common tables, median robust retention low/med/high/ultra: first 0.960 / 0.943 / 0.859 / 0.874, second 0.950 / 0.868 / 0.859 / 0.914
carry (margin 0.005) mean: first 0.911 / 0.918 / 0.925 / 0.924, second 0.910 / 0.905 / 0.912 / 0.908; drops > 0.02: independent 4 vs 5, carry 0 vs 2
