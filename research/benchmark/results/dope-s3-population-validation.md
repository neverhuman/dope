# DOPE all100 GPU research validation

All100 training-derived S3 validation lineages, four unchanged GPU research profiles, one fit seed; bounded training cohorts, no test certification

Single fit seed; three sample seeds. All four profiles remain visible. Retention is null-normalized TSTR/TRTR on training-derived validation. No fit uncertainty or paired superiority claim. Official tests sealed; MFS-v2/PTF-v1/release-safe scores null.

| Profile | Size | CatBoost informative lineages /100 | CatBoost median | Linear informative /100 | Linear median | MLP informative /100 | MLP median |
|---|---:|---:|---:|---:|---:|---:|---:|
| features12_steps512 | 1 | 97 | 0.757065 | 92 | 0.982217 | 75 | 0.954734 |
| features12_steps512 | 4 | 97 | 0.799674 | 92 | 0.984425 | 75 | 0.99886 |
| features12_steps2048 | 1 | 97 | 0.885562 | 92 | 0.983611 | 75 | 0.971284 |
| features12_steps2048 | 4 | 97 | 0.939876 | 92 | 0.994046 | 75 | 1.06112 |
| features24_steps512 | 1 | 93 | 0.757065 | 88 | 0.982217 | 72 | 0.961807 |
| features24_steps512 | 4 | 93 | 0.799674 | 88 | 0.986092 | 72 | 0.998482 |
| features24_steps2048 | 1 | 93 | 0.882459 | 88 | 0.976904 | 72 | 0.969295 |
| features24_steps2048 | 4 | 93 | 0.927634 | 88 | 0.989589 | 72 | 1.04829 |

Medians cover only complete informative three-sample groups; denominators and every failure remain explicit. They are descriptive aggregates, with no DOPE win or production family selection. Model plus projection bytes are charged. Bulk samples and runtime detail stay on scratch.
