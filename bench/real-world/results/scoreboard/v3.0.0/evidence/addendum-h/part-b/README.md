# Development comparison on seeds 13–16

Part (b) of [Addendum H](../ADDENDUM_H.md) repeats the paired comparison of signed DOPE v3.0.0 with library-default ARF, Gaussian copula, TVAE and CTGAN. The generators share the same fitting and test splits. The signed DOPE score and the five reference rows were reused from the already completed learned-tier evaluation. The third-party generators saw these splits for the first time. This comparison has no role in model selection or release qualification.

| Bound evidence | SHA-256 |
| --- | --- |
| Run manifest | `80d34c07ff4ed8223670530904badce0407ea0ac46fd3927ee7a7aacf382a741` |
| Schedule | `534d3da82eda377892dac1c2712df0035980868de2b9d5f9eeda4acb2ee53062` |
| Frozen real reference | `33f948ff49dd79c3e9ebd24b63da7d28c798a47a4863777c4a0a4ab465d87e00` |
| Native scoring executable | `ea855a89d4fad31ae1d197f3f2a1045a84daf98ec6b527a8e03c71b3653c2863` |
| Bound worker and score artifacts | `0cb5da2ecd71aef4adafb8e9be0331489eb6b0a6570086d665c5cef8d9ffefa8` |
| Logged reduction result | `99acb03cb209602a91a08db1e2bcb414f203ce4e27bff22a9c018402ef2899a4` |

The manifest was logged before launch at 15:47:31Z on 2026-11-24. The worker artifacts were bound at 17:58:04Z and the reduction logged at 17:58:44Z. The independent replay rehashed all 65 bound input files, reproduced the native primary and all eight native comparisons byte for byte and recomputed the four-way Holm adjustment from the signed engine's pairs. The disclosed research-finalist comparisons are outside the shipped-engine tables.

`summary.csv` takes each arm's 4× and 1× Transfer Scores and informative headline counts from the native primary. Scores are `100 × (1 + TRG)`. Paired differences and intervals, Holm-adjusted probabilities and contract decisions come from the corresponding comparisons and family receipt. Task guard values and thresholds are in Transfer Score points. The paper uses the report, part (a) and part (b) as separate split sets.
