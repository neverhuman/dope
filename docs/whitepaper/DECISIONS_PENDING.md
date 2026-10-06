# Pending paper decisions

The byline is selected. Official tests stay sealed and final evaluation is
not authorized. This file does not read `test.csv` and does not invent
replacement values.

## Author byline

Jepson Taylor and Alton Alexander, NEVERHUMAN Research. The running header
is Taylor and Alexander. `docs/whitepaper/scripts/build_pdf.sh` writes the
anonymous PDF by substituting that byline and `\markboth{Taylor and Alexander}`
only, then deletes the temporary tex. The method name DOPE in the prose is
the generator name in both PDFs.

| Build | File | Byline and running header | Body |
| --- | --- | --- | --- |
| Journal | `docs/whitepaper/dope-mfs.pdf` | Jepson Taylor and Alton Alexander, NEVERHUMAN Research | Unchanged manuscript |
| Alternate | `docs/whitepaper/dope-mfs-anonymous.pdf` | Anonymous | Same sentences and the same `generated/numbers.tex` macros |

## Sealed test split

The displayed evaluation uses the training-derived validation cut: a grouped
80/20 of the official training rows, seed `\SplitSeed` (1729). Workers hold
`train.csv` and `validation.csv`. Retention is the share of real-versus-null
improvement on real validation rows. Paired tests, Holm adjustments,
rank-biserial correlations, the Friedman--Nemenyi diagram, fidelity, and the
suffix and threshold checks are computed on that same validation panel.

### If the sealed test split stays disallowed

Every macro currently in `docs/whitepaper/generated/numbers.tex` keeps the
validation value already printed there. No test-split column is added.
Master Fitness v2, PTF-v1, and release-safe L3 stay null. `\TabSynPanel`,
`\TabDDPMPanel`, `\PrivacyAttackPanel`, `\FitSeedVariance`,
`\FullAblationGrid`, and `\CampaignElapsed` stay `not measured`.

### If final evaluation on the sealed test split is allowed

The quantities below are functions of the validation rows. A test-split
final evaluation would recompute them. The replacement numbers are
unmeasured.

Density CatBoost retention and its paired tests, n = `\NDopeCb` (97):

| Macro | Validation value | Quantity |
| --- | --- | --- |
| `\MedDopeCb`, `\LoDopeCb`, `\HiDopeCb` | 0.940, 0.914, 0.976 | DOPE median and interval |
| `\MedGaussCb`, `\LoGaussCb`, `\HiGaussCb` | 0.656, 0.431, 0.803 | Gaussian copula median and interval |
| `\MedChowCb`, `\LoChowCb`, `\HiChowCb` | 0.586, 0.465, 0.637 | Chow--Liu median and interval |
| `\MedIndCb`, `\LoIndCb`, `\HiIndCb` | −0.012, −0.020, −0.00649 | Independent-marginals median and interval |
| `\DiffGaussCb`, `\LoDiffGaussCb`, `\HiDiffGaussCb` | 0.224, 0.149, 0.290 | Paired difference versus the copula |
| `\HolmGaussCb`, `\HolmChowCb`, `\HolmIndCb` | 1.35e−13, 4.85e−16, 1.17e−16 | Holm Wilcoxon |
| `\SignHolmGaussCb`, `\SignHolmChowCb`, `\SignHolmIndCb` | 2.15e−15, 6.72e−24, 4.95e−27 | Holm sign tests |
| `\RGaussCb`, `\RChowCb`, `\RIndCb` | 0.890, 0.978, 0.999 | Rank-biserial |
| `\FriedmanPCb`, `\FriedmanCdCb`, `\FriedmanNCb` | 2.71e−44, 0.476, 97 | Friedman--Nemenyi |

Linear auditor on the same density lineages:

| Macro | Validation value | Quantity |
| --- | --- | --- |
| `\HolmLinGauss` | 0.373 | Holm Wilcoxon versus the copula |
| `\DiffLinGauss`, `\LoDiffLinGauss`, `\HiDiffLinGauss` | 0.00366, −0.00438, 0.012 | Paired difference and interval |
| `\RLinGauss` | 0.107 | Rank-biserial |
| `\FriedmanCdLin` | 0.489 | Nemenyi critical distance |
| `\RankDopeLin`, `\RankGaussLin` | 1.533, 1.598 | Average ranks |

Neural block, n = `\NNeuCb` (21), unpooled:

| Macro | Validation value | Quantity |
| --- | --- | --- |
| `\MedDopeNeuCb`, `\LoDopeNeuCb`, `\HiDopeNeuCb` | 0.953, 0.896, 0.990 | DOPE CatBoost median and interval |
| `\MedCtganCb` | −0.081 | CTGAN median |
| `\MedTvaeCb` | 0.669 | TVAE median |

Forest-Flow confirmation, n = `\NForestCb` (6), unpooled. These are
retention scores. The byte charges in the next section stay.

| Macro | Validation value | Quantity |
| --- | --- | --- |
| `\MedForestCb`, `\LoForestCb`, `\HiForestCb` | 1.003, 0.926, 1.026 | Forest-Flow CatBoost median and interval |
| `\MedDopeForestCb` | 0.940 | DOPE median on those lineages |

Suffix check on the validation panel (lineages whose real-versus-null gap
clears the paper's floor):

| Macro | Validation value | Quantity |
| --- | --- | --- |
| `\NSufCb`, `\MedSufACb`, `\MedSufBCb`, `\DiffSufCb` | 68, 0.926, 0.970, 0.034 | CatBoost |
| `\NSufLin`, `\MedSufALin`, `\MedSufBLin`, `\DiffSufLin` | 64, 0.990, 0.987, −0.000283 | Linear |
| `\NSufMlp`, `\MedSufAMlp`, `\MedSufBMlp`, `\DiffSufMlp` | 52, 1.065, 1.057, 0.00451 | MLP |

Fidelity of synthetic rows against the real validation rows:

| Macro | Validation value | Quantity |
| --- | --- | --- |
| `\CtwoDope`, `\CtwoLo`, `\CtwoHi`, `\CtwoN` | 0.517, 0.504, 0.532, 98 | Two-sample AUC |
| `\KsDope`, `\KsLo`, `\KsHi`, `\KsN` | 0.114, 0.103, 0.135, 98 | Mean marginal KS |
| `\CorrDope`, `\CorrLo`, `\CorrHi`, `\CorrN` | 0.953, 0.945, 0.961, 98 | Pairwise correlation |

`\ThrSentence` is the validation-threshold check: thresholds 0, 0.001, and
0.05 keep the CatBoost median and count; at 0.1 the count is 96 and the
median is 0.942. That sentence would be recomputed with the test-split
retentions.

`\NDopeCb`, `\FriedmanNCb`, `\NNeuCb`, `\NForestCb`, and the suffix and
fidelity counts are counts of validation lineages with a finite result. A
test-split rerun can change the count when a lineage lacks a finite test
pair. Those counts are unmeasured on the test split.

### Numbers that stay under either decision

These are artifact sizes, fit receipts, catalog facts, or the fit's own
training monitor. Authorizing a later test-split score does not refit the
generators and does not rewrite these values.

Charged artifact bytes, including the 10,240-byte cap comparisons:

| Macro | Value |
| --- | --- |
| `\DopeByteLo`, `\DopeByteMedian`, `\DopeByteHi`, `\DopeWithinCap` | 758; 1,895.5; 22,233; 98/100 |
| `\MissA`, `\MissABytes`, `\MissB`, `\MissBBytes` | the two named over-cap DOPE lineages, 22,233 and 19,019 bytes |
| `\ForestByteLo`, `\ForestByteMedian`, `\ForestByteHi` | 115,202,253; 325,848,913.5; 1,085,155,062 |
| `\DopeForestByteMedian` | 2,067 |
| `\GaussByteMedian`, `\GaussWithin` | 5,464.5; 62/100 |
| `\ChowByteMedian`, `\ChowWithin` | 29,865; 31/100 |
| `\IndByteMedian`, `\IndWithin` | 2,549.5; 94/100 |
| `\CtganByteMedian`, `\CtganWithin` | 741,834; 0/22 |
| `\TvaeByteMedian`, `\TvaeWithin` | 317,141.5; 0/22 |

Fit and replay receipts:

| Macro | Value |
| --- | --- |
| `\ForestFitCore`, `\ForestSampleOp`, `\ForestGpuFits` | 953; 8,801; 12 |
| `\ReplayMedA`, `\ReplayMedB`, `\ReplayN` | 4.56; 14.4; 100 |
| `\LossTrainStart`, `\LossValStart` | 0.321; 0.271 |
| `\LossTrainEndA`, `\LossValEndA` | 0.00274; 0.00443 |
| `\LossTrainEndB`, `\LossValEndB` | 0.000763; 0.00181 |

The loss macros are the displayed fit's training loss and its
training-derived validation loss. They are monitors from the fit, separate
from a later official-test score.

Catalog and design counts:

| Macro | Value |
| --- | --- |
| `\SplitSeed` | 1729 |
| `\CatalogSha` | ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120 |
| `\EligibleN`, `\PreparedN`, `\ExcludedN` | 104; 100; 4 |
| `\UnknownRights` | 3,755 |
| `\ArfWatchClosed`, `\ArfWatchOk`, `\ArfWatchPlanned` | 799; 799; 800 |
| `\BeyondPrepared`, `\BeyondFailed`, `\BeyondFamilies` | 5; 7; 12 |
| `\BeyondFeatureRange`, `\BeyondOverlap`, `\BeyondInventory` | 4; 3; 142 |
| `\BeyondRevision` | 2ecfe882ccfb814fc27c4de10a64ceefd5d7655c |
| `\BeyondElapseA`, `\BeyondElapseB` | 5.00; 14.0 |
| `\NotInfName` | the named non-finite lineage label |

Prose constants that are the displayed generator, not a split score: at most
12 inputs, hidden width 16, tanh, refit linear readout, AdamW, 2,048 steps,
learning rate 0.002, gradient clip 5.0, no embedding, fit seed 11, and the
10,240-byte cap.

Nulls and unmeasured panels stay null or `not measured` under both
decisions. A test-split score does not create MFS-v2, PTF-v1, a release-safe
L3 claim, a privacy-attack result, a TabSyn panel, a 100-lineage TabDDPM
panel, a fit-seed variance, or the full ablation grid.

### Open pull request #133

Head `5c7b3e231159b2be9e236695a60aa0840c5ec2c7` adds lineage
wins/ties/losses. Those counts are signs of the existing validation paired
differences, DOPE minus the baseline. They are not in the merged PDF on
this branch. If that pull request merges and the test split stays sealed,
the counts below remain the validation counts. If final test-split
evaluation is allowed, each triple would be recomputed, and the test-split
triples are unmeasured.

| Comparison | Validation W/T/L |
| --- | --- |
| Density CatBoost, Gaussian copula | 86/0/11 |
| Density linear, Gaussian copula | 49/0/43 |
| Density MLP, Gaussian copula | 65/0/10 |
| Neural CatBoost, TVAE | 19/0/2 |
| Neural linear, TVAE | 20/0/1 |
| Neural MLP, TVAE | 17/0/2 |
| Forest CatBoost, Forest-Flow | 1/0/5 |
| Forest linear, Forest-Flow | 3/0/2 |
| Forest MLP, Forest-Flow | 0/0/4 |

The nine density Holm tests, the neural family, and the Forest-Flow family
stay the families already defined for those blocks. This list does not
shrink a family and does not pool the blocks.
