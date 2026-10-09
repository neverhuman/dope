# BeyondArena training input preparation

This is a prospective, pure preparation helper. It opens no datasets, downloads
no files, reads no official TEST, fits no model and publishes no retention score.
The existing 73-family lock and retention publisher remain unchanged.

## Width rule

Freeze the rule before accessing a training view. For
`early_learning_predictors`, `lending_club`, `kick` and `wine_world_cost`, visit
raw inputs in their original column order. Estimate each input using the same
classification as `manifest.fit_projection`: a finite numeric input costs one
feature; a categorical input costs all distinct nonempty inner-fit levels.
Omit an all-missing input. Keep an entire input only when its estimated width
fits the remaining 2,000-feature budget and fewer than 12 raw inputs have
been selected. Continue to later inputs after a large
input is omitted. No categorical levels or encoded vectors are truncated.

The target is always retained once in its original position. The original
projector counts predictor features separately from the target: the feature
cap is 2,000, and the numeric output row can contain 2,001 values including its
mandatory target. The helper reports both widths. A width estimate is not an
artifact charge: an eventual generator must charge its complete model, stored
private projection, target mapping and framing. That byte charge remains null.

Only the inner-fit partition supplies the width estimate. Validation replays
the same column selection and fitted projection. Target values do not rank or
select inputs. The ordinary projector still validates target classes, fits its
training min/max and categories, and verifies its actual output width. Screening
does not turn an invalid or insufficient training view into a successful fit.

This source-order policy is a separate prospective rule. It does not claim
equivalence with the earlier private target-correlated top-12 screening proposal.
Its projection-only utility cost must be measured separately on inner validation
before making any evaluation claim; validation must not feed the selection rule.

## Complete TRAIN row groups

`split_training_groups` retains the original seed-1729 grouped 80/20 algorithm.
A full raw-row hash includes the target and every raw input; identical groups
remain together. For the three overlap families, removal may precede that split
only when an externally pinned, existing evaluator-owned receipt supplies the
TRAIN-intersection group hashes and exact source identities. Every matching
TRAIN occurrence is removed. The helper cannot construct that receipt by
reading held-out rows, and does not receive TEST-only values or indices.

## Finite catalog planning

The public lock includes `heart_disease_va_long_beach`, `qsar_biodeg` and
`hotel_booking_demand`. No absence from that lock is claimed. If a separate,
hash-bound supplied catalog omits one of those names, the helper reports only
that catalog's name-membership gap. The frozen successor order is SHA256 of
UTF-8 `source_family_id`, ascending, then family name. Candidates must use an exact recorded license string (MIT, CC-BY-4.0,
CC0-1.0 or LicenseRef-Public-Domain) and one of the recorded iid/grouped/temporal
strata. Candidates must be in
the licensed binary/regression lock and outside its original 12-family pilot.
They must also be named in the supplied catalog. Original requests are processed
in the order heart disease, QSAR, hotel booking; each successor is used once.

A successor is a separately declared cohort choice. It does not repair the
missing original family, preserve its task or stratum automatically, change the
original denominator, or prove a training export exists. The result records
stratum changes explicitly. With no supplied catalog the helper emits one
bounded input-gap status and makes no successor choices.

## Inputs still required

Each family needs a dedicated official TRAIN export with independent source,
rights, partition and target-position custody. The earlier named readiness
snapshot supplied none of those qualified export bindings. The overlap families
also need the existing evaluator intersection receipt. That is a finite named
packet observation, not a search proving absence elsewhere. Mixed TRAIN/TEST
Parquet decoding is not authorized by this helper.

No real family has been repaired by this source packet. Five-fit coverage,
retention, quality, MFS, privacy, production certification and release claims
remain unmeasured. The existing publisher continues to emit `not measured`.

## Verification

The eight synthetic controls compare estimates and grouping with AST-extracted
original pure functions, test the 12-input and full-input width caps and target position,
reject unbound or TEST-decoding overlap receipts, and check scoped catalog
choices and invalid rights/source identities. They open only code and synthetic
metadata; no fit, source dataset, sample or official test is used.

Run the focused controls after applying the overlay with:

```sh
python3 -B -m unittest research.benchmark.tests.test_beyondarena_train_screen -v
```
