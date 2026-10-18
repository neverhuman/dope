# Cross-system bike-demand observation

This separate, limited observation predicts aggregate bike rentals per active
hour. It fits DOPE only on the 2011 Capital Bikeshare source and evaluates
synthetic-trained and real-trained Ridge auditors on date-held-out Seoul
Bike Sharing rows. The source-system holdout is real; independent-curator
replication and TDL qualification are not established.

Both UCI source records declare CC BY 4.0. The source identifiers, attribution,
license assertions, archive/member SHA256 pins, population, period and holdout
meaning are recorded in `source-manifest.json`. Raw archives are not committed.
The fields are hourly system aggregates, not rider records. The model has no
healthcare endpoint or healthcare inference.

The split policy and recipe were frozen in `protocol.json` before either source
archive was acquired or evaluated. Service dates are indivisible entities. For
Capital Bikeshare, 2011 dates are ordered and split 75/25 into generator fit
and selection. For Seoul, active service dates are split 50/20/30 into the
real auditor fit, reserved selection and source-holdout test. The reserved
selection rows are not consumed for model or candidate choice: the candidate
and Ridge recipe are each a single fixed choice. The source year 2012 is
excluded from Washington.

The common predictor view uses hour, weekday, month, holiday, working-day,
temperature, humidity and wind speed. Count-derived predictors and source row
IDs are excluded. Washington's published normalized weather values are mapped
to Celsius, percent and metres per second. Seoul non-operating hours are
excluded by the frozen rule. The target remains the original hourly count.
The transfer reducer uses row-weighted squared error; its real-gain lower bound
resamples held-out service dates as clusters.

## Reproduce

Use Python 3.12.3 with NumPy 1.26.4, SciPy 1.16.3, scikit-learn 1.7.2 and
threadpoolctl 3.6.0, plus the repository environment. Download the two UCI
archives to persistent `target/` storage, then run from the repository root:

```sh
mkdir -p target/bikeshare-observational/source
curl -fL 'https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip' -o target/bikeshare-observational/source/capital-bike.zip
curl -fL 'https://archive.ics.uci.edu/static/public/560/seoul%2Bbike%2Bsharing%2Bdemand.zip' -o target/bikeshare-observational/source/seoul-bike.zip
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo build --locked --bin dope-kernel
PYTHONPATH=. python3 -B examples/public-regression/bikeshare-observational/run.py \
  --capital-zip target/bikeshare-observational/source/capital-bike.zip \
  --seoul-zip target/bikeshare-observational/source/seoul-bike.zip \
  --binary "$PWD/target/debug/dope-kernel" \
  --out target/bikeshare-observational/observation-1
```

The runner verifies both pinned archives and exact archive members. It compiles
the same prepared Washington fit view under 16 KiB and 32 KiB complete-byte
caps, reloads and samples each raw artifact in an isolated worker at `n` and
`4n`, and checks native sample digests. Ridge is fit separately on Seoul's
training rows and each synthetic sample; all arms score the same unseen Seoul
dates. Paired row losses enter the TDL transfer reducer and the resulting cell
enters the TDL aggregate reducer. The driver diagnostic compares
single-feature permutation-loss increases and coefficient signs on held-out
Seoul rows; interaction evidence is explicitly unavailable.

`receipt.json` includes source and split hashes, actual fit/select/test rows,
artifact/sample bytes, consumed rows, elapsed time, process RSS upper bound,
losses, reducers, drivers, dependencies, failed-step states and limitations.
The measured source holdout is an observational example, not a release result,
privacy guarantee, official TDL panel result, or independent-curator
confirmation. A weak reference, failed gate, or unavailable driver stays
visible in the receipt.
