# Checkpoint and figure inputs

These tables export recorded checkpoint metadata and previously published
official-training-derived validation utility. They do not execute fits, sampling,
auditors or official tests. The benchmark matrix and five-fit-seed coverage remain
incomplete; MFS-v2, PTF-v1, release and superiority claims remain null.

- `checkpoints.csv`: 1,992 logical checkpoint records with fit receipt paths,
  recorded artifact paths, hashes and charged bytes. 1,273 have measured CatBoost
  validation TSTR MSE at 4n, taking the median of sample seeds 101/211/307. Missing
  measurements retain a reason. Physical aliases are identified by fit receipt
  hash; additional table rows do not imply additional compute.
- `figure-kpis.csv`: 3,600 groups from four separately identified committed
  publications. 3,408 groups have three OK sample cells. Losses are regression
  MSE in each frozen numeric target representation. Uninformative groups retain
  measured losses and null retention; missing groups receive no partial median.
  Each source hash and complete cell identity links back to the per-seed receipts
  in the committed source publication. Do not pool panels with different scopes
  or rank raw MSE across datasets.
- `forest-key-errors.csv`: a separate descriptive fragment with 37 closed cell
  rows covering 33 recorded Forest-Flow checkpoints. Its common error columns use n/seed101 where measured,
  which differs from the first table's 4n/three-seed schedule. Native author mean
  R² is reported separately. Incomplete lifecycle cells remain incomplete.
- `tabddpm-historical-v8-checkpoints.csv`: 43 historical recorded provider
  inventories, not a reconstruction of all 63 current providers. Five retained
  native R² values remain scores; error losses and common/privacy outcomes stay
  null where unmeasured. `tabddpm-current-operation-receipts.csv` preserves 20
  current operations. Current aggregate counts are 63 fit providers, 49 native
  completions and 21 matched batches: different units, never substituted for
  one another or for a complete 100-lineage comparison. The exact current
  inventory gap is scoped to the inspected local custody/readset.
- `publication.json` and its schema bind tables, source publications and immutable
  private metadata inputs. Current model/sample payloads were not read or
  rehashed. No present retention or availability claim follows from a recorded
  artifact path. Native KPI values remain method-specific, without cross-method
  rankings or a change to native selection.

Regenerate from the pinned metadata inputs and committed source publications:

```sh
python3 -B -m research.benchmark.publish_checkpoint_index
python3 -B -m unittest research.benchmark.tests.test_checkpoint_index -v
```

The private full checkpoint index/readset remain referenced by immutable digests;
weights, source rows, samples and detailed logs are not committed. TabDDPM's full
checkpoint inventory, TabSyn's common validation metrics, Forest-Flow's complete
panel, and the common expanded diagnostic matrix still require measured evidence.
This index does not establish an IEEE-ready complete comparison.
