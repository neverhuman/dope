# CDTD author-source admission audit

CDTD remains `source_audit_pending`. This audit admits no runtime, model,
fit/sample job or generator search. Official tests stay sealed;
MFS-v2/PTF-v1/release/superiority stay null, and no DOPE win is counted.

## Source and rights

The author [replication repository](https://github.com/muellermarkus/cdtd) is
pinned at `673ff190e4baed651809db6b18174ecebe93bd11`.
[cdtd-source.audit.json](cdtd-source.audit.json) binds the MIT license,
untruncated Git tree and 46 declared code/configuration files. Each fetched
file matches its Git blob and SHA256. Only immutable individual source blobs
were fetched; the full repository archive, raw datasets, notebooks, images
and weights were excluded. Author modules were parsed, never imported.
This selected inventory is not a complete executable runtime closure.
Citation: `mueller2025continuous` from the corrected local bibliography.

## Released configuration and checkpoint policy

The author provides single, per-type and per-feature noise-schedule variants.
Their released YAMLs specify 30,000 steps, batches of 4096, five hidden layers
of 796 units, embedding dimension 16, 200 generation steps, AdamW with zero
weight decay and EMA decay 0.999. The terminal EMA model is saved; the training
loop has no best-validation checkpoint selector. The paper calls the optimizer
Adam; the released YAML says AdamW. Preserve this provenance distinction.

`main.py` needs an explicit CDTD `--cfg_path`; it also automatically runs the
author evaluation and plots after training. Common defaults are CUDA, seed 0,
five sample seeds and ten auditor seeds. The sample wrapper only seeds when
the seed is truthy, so seed 0 requires an explicit contract. YAML facts were
recorded with `yaml.safe_load`; dependency-resolved OmegaConf interpretation,
including the textual optimizer epsilon, remains unverified.

The declared ablation files document author research; they are not an admitted
study tuning grid. Freeze a default variant and any allowed search before
validation selection. Author defaults remain visible separately from tuned
configurations. Charge timeout and failed trials under the unchanged caps.

## Native objective

The [author paper](https://arxiv.org/pdf/2312.10431v7), Section 4.1, evaluates
regression by the absolute gap between real-trained and synthetic-trained
model-averaged RMSE, minimized. The released evaluators use Ridge, random
forest and CatBoost with equal weights. They return raw real/synthetic RMSE
and R² summaries; they do not implement a generator-search gap scalar.
Each evaluation fits its target standardizer to its own training labels.
The exact scaling, aggregation, validation binding, generator grid and
tie-breaks require a frozen study objective before tuning. Auditor/detection
CatBoost tuning must not be reported as generator tuning.

Native selection must use official-training-derived validation rows.
Our retention/MFS outcomes cannot select CDTD configurations. If a defensible
objective cannot be frozen, run the author default with tuning inapplicable.
Native KPI values are never ranked across methods.

## Admission blockers in the released entry points

The named-dataset loader creates its own splits and the initial categorical
encoder fits concatenated train/validation/test categories. A study wrapper
must bind the frozen grouped fit/validation partitions and fit preprocessing
only on fit rows. Official tests cannot enter author routines during selection.

With no categorical columns, the loader sets `num_cats` and categorical data
to `None`; the experiment calls `min/max(num_cats)`, the diffusion constructor
calls `len(categories)`, and the loss stacks categorical terms unconditionally.
These source assumptions leave the common-numeric contract unverified.
Any original-source adaptation needs versioned changes and contract evidence;
adding fake categorical features would change the study representation.

The checkpoint stores both `data_wrangler` and `train_loader`, including
original partition arrays and training tensors. Retaining those objects in
an inference artifact is a row-copy/privacy and byte-accounting blocker.
Remove original rows only through a verified inference-state contract, prove
safe local reload, and charge all required weights, noise schedules, learned
encoders/statistics, configuration and projection. Detailed errors, saved
data and plots remain on scratch.

The Python 3.10 requirements pin many direct versions but omit explicit
scikit-learn/tqdm versions. Eager imports also reach other baselines and
vendored code. Verify the complete runtime, import roots, bytecode and rights
before dependency initializers. Fit/sample replay, finite rows, artifact
privacy, real-vs-real controls and fresh aggregate xbabe1/2/3 admission remain
pending. This audit establishes no formal DP or measured benchmark result.
