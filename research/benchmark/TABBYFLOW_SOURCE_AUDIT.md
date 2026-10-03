# TabbyFlow author-source admission audit

TabbyFlow remains `source_audit_pending`. No author code, runtime or fit/sample
job is executed or admitted. Official tests stay sealed; MFS-v2/PTF-v1,
release-safe and superiority remain null.

The [ICML paper](https://proceedings.mlr.press/v267/guzman-cordero25a.html) and
[original author repository](https://github.com/andresguzco/ef-vfm) identify the
released EF-VFM implementation, pinned at
`11fa189f0fd2b11070831b45830b5a6952a2e323`. Its `LICENCE` is MIT;
`eval/mle/tabular_dataload.py` retains a Google Research Apache-2.0 notice.
[tabbyflow-source.audit.json](tabbyflow-source.audit.json) binds 24 selected files
by Git blob and SHA256. Full dependency/third-party rights closure is pending.
Citation: `guzmancordero2025exponential`.

## Released defaults and native objective

The released configuration uses batch 4096, learning rate 0.001, EMA 0.997 and
nominal steps 8000. The executable interprets steps as outer epochs, each running
all batches. The paper calls these iterations. Checkpoint selection uses the
lowest fresh training-loss sum, rounded to four decimals per component, after
4000 epochs, separately for regular and EMA models. The default test entry loads
a best EMA checkpoint. The paper mentions as few as 25 inference steps; released
sampling uses adaptive `dopri5`, tolerances 1e-5, over [0,0.999]. These executable
and paper facts require an explicit contract before comparison.

Author downstream efficacy uses XGBoost RMSE for regression (minimize) and AUROC
for classification (maximize). The author README says cluster hyperparameter
launch scripts were removed. No generator search grid/ties is frozen. Released
regression evaluation log-transforms clipped training and test targets but leaves
validation targets untransformed before auditor parameter selection. Verify that
metric's scale consistency and target-domain applicability with an explicit
version/label before native tuning. Our shared KPI is never substituted for the
author objective, and native values from different methods are never ranked.

## Data, inference and execution gates

Fit construction loads both train and test splits; routine density generation
evaluation reads the real training table. Replace or redirect original test-split
access to grouped, training-derived validation before execution. Official tests
remain evaluator-only. No source, demo or benchmark rows were opened in this audit.

Best model files contain vector-field state dictionaries. Inference also needs
configuration, inverse transforms and dataset information; the original config
uses pickle. Verify a safe row-free inference codec, all charged bytes, finite
requested rows and artifact-only replay. Bind all fit/sample seeds; the original
deterministic flag fixes seed zero and supplies no five-seed campaign contract.

Full interpreter/import-root/bytecode/library rights/runtime custody and fresh
aggregate admission remain pending. Enforce external whole-phase 600-second and
per-cell eight-trial/twelve-hour bounds, including failures, without weakening
author-default or checkpoint accounting. Any independent replacement requires two
reported-experiment reproductions before comparison. This audit is no reproduction,
benchmark win or production certification.
