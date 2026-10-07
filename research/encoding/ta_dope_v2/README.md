# TA_Dope_V2: target-aware, learned successor to the Dope dataset sketch encoder

`ta_dope_v2.py` holds the model and the featuriser. Training, data prep, leak tests and the GP-HWM checkpoint probe live in
JopeDime `experiments/ta_encoders/` (branch `exp/ta-hyperion-v8`), which imports this module.

How Dope v1 sees the label: the 856-d `DatasetSketch::from_train` (rust/router/part_01.rs) includes the target
moments, quantiles and rare share, plus each feature's |Pearson corr(x, y)|. It then goes through a frozen, quantised
3-layer router to give the 4168-d vector. That label signal is marginal: Dope v1 never models how y depends on X jointly,
and the router was trained for operator routing, not for this.

What V2 changes:
- It's learned and permutation-invariant over features, with no canonical sort or top-64 truncation (128 features in training).
- It adds target-aware per-feature stats: signed Pearson, Spearman, eta², corr(x², y), univariate R², in-context ridge
  coefficient and |coefficient|, drop-one gain, and rank slope.
- It adds a learned DeepSets sketch of the (x, y) pairs per feature.
- It adds learnability globals: ridge LOO R², 5-NN LOO R², and residual skew and kurtosis.
- It uses a set transformer with label-conditioned pooling, giving a 1024-d context embedding plus 192-d row embeddings.
- It's trained self-supervised with masked-y in-context regression plus two-view InfoNCE. GP labels are never used.
