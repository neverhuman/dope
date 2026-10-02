# TabSyn author-source admission audit

This is a source audit, not a trained comparator or a final method lock.
TabSyn remains `source_audit_pending`; its runtime, fit/sample contract and
native tuning are incomplete. Official tests have not been opened, and
MFS-v2, PTF-v1 and release-safe results remain null. It contributes no DOPE win.

## Source and rights

The author repository is [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn),
commit `cb5ac0f74ec36ee88e7a974a393dfbef50d42da7`, under Apache-2.0.
The local bibliography entry is `zhang2024mixed`.
[tabsyn-source.audit.json](tabsyn-source.audit.json) records the archive,
license, all 96 declared source/documentation hashes, cached inventory and
immutable verification receipt. Each declared extracted file matches both
the archive member and the Git blob from the cached, untruncated author tree.
Archive links and special members were rejected without importing author code.
Skipped bytecode, images and dataset metadata were not read. This scope does
not establish an executable runtime closure or a reported-experiment reproduction.

## Author defaults

| Stage | Author settings |
| --- | --- |
| VAE | 4,000 epochs; batch 4,096; Adam learning rate 0.001, weight decay 0; token dimension 4; two transformer layers; one head; factor 32; beta maximum 0.01, minimum 0.00001, decay 0.7 |
| Latent diffusion | At most 10,001 epochs; batch 4,096; Adam learning rate 0.001, weight decay 0; hidden dimension 1,024; patience 500 |
| Sampling | 50 diffusion steps by function default; number of rows equals the training latent row count |

Sources are [VAE training](https://github.com/amazon-science/tabsyn/blob/cb5ac0f74ec36ee88e7a974a393dfbef50d42da7/tabsyn/vae/main.py),
[diffusion training](https://github.com/amazon-science/tabsyn/blob/cb5ac0f74ec36ee88e7a974a393dfbef50d42da7/tabsyn/main.py),
[CLI defaults](https://github.com/amazon-science/tabsyn/blob/cb5ac0f74ec36ee88e7a974a393dfbef50d42da7/utils.py)
and [sampling](https://github.com/amazon-science/tabsyn/blob/cb5ac0f74ec36ee88e7a974a393dfbef50d42da7/tabsyn/sample.py).
The sampling entry point reads `args.steps` but does not pass it to the diffusion
sampler. A requested sample count or step override is not an implemented contract.
The campaign's 600-second whole-fit ceiling remains mandatory, including both stages.

## Native regression efficacy

The author [ML efficacy evaluator](https://github.com/amazon-science/tabsyn/blob/cb5ac0f74ec36ee88e7a974a393dfbef50d42da7/eval/mle/mle.py)
reports `best_r2_scores.XGBRegressor.r2` and a separately selected RMSE result.
R2 is the candidate objective to maximize; its implementation is identified,
but its RNG rules, generator search space and generator tie-breaks have not
been frozen for tuning. The source's active regression auditor grid has 36
XGBoost configurations. That grid tunes the auditor, not the generator.

The evaluator fits feature transforms to synthetic data and reuses them on
the real evaluation partition. It shuffles synthetic rows with NumPy, reserves
`floor(n / 9)` for internal auditor validation, and uses the remaining rows
to fit the auditor. It log-transforms labels clipped to `[1, 20000]` for auditor
training and real evaluation. Internal synthetic validation labels remain
untransformed. Auditor parameters are selected by R2 on that internal partition,
then refitted and measured on real evaluation rows. These are the pinned
implementation's behaviors; they must not be silently replaced with the shared
benchmark's raw-target utility formula.

For generator tuning, the author test-named input must contain only the study's
frozen validation rows derived from official training. An uninformative target
after the author transform needs an explicit tuning-inapplicable receipt.
It cannot establish a native winner. Native values are never ranked across methods.

## Work required before execution

1. Pin an isolated runtime and verify it before dependency imports. The README
   specifies Python 3.10 and offers a PyTorch 2.0.1/CUDA 11.7 installation example;
   most requirements are unversioned. No installed environment is certified here.
2. Stage frozen train/validation partitions under the expected author names.
   `utils_train.preprocess` uses `change_val=False` and reads test-named arrays;
   the original dataset preparation must not generate new study splits.
3. Complete the purely numeric representation contract. The VAE loss fails when
   the categorical reconstruction list is empty (`idx` is unbound and the accuracy
   denominator is zero). A private proposed loss patch has generated-input checks,
   but it has not passed a complete fit/sample contract or been admitted.
4. Preserve or explicitly disclose VAE checkpoint behavior: categorical validation
   loss selects the saved full model with zero weight on numeric MSE, while final
   encoder/decoder exports use final-epoch weights. A numeric-only adaptation must
   freeze its checkpoint semantics before any fit.
5. Verify scheduler, XGBoost and sklearn API compatibility; freeze native seeds,
   sample sizes, objective direction, tie-breaks and any defensible generator grid.
   If there is no defensible author tuning objective for a cell, retain the author
   default and report tuning inapplicable.
6. Implement repeated seeded sampling at `n/2n/4n/8n`, safe weight/preprocessing
   serialization, complete charged bytes, fresh host admission and failure costs.
   Learned preprocessing and latent-derived state cannot be omitted from charges.

No source observations, samples, weights or detailed logs are included in this audit.
