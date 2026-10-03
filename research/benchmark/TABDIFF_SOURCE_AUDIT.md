# TabDiff author source audit

Original author source is pinned at `5ecdb3356261aea72716cc9a779f31d7ad083bf4` under its MIT license. Twenty-seven source/documentation files match the cached archive and Git blob inventories. The local bibliography key is `shi2025tabdiff`.

This is source admission preparation. No runtime, fit/sample contract, generator tuning grid or campaign admission is certified, and no new fit ran. The paired paper comparison and release scores remain unavailable here.

## Author objective

`TabMetrics.evaluate_mle` reports XGBoost RMSE for regression, minimized, and ROC-AUC for binary classification, maximized. For regression, the 36 configurations tune the auditor by validation RMSE, refit it, then evaluate on the real evaluation partition. Those configurations are auditor settings; the generator tuning space is still unfrozen. The real evaluation path must receive official-training-derived validation during tuning, with official tests sealed.

The author implementation applies `log(clip(y, 1, 20000))` to auditor training and real evaluation labels while leaving validation labels untransformed. Informative target compatibility must be established before tuning. No shared DOPE KPI substitutes for this objective. Native values are never ranked across methods.

## Author defaults and implementation contracts

The default TOML runs 8,000 epochs with batch size 4,096, learning rate 0.001, zero weight decay and EMA decay 0.997. The value named `steps` controls the epoch loop. Best EMA weights are saved from unweighted training mixed loss only after `curr_epoch > 4000`; this selection does not use validation loss. The training DataLoader has four workers. Default sampling uses 50 diffusion steps; the sample batch size is 10,000 and report mode averages 20 runs.

Preprocessing reads test-named arrays during fitting. Those paths must receive only frozen official-training-derived validation in the adapter. Pure numeric staging needs verified zero-column categorical arrays. The deterministic author flag fixes seed zero, so protocol seed mapping needs an explicit adapter contract. Config pickle and Torch checkpoint loads require owned, hash-verified artifacts. Learned transforms, noise schedules, weights, metadata and projection must all be charged.

The author environment specifies Python 3.10, Torch 2.0.1 and CUDA 11.7, with partially unpinned dependencies. XGBoost `gpu_hist`/`reg:linear` and sklearn `mean_squared_error(squared=False)` need pinned compatible versions. A bounded fit must retain the author default checkpoint behavior and receipt any timeout; shorter research runs need separate labels.
