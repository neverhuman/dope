# synthpop CART pilot adapter

The research adapter calls synthpop 1.9-3 `syn()` to fit sequential CART models on
the frozen common-numeric training partition. It measures the method's own
validation objective with `utility.gen(..., method="cart",
resamp.method="none")$pMSE`; lower is better. The four author-supported
`cart.minbucket`/`cart.cp` configurations and tie breaks are pinned in
[`synthpop-cart-source.lock.json`](synthpop-cart-source.lock.json). No official
test partition is used for fitting, tuning, or this metric.

The package source is GPL-2 or GPL-3. The separate
[`synthpop_cart.R`](synthpop_cart.R) wrapper follows its CART donor-leaf
sampling algorithm and carries the same license choice. It is restricted
benchmark code, outside the MIT product runtime. The fitted artifact retains
donor values and rpart models; it stays on benchmark scratch and every byte is
charged. It has no formal privacy or L3 release claim.

The adapter's `fit` command writes one fitted artifact and a native KPI
receipt. Its `sample` command takes only that artifact, a row count, and a
seed. The synthetic contract probe covers CART leaf support, deterministic
artifact-only replay, constant and collinear columns, and finite numeric
output. Run it with the pinned scratch R environment:

```sh
/mnt/fast-scratch/dope-benchmark/envs/synthpop-r44/bin/Rscript \
  research/benchmark/tests/probe_synthpop_cart.R
```

The source audit is complete. The exact method matrix remains unadmitted
until the active ARF rounds finish, because those immutable rounds pin the
current `methods.lock.json` hash. The full pilot comparison, MFS-v2, and
PTF-v1 remain unmeasured.
