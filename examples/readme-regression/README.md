# README regression toy

This directory is a small regression problem for the install example. It is
not the 97-lineage PMLB panel in the manuscript, and compiling it does not
certify a release.

`train.csv` and `test.csv` are headerless. Each row is `x1,x2,x3,y`. Values
are already in `[0, 1]`. The target is the last column. `compile` opens only
`train.csv`. `test.csv` is here because `certify` expects an evaluator-owned
test file. The README example does not score it.

The files are written by `make_tables.py`. Training uses NumPy seed
`20261008` and 400 rows. The test file uses seed `20261009` and 100 rows.
Four uniform draws `x1`, `x2`, `x3`, and `u` come from `[0, 1)`. The stored
target is

```text
y = 0.55 x1 + 0.25 x2 + 0.10 u
```

`u` is noise and is not a column. `x3` is stored and unused by that formula.
The coefficients keep `y` inside `[0, 0.90]`, so no further scaling is
applied. `0.55`, `0.25`, and `0.10` are the toy's coefficients, not a result
from the paper.

`toy.dpk`, its compile report, and `synth.csv` are local outputs. They are
gitignored. `INSPECT.txt` is the inspect transcript from the binary built in
this tree.
