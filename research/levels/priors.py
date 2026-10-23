"""Owned synthetic development mechanisms; no claim of independent real-source generalization."""
import numpy as np
from .data import role_indices
from .contracts import require


MECHANISMS = ("linear", "threshold", "interaction", "heteroscedastic", "multimodal",
              "heavy_tail", "sparse", "redundant", "low_signal", "conditional_missing")


def table(family, seed, *, rows=256, features=8):
    require(type(family)is int and 0<=family<len(MECHANISMS), "unknown_prior_family")
    rng = np.random.default_rng(np.random.SeedSequence([seed, family]))
    x = rng.normal(size=(rows, features))
    mechanism = MECHANISMS[family]
    noise = rng.normal(size=rows) * .1
    signal = x[:, 0] + .5*x[:, 1]
    if mechanism == "threshold": signal = np.where(x[:, 0] > 0, 2., -1.) + x[:, 1]*.2
    if mechanism == "interaction": signal = x[:, 0]*x[:, 1] + np.sin(x[:, 2]*3)
    if mechanism == "heteroscedastic": noise *= .1 + 3*np.abs(x[:, 1])
    if mechanism == "multimodal": signal += rng.choice([-2., 2.], size=rows)
    if mechanism == "heavy_tail": noise += rng.standard_t(3, size=rows)*.3
    if mechanism == "sparse": signal = x[:, -1]*2
    if mechanism == "redundant": x[:, 1] = x[:, 0] + noise
    if mechanism == "low_signal": signal *= .02; noise *= 5
    y = signal + noise
    categorical = family % 2 == 1
    data = x.astype(object)
    kinds = ["numeric"]*features + ["categorical" if categorical else "numeric"]
    if features > 3 and family % 3 == 0:
        data[:, -1] = np.where(x[:, -1] > 0, "c1", "c0")
        kinds[-2] = "categorical"
    if mechanism == "conditional_missing":
        data[x[:, 0] > .5, 2] = None
    target = np.where(y > 0, "positive", "negative") if categorical else y
    full = [list(data[i])+[target[i]] for i in range(rows)]
    ids = [f"prior:{family}:{seed}:{i}" for i in range(rows)]
    return dict(rows=full, kinds=kinds, slots=list(range(1, features+1))+[0], row_ids=ids,
                task="classification" if categorical else "regression",
                roles=role_indices(ids, seed), family_id=f"owned-prior:{family}",
                evidence="synthetic_development_only")
