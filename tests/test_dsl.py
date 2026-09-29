from __future__ import annotations

import pytest

from dope_kernel.dsl import DSLNode, canonicalize, parse_sexpr


@pytest.mark.parametrize(
    "expr",
    [
        "(dk seed=7 p=3 task=regression v=1 n=20 (marginals count=3 (beta i=0) (zi_beta i=1) (empq i=2 k=17)) (dependence rank=3 kind=gauss_copula) (target kind=linear_sparse) (residual kind=homo))",
        "(dk v=1 task=binary n=10 p=2 seed=9 (marginals count=2 (bern i=0) (ordinal i=1 k=4)) (dependence kind=ind) (target kind=lift_logit) (residual kind=bernoulli_cal))",
        "(SCM edges=3 (RFFGP rank=8) (TREEPW leaves=6) (HETQ k=5) (REGIME k=2) (MOE experts=3) (CENS lo=0 hi=1) (COUNT mode=zero_one))",
        "(target kind=gam (poly_cross degree=2) (mlp_lowrank r=3) (scm nodes=4) (regime k=2) (moe experts=2))",
        "(dependence kind=vine k=4 (chow_liu mi=emp) (factor_copula r=2) (mixture gate=soft) (manifold knn=8))",
        "(residual kind=quantile (censored lo=0 hi=1) (zero_one_inflated p0=0.1 p1=0.2))",
    ],
)
def test_canonical_round_trip(expr: str) -> None:
    canonical = canonicalize(expr)
    assert parse_sexpr(canonical).to_sexpr() == canonical


def test_macro_expansion_counts_macro_as_one_gp_token() -> None:
    node = parse_sexpr("(TREEPW leaves=3 (homo sigma=0.1))")
    assert node.gp_token_count() == 1
    expanded = node.expand_macros()
    assert isinstance(expanded, DSLNode)
    assert expanded.op == "tree_piecewise"
