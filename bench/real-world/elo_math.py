"""Order-independent Bradley–Terry ratings in Elo units (offline diagnostics)."""
from __future__ import annotations

import numpy as np

BASE, SCALE, PRIOR = 1500.0, 400.0 / np.log(10.0), 1.0
LABELS = {"dope": "DOPE", "rows": "Real rows (xz)", "kernel": "DOPE research kernel",
          "copula": "Gaussian copula", "arf": "ARF", "tvae": "TVAE", "ctgan": "CTGAN"}


def outcome(a: float, b: float, margin: float) -> float:
    return 0.5 if abs(a - b) <= margin else float(a > b)


def ratings(scores: np.ndarray, margin: float = 1.0) -> np.ndarray:
    """One equally weighted game per dataset/pair; N(0,1) log-odds prior.

    scores[dataset, method] are BTR-AUC percentages. Ridge-regularized batch
    logistic fitting avoids ordering effects and infinite undefeated ratings.
    A draw contributes half a win; the prior is fixed, not tuned to the panel.
    """
    n, m = scores.shape
    if n == 0 or m < 2 or not np.isfinite(scores).all():
        raise ValueError("ratings require finite scores for at least one dataset and two methods")
    wins = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            delta = scores[:, i] - scores[:, j]
            w = np.where(np.abs(delta) <= margin, .5, delta > 0).sum()
            wins[i, j], wins[j, i] = w, n - w
    skill = np.zeros(m)
    for _ in range(60):
        diff = np.clip(skill[:, None] - skill[None, :], -50, 50)
        p = 1 / (1 + np.exp(-diff))
        np.fill_diagonal(p, 0)
        grad = (n * p - wins).sum(axis=1) + PRIOR * skill
        weights = n * p * (1 - p)
        np.fill_diagonal(weights, 0)
        hessian = np.diag(weights.sum(axis=1) + PRIOR) - weights
        step = np.linalg.solve(hessian, grad)
        # Damping also bounds Newton steps for large, nearly separated panels.
        step /= max(1., float(np.max(np.abs(step))))
        skill -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return BASE + SCALE * (skill - skill.mean())


def leaderboard(scores: np.ndarray, methods: list[str], margin: float, draws: int, seed: int) -> list[dict]:
    point = ratings(scores, margin)
    rng = np.random.default_rng(seed)
    # Resample whole datasets, preserving all alternatives and their shared tests.
    samples = np.array([ratings(scores[rng.integers(0, len(scores), len(scores))], margin)
                        for _ in range(draws)])
    rows = []
    for i, method in enumerate(methods):
        w = t = l = 0
        for j in range(len(methods)):
            if i == j:
                continue
            delta = scores[:, i] - scores[:, j]
            w += int((delta > margin).sum())
            t += int((np.abs(delta) <= margin).sum())
            l += int((delta < -margin).sum())
        rows.append({"method": method, "elo": float(point[i]),
                     "lo": float(np.quantile(samples[:, i], .025)),
                     "hi": float(np.quantile(samples[:, i], .975)),
                     "datasets": len(scores), "games": w + t + l,
                     "wins": int(w), "ties": int(t), "losses": int(l),
                     "mean_btr_auc": float(scores[:, i].mean())})
    return sorted(rows, key=lambda r: (-r["elo"], r["method"]))
