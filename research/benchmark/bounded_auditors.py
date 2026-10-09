"""Explicit four-thread auditors for new measurements.

The historical pilot and expanded harness files retain their measured identities.
This module reuses their metric equations and adds a separate resource profile;
callers must bind this source together with both imported historical sources.
Numerical libraries are initialized only when an auditor is requested.
"""
from functools import partial

from research.benchmark import pilot_metrics as historical_pilot
from research.benchmark.expanded_validation_metrics import (
    SEED, MAX_ROWS, unavailable, measured, checked_arrays, subset, row_groups,
    fidelity, alpha_beta, privacy, unique_group_rows,
)


def bound_catboost(model, task):
    model.get_feature_importance = partial(model.get_feature_importance, thread_count=4)
    model.predict = partial(model.predict, thread_count=4)
    if task == "binary":
        model.predict_proba = partial(model.predict_proba, thread_count=4)
    return model


def _model(name, task, seed):
    model = historical_pilot._model(name, task, seed)
    return bound_catboost(model, task) if name == "catboost" else model


def c2st(real, synthetic, auditor):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    real, synthetic = checked_arrays(real, synthetic)
    n = min(len(real), len(synthetic), MAX_ROWS)
    x = np.concatenate((subset(real, n), subset(synthetic, n, SEED + 1)))
    y = np.concatenate((np.zeros(n, dtype=int), np.ones(n, dtype=int)))
    groups = row_groups(x)
    if any(len(np.unique(groups[y == k])) < 5 for k in (0, 1)):
        return unavailable("fewer_than_five_unique_row_groups_per_class")
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(x, y, groups))
    if any(len(np.unique(y[indices])) != 2 for pair in folds for indices in pair):
        return unavailable("grouped_fold_missing_detection_class")
    pred = np.full(len(y), np.nan)
    scores = []
    for fit, test in folds:
        assert not set(groups[fit]) & set(groups[test])
        if auditor == "logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=SEED))
        elif auditor == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(iterations=100, depth=6, random_seed=SEED, thread_count=4,
                                       verbose=False, allow_writing_files=False, task_type="CPU")
            # fit() computes feature importance with its own default thread count.
            # Bind that calculation to the same budget as detector training.
            model = bound_catboost(model, task="binary")
        else:
            raise ValueError("unknown_detection_auditor")
        model.fit(x[fit], y[fit])
        if auditor == "catboost":
            predictions = model.predict_proba(x[test], thread_count=4)
        else:
            predictions = model.predict_proba(x[test])
        pred[test] = predictions[:, 1]
        scores.append(float(roc_auc_score(y[test], pred[test])))
    return measured(float(np.mean(scores)), fold_aucs=scores, pooled_oof_auc=float(roc_auc_score(y, pred)),
                    folds=5, rows_per_class=n, auditor=auditor, duplicate_group_leakage=False)

def evaluate(fit, validation, synthetic, column_kinds):
    fit, validation, synthetic = checked_arrays(fit, validation, synthetic)
    if len(column_kinds) != fit.shape[1] or set(column_kinds) - {"continuous", "categorical"}:
        raise ValueError("explicit_column_types_required")
    if set(row_groups(fit)) & set(row_groups(validation)):
        raise ValueError("training_validation_projected_overlap")
    return {"format": "dope-expanded-validation-diagnostics", "version": 1,
            "fidelity": fidelity(validation, synthetic, column_kinds),
            "alpha_beta": alpha_beta(validation, synthetic),
            "detection": {name: c2st(validation, synthetic, name) for name in ("catboost", "logistic")},
            "privacy": privacy(fit, validation, synthetic, column_kinds),
            "official_tests_opened": False, "mfs_v2": None, "ptf_v1": None,
            "release_safe_l3": None, "superiority": None}
