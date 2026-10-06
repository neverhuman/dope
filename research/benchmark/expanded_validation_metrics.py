"""Training-derived research diagnostics; no composite, certification or DP score.

Inputs are already frozen common-numeric projections with explicit column kinds.
No header, source value, attack score or nearest-row identifier is returned.
Alpha/beta is the unembedded variant from pinned Synthcity; DOMIAS is the
isolated author KDE equation, not the original training-inclusive executable.
"""
import hashlib
import math

SEED = 1729
MAX_ROWS = 800


def unavailable(reason):
    return {"status": "unavailable", "reason": reason, "value": None}


def measured(value, **more):
    return {"status": "ok", "value": value, **more}


def checked_arrays(*arrays):
    import numpy as np
    try:
        result = [np.asarray(x, dtype=np.float64) for x in arrays]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("invalid_numeric_matrix_values") from None
    if not result or any(x.ndim != 2 or len(x) < 2 or x.shape[1] == 0 for x in result):
        raise ValueError("invalid_numeric_matrix_shape")
    if len({x.shape[1] for x in result}) != 1 or any(not np.isfinite(x).all() for x in result):
        raise ValueError("invalid_numeric_matrix_values")
    return result


def subset(x, n, seed=SEED):
    import numpy as np
    return x[np.random.default_rng(seed).permutation(len(x))[:n]]


def summary(x):
    import numpy as np
    return {"rows": len(x), "mean": float(np.mean(x)), "median": float(np.median(x)),
            "p05": float(np.quantile(x, .05)), "p95": float(np.quantile(x, .95))}


def fidelity(real, synthetic, column_kinds):
    import numpy as np
    from scipy.stats import ks_2samp
    real, synthetic = checked_arrays(real, synthetic)
    if len(column_kinds) != real.shape[1] or set(column_kinds) - {"continuous", "categorical"}:
        raise ValueError("explicit_column_types_required")
    marginals = []
    categorical = []
    for i, kind in enumerate(column_kinds):
        a, b = real[:, i], synthetic[:, i]
        if kind == "continuous":
            value = float(ks_2samp(a, b).statistic)
            metric = "ks_statistic"
        else:
            categorical.append(i)
            cats = np.unique(np.concatenate((a, b)))
            value = float(sum(abs(np.mean(a == c) - np.mean(b == c)) for c in cats) / 2)
            metric = "total_variation"
        marginals.append({"column_index": i, "kind": kind, "metric": metric, "value": value})
    correlation = []
    excluded = 0
    eligible = 0
    mixed = 0
    for i in range(real.shape[1]):
        for j in range(i):
            if column_kinds[i] != column_kinds[j]:
                mixed += 1
            if column_kinds[i] != "continuous" or column_kinds[j] != "continuous":
                continue
            eligible += 1
            if any(np.std(x[:, k]) == 0 for x in (real, synthetic) for k in (i, j)):
                excluded += 1
                continue
            correlation.append(abs(float(np.corrcoef(real[:, [i, j]].T)[0, 1]) -
                                   float(np.corrcoef(synthetic[:, [i, j]].T)[0, 1])))
    contingency = []
    for ii, i in enumerate(categorical):
        for j in categorical[:ii]:
            a, b = real[:, [i, j]], synthetic[:, [i, j]]
            cats = np.unique(np.concatenate((a, b)), axis=0)
            contingency.append(float(sum(abs(np.mean(np.all(a == c, axis=1)) -
                                             np.mean(np.all(b == c, axis=1))) for c in cats) / 2))
    return {"marginals": marginals,
            "pairwise_pearson_eligibility": {
                "total_pairs": real.shape[1] * (real.shape[1] - 1) // 2,
                "eligible_continuous_pairs": eligible, "excluded_constant_pairs": excluded,
                "measured_pairs": len(correlation), "mixed_pairs": mixed,
                "categorical_pairs": len(contingency)},
            "pairwise_pearson_difference": measured(summary(correlation), excluded_constant_pairs=excluded)
            if correlation else unavailable("no_nonconstant_pairs"),
            "categorical_contingency_tv": measured(summary(contingency)) if contingency
            else unavailable("no_categorical_pairs"),
            "mixed_type_dependence": unavailable("mixed_type_dependence_not_measured" if mixed
                                                  else "no_mixed_type_pairs")}


def alpha_beta(real, synthetic):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    real, synthetic = checked_arrays(real, synthetic)
    n = min(len(real), len(synthetic), MAX_ROWS)
    real, synthetic = subset(real, n), subset(synthetic, n, SEED + 1)
    center = real.mean(axis=0)
    alphas = np.linspace(0, 1, 30)
    radii = np.quantile(np.linalg.norm(real - center, axis=1), alphas)
    real_d = NearestNeighbors(n_neighbors=2, n_jobs=1).fit(real).kneighbors(real)[0][:, 1]
    synth_d, synth_i = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(synthetic).kneighbors(real)
    synth_d, synth_i = synth_d[:, 0], synth_i[:, 0]
    closest_radius = np.linalg.norm(synthetic[synth_i] - synthetic.mean(axis=0), axis=1)
    closest_quantiles = np.quantile(closest_radius, alphas)
    synth_radius = np.linalg.norm(synthetic - center, axis=1)
    precision = np.asarray([np.mean(synth_radius <= r) for r in radii])
    recall = np.asarray([np.mean((synth_d <= real_d) & (closest_radius <= r)) for r in closest_quantiles])
    delta_p = float(1 - abs(precision - alphas).sum() / alphas.sum())
    delta_r = float(1 - abs(recall - alphas).sum() / alphas.sum())
    if min(delta_p, delta_r) < 0:
        return unavailable("negative_author_alpha_beta_delta")
    return measured({"alpha_precision": delta_p, "beta_recall": delta_r},
                    equal_subsample_rows=n, variant="unembedded_pinned_synthcity_equations",
                    alpha_grid=alphas.tolist(), precision_curve=precision.tolist(), recall_curve=recall.tolist())


def row_groups(x):
    """Canonical projected-row digests; -0 and +0 have the same identity."""
    import numpy as np
    x = np.asarray(x, dtype="<f8").copy()
    x[x == 0] = 0
    return np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in x])


def unique_group_rows(x, seed):
    """One row per projected group, with uniformly seeded group ordering."""
    import numpy as np
    indices = np.unique(row_groups(x), return_index=True)[1]
    return subset(x[indices], len(indices), seed)


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
        else:
            raise ValueError("unknown_detection_auditor")
        model.fit(x[fit], y[fit])
        pred[test] = model.predict_proba(x[test])[:, 1]
        scores.append(float(roc_auc_score(y[test], pred[test])))
    return measured(float(np.mean(scores)), fold_aucs=scores, pooled_oof_auc=float(roc_auc_score(y, pred)),
                    folds=5, rows_per_class=n, auditor=auditor, duplicate_group_leakage=False)


def privacy(fit, validation, synthetic, column_kinds):
    import numpy as np
    from scipy.stats import gaussian_kde
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    fit, validation, synthetic = checked_arrays(fit, validation, synthetic)
    if len(column_kinds) != fit.shape[1] or set(column_kinds) - {"continuous", "categorical"}:
        raise ValueError("explicit_column_types_required")
    if set(row_groups(fit)) & set(row_groups(validation)):
        raise ValueError("training_validation_projected_overlap")
    factor = math.sqrt(fit.shape[1])
    synth = subset(synthetic, min(MAX_ROWS, len(synthetic)), SEED + 2)
    def near(q, r, k=1):
        return NearestNeighbors(n_neighbors=k, n_jobs=1).fit(r).kneighbors(q)[0] / factor
    train_d = near(synth, fit)[:, 0]
    val_d = near(synth, validation)[:, 0]
    nref = min(len(fit), len(validation), MAX_ROWS)
    balanced_train = near(synth, subset(fit, nref))[:, 0]
    balanced_val = near(synth, subset(validation, nref, SEED + 1))[:, 0]
    distances = near(synth, fit, 2)
    defined = distances[:, 1] != 0
    val_unique = unique_group_rows(validation, SEED + 3)
    fit_unique = unique_group_rows(fit, SEED + 4)
    nq = min(len(fit_unique), len(val_unique) // 2, MAX_ROWS)
    result = {"empirical_only": True, "formal_dp": False, "hipaa_deidentification": False,
              "attack_query_weighting": "uniform_unique_projected_row_groups",
              "kde_reference_weighting": "remaining_disjoint_real_rows",
              "unique_query_group_counts": {"fit": len(fit_unique), "validation": len(val_unique)},
              "dcr_fit": summary(train_d), "dcr_validation": summary(val_d),
              "full_reference_counts": {"fit": len(fit), "validation": len(validation)},
              "share_closer_to_train": float(np.mean((train_d < val_d) + .5 * (train_d == val_d))),
              "balanced_reference_rows": nref,
              "balanced_share_closer_to_train": float(np.mean((balanced_train < balanced_val) +
                                                               .5 * (balanced_train == balanced_val))),
              "nndr_fit": measured(summary(distances[defined, 0] / distances[defined, 1]),
                                    undefined_zero_second_distance_rows=int((~defined).sum()))
              if defined.any() else unavailable("all_second_neighbor_distances_zero")}
    if nq < 2:
        result.update(distance_mia=unavailable("insufficient_disjoint_attack_queries"),
                      domias_kde=unavailable("insufficient_disjoint_reference_queries"))
        return result
    nonmembers = val_unique[:nq]
    attack_groups = set(row_groups(nonmembers))
    reference = validation[[group not in attack_groups for group in row_groups(validation)]]
    members = fit_unique[:nq]
    queries = np.concatenate((members, nonmembers))
    y = np.concatenate((np.ones(nq), np.zeros(nq)))
    assert not set(row_groups(reference)) & set(row_groups(nonmembers))
    result["distance_mia"] = measured(float(roc_auc_score(y, -near(queries, synth)[:, 0])),
                                       member_queries=nq, nonmember_queries=nq)
    if set(column_kinds) != {"continuous"}:
        result["domias_kde"] = unavailable("continuous_kde_not_applicable_to_declared_categories")
    elif min(len(synth), len(reference)) <= fit.shape[1]:
        result["domias_kde"] = unavailable("insufficient_rows_for_full_dimensional_kde")
    elif any(np.linalg.matrix_rank(x - x.mean(axis=0)) < fit.shape[1] for x in (synth, reference)):
        result["domias_kde"] = unavailable("singular_full_dimensional_kde")
    else:
        try:
            pg = gaussian_kde(synth.T)(queries.T)
            pr = gaussian_kde(reference.T)(queries.T)
        except np.linalg.LinAlgError:
            result["domias_kde"] = unavailable("numerically_singular_full_dimensional_kde")
        else:
            ratio = pg / (pr + 1e-10)
            if not np.isfinite(ratio).all():
                result["domias_kde"] = unavailable("nonfinite_author_density_ratio")
            else:
                result["domias_kde"] = measured(float(roc_auc_score(y, ratio)), member_queries=nq,
                                                nonmember_queries=nq, reference_rows=len(reference),
                                                variant="isolated_author_kde_equation_scott_bandwidth")
    return result


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
