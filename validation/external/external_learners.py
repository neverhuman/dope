"""Pinned shadow learners and permutation-importance comparison."""

from __future__ import annotations

import math
import warnings

import numpy as np

MAX_FI_HOLDOUT_ROWS = 256


def library_version(library: str) -> str:
    if library == "scikit-learn":
        import sklearn

        return sklearn.__version__
    if library == "xgboost":
        import xgboost

        return xgboost.__version__
    if library == "lightgbm":
        import lightgbm

        return lightgbm.__version__
    raise ValueError("unknown external library")


def json_safe(value):
    """Preserve model-parameter sentinels without allowing NaN evidence numbers."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    return value


def make_estimator(model_id: str, task: str, seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    if model_id == "sklearn_linear":
        if task == "binary":
            from sklearn.linear_model import LogisticRegression

            estimator = LogisticRegression(
                solver="liblinear", max_iter=1000, random_state=seed
            )
        else:
            from sklearn.linear_model import Ridge

            estimator = Ridge(alpha=1.0, solver="svd")
        library = "scikit-learn"
    elif model_id == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if task == "binary" else RandomForestRegressor
        estimator = cls(n_estimators=64, max_depth=6, n_jobs=1, random_state=seed)
        library = "scikit-learn"
    elif model_id == "xgboost":
        import xgboost as backend
        cls = backend.XGBClassifier if task == "binary" else backend.XGBRegressor
        estimator = cls(
            n_estimators=64,
            max_depth=4,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            n_jobs=1,
            tree_method="hist",
            random_state=seed,
            verbosity=0,
        )
        library = "xgboost"
    elif model_id == "lightgbm":
        import lightgbm as backend
        cls = backend.LGBMClassifier if task == "binary" else backend.LGBMRegressor
        estimator = cls(
            n_estimators=64,
            max_depth=4,
            learning_rate=0.05,
            num_threads=1,
            deterministic=True,
            force_col_wise=True,
            random_state=seed,
            verbosity=-1,
        )
        library = "lightgbm"
    else:
        raise ValueError("unknown external model ID")
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), estimator
    ), library


def predictions(model, features: np.ndarray, task: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBM",
            category=UserWarning,
        )
        if task == "binary":
            return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
        return np.asarray(model.predict(features), dtype=np.float64)


def loss(target: np.ndarray, predicted: np.ndarray, task: str) -> float:
    if task == "binary":
        probabilities = np.clip(predicted, 1e-12, 1.0 - 1e-12)
        return float(
            -np.mean(
                target * np.log(probabilities)
                + (1.0 - target) * np.log1p(-probabilities)
            )
        )
    return float(np.sqrt(np.mean((target - predicted) ** 2)))


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0
        position = end
    return ranks


def compare_importance(
    real: np.ndarray, synthetic: np.ndarray, top_k: int | None
) -> dict:
    if real.shape != synthetic.shape:
        raise ValueError("permutation-importance feature counts differ")
    informative = np.flatnonzero(real > 1e-12)
    real_positive = np.maximum(real, 0.0)
    synthetic_positive = np.maximum(synthetic, 0.0)
    real_total = float(real_positive.sum())
    synthetic_total = float(synthetic_positive.sum())
    real_shares = real_positive / real_total if real_total > 0 else np.zeros_like(real)
    synthetic_shares = (
        synthetic_positive / synthetic_total
        if synthetic_total > 0
        else np.zeros_like(synthetic)
    )
    applicable = len(informative) >= 3
    result = {
        "feature_count": int(len(real)),
        "informative_feature_count": int(len(informative)),
        "applicable": applicable,
        "real_normalized_shares": real_shares.tolist(),
        "synthetic_normalized_shares": synthetic_shares.tolist(),
        "mean_ratio_error": None,
        "spearman": None,
        "top_k": None,
        "top_k_jaccard": None,
        "real_top_indices": [],
        "synthetic_top_indices": [],
    }
    if not applicable:
        return result
    real_rank = average_ranks(real[informative])
    synthetic_rank = average_ranks(synthetic[informative])
    if np.std(real_rank) > 0 and np.std(synthetic_rank) > 0:
        result["spearman"] = float(np.corrcoef(real_rank, synthetic_rank)[0, 1])
    k = (
        min(10, math.ceil(len(informative) / 5))
        if top_k is None
        else min(top_k, len(informative))
    )
    real_top = set(
        sorted(informative, key=lambda index: (-real[index], int(index)))[:k]
    )
    synthetic_top = set(
        sorted(informative, key=lambda index: (-synthetic[index], int(index)))[:k]
    )
    result["top_k"] = k
    result["top_k_jaccard"] = len(real_top & synthetic_top) / len(
        real_top | synthetic_top
    )
    result["real_top_indices"] = sorted(int(index) for index in real_top)
    result["synthetic_top_indices"] = sorted(int(index) for index in synthetic_top)
    result["mean_ratio_error"] = float(
        np.mean(
            np.abs(synthetic_shares[informative] - real_shares[informative])
            / np.maximum(real_shares[informative], 1e-12)
        )
    )
    return result


def evaluate_model(
    model_id: str,
    task: str,
    seed: int,
    real_train,
    synthetic_train,
    holdout,
    top_k: int | None,
) -> dict:
    from sklearn.inspection import permutation_importance

    real_x, real_y = real_train
    synthetic_x, synthetic_y = synthetic_train
    holdout_x, holdout_y = holdout
    real_model, library = make_estimator(model_id, task, seed)
    synthetic_model, _ = make_estimator(model_id, task, seed)
    model_config = json_safe(real_model.steps[-1][1].get_params(deep=False))
    real_model.fit(real_x, real_y)
    synthetic_model.fit(synthetic_x, synthetic_y)
    real_predictions = predictions(real_model, holdout_x, task)
    synthetic_predictions = predictions(synthetic_model, holdout_x, task)
    null_prediction = float(np.mean(real_y))
    if task == "binary":
        null_prediction = min(max(null_prediction, 1e-12), 1.0 - 1e-12)
    null_loss = loss(holdout_y, np.full(len(holdout_y), null_prediction), task)
    trtr_loss = loss(holdout_y, real_predictions, task)
    tstr_loss = loss(holdout_y, synthetic_predictions, task)
    informative = (
        null_loss - trtr_loss >= 0.01 * abs(null_loss) and null_loss > trtr_loss
    )

    def score(model, features, target):
        return -loss(target, predictions(model, features, task), task)

    fi_rows = np.sort(
        np.random.default_rng(seed).choice(
            len(holdout_y), size=min(len(holdout_y), MAX_FI_HOLDOUT_ROWS), replace=False
        )
    )
    fi_x, fi_y = holdout_x[fi_rows], holdout_y[fi_rows]
    real_importance = permutation_importance(
        real_model, fi_x, fi_y, scoring=score, n_repeats=3, random_state=seed, n_jobs=1
    ).importances_mean
    synthetic_importance = permutation_importance(
        synthetic_model,
        fi_x,
        fi_y,
        scoring=score,
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
    ).importances_mean
    metric = {
        "null_loss": null_loss,
        "trtr_loss": trtr_loss,
        "tstr_loss": tstr_loss,
        "retention": (null_loss - tstr_loss) / (null_loss - trtr_loss)
        if informative
        else None,
        "regret": tstr_loss - trtr_loss,
        "informative": informative,
    }
    if task == "binary":
        from sklearn.metrics import roc_auc_score

        metric["trtr_auc"] = (
            float(roc_auc_score(holdout_y, real_predictions))
            if len(np.unique(holdout_y)) == 2
            else None
        )
        metric["tstr_auc"] = (
            float(roc_auc_score(holdout_y, synthetic_predictions))
            if len(np.unique(holdout_y)) == 2
            else None
        )
    return {
        "status": "measured",
        "model_id": model_id,
        "estimator": type(real_model.steps[-1][1]).__name__,
        "library": library,
        "library_version": library_version(library),
        "model_config": model_config,
        "fi_holdout_rows": len(fi_rows),
        "metrics": metric,
        "feature_importance": compare_importance(
            real_importance, synthetic_importance, top_k
        ),
    }
