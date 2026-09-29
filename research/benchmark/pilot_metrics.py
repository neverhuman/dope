"""Validation-only descriptive metrics for pilot costing and adapter checks.

These are not the complete frozen generator gate or a paper result. The sealed
test partition is deliberately not accepted by this entry point.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load(path: Path):
    import numpy as np
    table = np.loadtxt(path, delimiter=",", ndmin=2)
    if table.ndim != 2 or not np.isfinite(table).all():
        raise ValueError("nonfinite or malformed numeric table")
    return table


def _loss(model, x, y, task):
    import numpy as np
    from sklearn.metrics import log_loss, mean_squared_error
    if task == "binary":
        probability = model.predict_proba(x)[:, 1]
        return float(log_loss(y, np.clip(probability, 1e-9, 1 - 1e-9), labels=[0, 1]))
    return float(mean_squared_error(y, model.predict(x)))


def _model(name, task, seed):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    if name == "linear":
        return LogisticRegression(max_iter=1000, random_state=seed) if task == "binary" else LinearRegression()
    if name == "mlp":
        common = {"hidden_layer_sizes": (32,), "max_iter": 200, "random_state": seed,
                  "early_stopping": True, "n_iter_no_change": 15}
        return MLPClassifier(**common) if task == "binary" else MLPRegressor(**common)
    if name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        common = {"iterations": 100, "depth": 6, "learning_rate": 0.05,
                  "random_seed": seed, "verbose": False, "thread_count": 4}
        return CatBoostClassifier(**common) if task == "binary" else CatBoostRegressor(**common)
    raise ValueError("unknown auditor")


def measure(train_path: Path, validation_path: Path, synthetic_path: Path,
            task: str, seed: int = 1729) -> dict:
    import numpy as np
    import scipy
    import sklearn
    import catboost
    from scipy.stats import ks_2samp
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors

    if any(path.name == "test.csv" for path in (train_path, validation_path, synthetic_path)):
        raise ValueError("pilot metrics cannot open the test partition")
    start = time.perf_counter()
    real, valid, synth = (_load(path) for path in (train_path, validation_path, synthetic_path))
    if not (real.shape[1] == valid.shape[1] == synth.shape[1]):
        raise ValueError("table width mismatch")
    x_real, y_real = real[:, :-1], real[:, -1]
    x_valid, y_valid = valid[:, :-1], valid[:, -1]
    x_synth, y_synth = synth[:, :-1], synth[:, -1]
    if task == "binary" and not set(np.unique(y_synth)).issubset({0.0, 1.0}):
        raise ValueError("synthetic binary target is not binary")
    ks = [ks_2samp(valid[:, i], synth[:, i]).statistic for i in range(real.shape[1])]
    corr_real = np.nan_to_num(np.corrcoef(valid, rowvar=False))
    corr_synth = np.nan_to_num(np.corrcoef(synth, rowvar=False))
    off_diagonal = ~np.eye(real.shape[1], dtype=bool)
    pair_fidelity = float(1 - np.abs(corr_real - corr_synth)[off_diagonal].mean() / 2)
    count = min(len(valid), len(synth))
    x_c2st = np.vstack((valid[:count], synth[:count]))
    y_c2st = np.concatenate((np.zeros(count), np.ones(count)))
    x_train, x_test, c_train, c_test = train_test_split(
        x_c2st, y_c2st, test_size=0.3, random_state=seed, stratify=y_c2st)
    detector = LogisticRegression(max_iter=1000, random_state=seed).fit(x_train, c_train)
    c2st_auc = float(roc_auc_score(c_test, detector.predict_proba(x_test)[:, 1]))
    neighbors = NearestNeighbors(n_neighbors=1, n_jobs=4).fit(real)
    distance, _ = neighbors.kneighbors(synth)
    rms = distance[:, 0] / np.sqrt(real.shape[1])
    copy_counts = {"exact": int((rms <= 1e-12).sum()), "near": int((rms <= 1e-3).sum())}
    # Real-vs-real control is a descriptive false-positive check on held-out validation rows.
    control_distance, _ = neighbors.kneighbors(valid)
    control_rms = control_distance[:, 0] / np.sqrt(real.shape[1])
    control = {"exact": int((control_rms <= 1e-12).sum()),
               "near": int((control_rms <= 1e-3).sum())}
    null_prediction = float(y_real.mean())
    from sklearn.metrics import log_loss, mean_squared_error
    null_loss = (float(log_loss(y_valid, np.full(len(y_valid), np.clip(null_prediction, 1e-9, 1 - 1e-9)), labels=[0, 1]))
                 if task == "binary" else float(mean_squared_error(y_valid, np.full(len(y_valid), null_prediction))))
    utility = {}
    for name in ("catboost", "linear", "mlp"):
        try:
            real_model = _model(name, task, seed).fit(x_real, y_real)
            synth_model = _model(name, task, seed).fit(x_synth, y_synth)
            trtr = _loss(real_model, x_valid, y_valid, task)
            tstr = _loss(synth_model, x_valid, y_valid, task)
            informative = null_loss - trtr >= 0.01 * abs(null_loss)
            retention = (null_loss - tstr) / (null_loss - trtr) if informative else None
            utility[name] = {"trtr_loss": trtr, "tstr_loss": tstr,
                             "informative": informative, "retention": retention,
                             "low_signal_noninferior": tstr <= trtr + 0.01 * null_loss if not informative else None}
        except (ValueError, RuntimeError) as error:
            utility[name] = {"status": "failed", "error_type": type(error).__name__}
    return {
        "format": "dope-benchmark-validation-pilot-metrics", "version": 1,
        "implementation_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "scikit-learn": sklearn.__version__, "catboost": catboost.__version__},
        "task": task, "rows": {"train": len(real), "validation": len(valid), "synthetic": len(synth)},
        "marginal_ks_mean": float(np.mean(ks)), "pair_correlation_fidelity": pair_fidelity,
        "c2st_auc": c2st_auc, "copy_counts": copy_counts,
        "real_vs_real_control_counts": control, "null_loss": null_loss,
        "utility": utility, "metric_seconds": time.perf_counter() - start,
        "gate_profile_complete": False, "mfs_v2": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker_dir", type=Path)
    parser.add_argument("synthetic", type=Path)
    args = parser.parse_args()
    projection = json.loads((args.worker_dir / "projection.json").read_text())
    report = measure(args.worker_dir / "train.csv", args.worker_dir / "validation.csv",
                     args.synthetic, projection["task"])
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
