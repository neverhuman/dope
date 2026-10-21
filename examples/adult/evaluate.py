"""Score synthetic Adult rows against the real training rows on the official test file.

Every learner is fitted on the real training rows, on the DOPE synthetic rows,
and on the largest random set of real rows that fits in the same number of
bytes as the DOPE artifact, and scored on the same untouched test rows. Real
rows are packed two ways and both are scored: as CSV under the better of LZMA2
and bzip2, and as 8-bit column codes (category indices, numbers quantized
between their training extremes), column-major under the same compressors,
charged with their compressed dictionary of labels and extremes. The MLP, whose
epoch budget ties its optimizer steps to the rows it sees, keeps its better
test loss of training on a subset as given or resampled to the training size
(early-stopped on held-out distinct rows, so validation never sees a duplicate).
Categorical columns are one-hot encoded (pandas.get_dummies) and numeric
columns standardized with real-training statistics, so all arms see the same
ML-ready view. Retention is the share of the real rows' log-loss gain over the
class prior that each arm recovers (unclipped: below zero is worse than the
prior).

    python3 examples/adult/evaluate.py --dir target/adult --model target/adult/adult.dope
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

TARGET, POSITIVE, SEED = "income", ">50K", 1729


def learners():
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    return {
        "logistic": lambda: LogisticRegression(max_iter=2000),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, early_stopping=True,
                                     n_iter_no_change=15, random_state=SEED),
        "catboost": lambda: CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, random_seed=SEED,
                                               verbose=False, thread_count=2, allow_writing_files=False),
    }


def ml_view(frames: list[pd.DataFrame], reference: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """One-hot categoricals and standardized numerics on a shared column set."""
    features = [f.drop(columns=[TARGET]) for f in frames]
    numeric = [c for c in reference.columns if c != TARGET and pd.api.types.is_numeric_dtype(reference[c])]
    stats = reference[numeric].agg(["mean", "std"])
    stacked = pd.concat(features, keys=range(len(features)))
    for column in numeric:
        stacked[column] = (pd.to_numeric(stacked[column], errors="coerce") - stats.loc["mean", column]) / (
            stats.loc["std", column] or 1.0)
    stacked[numeric] = stacked[numeric].fillna(0.0)
    encoded = pd.get_dummies(stacked, columns=[c for c in stacked.columns if c not in numeric], dtype=float)
    return [(encoded.xs(i).to_numpy(), (f[TARGET] == POSITIVE).to_numpy().astype(int))
            for i, f in enumerate(frames)]


def read(path: Path) -> pd.DataFrame:
    """Read every cell as text, then make fully numeric columns numeric."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in frame.columns:
        numbers = pd.to_numeric(frame[column].replace("", np.nan), errors="coerce")
        if numbers.notna().sum() == (frame[column] != "").sum():
            frame[column] = numbers
    return frame


def compressed_sizes(path: Path) -> dict:
    raw = path.read_bytes()
    return {"csv": len(raw), "gzip": len(gzip.compress(raw, 9)), "bzip2": len(bz2.compress(raw, 9)),
            "xz": len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME))}


def best_compressed(payload: bytes) -> int:
    xz = lzma.compress(payload, format=lzma.FORMAT_RAW,
                       filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}])
    return min(len(xz), len(bz2.compress(payload, 9)))


class Codes:
    """8-bit column codes: category index or a number quantized between its training extremes."""

    def __init__(self, train: pd.DataFrame):
        self.levels, self.bounds = {}, {}
        for column in train.columns:
            if pd.api.types.is_numeric_dtype(train[column]):
                self.bounds[column] = (float(train[column].min()), float(train[column].max()))
            else:
                self.levels[column] = sorted(train[column].astype(str).unique())
        dictionary = json.dumps({"levels": self.levels, "bounds": self.bounds}, separators=(",", ":")).encode()
        self.dictionary_bytes = best_compressed(dictionary)

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for column in frame.columns:
            if column in self.bounds:
                low, high = self.bounds[column]
                unit = (frame[column].fillna(low).to_numpy(float) - low) / ((high - low) or 1.0)
                columns.append(np.round(np.clip(unit, 0, 1) * 255).astype(np.uint8))
            else:
                index = {level: i for i, level in enumerate(self.levels[column])}
                columns.append(frame[column].astype(str).map(index).to_numpy(np.uint8))
        return np.stack(columns, axis=1)

    def size(self, frame: pd.DataFrame) -> int:
        codes = self.encode(frame)
        codes = codes[np.lexsort(codes.T[::-1])]
        return best_compressed(np.ascontiguousarray(codes.T).tobytes()) + self.dictionary_bytes

    def decode(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The values a receiver of the codes would see: numbers back on their 8-bit grid."""
        out = frame.copy()
        for column, (low, high) in self.bounds.items():
            codes = self.encode(frame[[column]])[:, 0]
            out[column] = low + codes / 255 * (high - low)
        return out


def rows_in_budget(train: pd.DataFrame, budget: int, size) -> pd.DataFrame:
    """Largest random subset whose packed size fits the budget."""
    shuffled = train.sample(frac=1.0, random_state=SEED)
    low, high = 2, len(shuffled)
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if size(shuffled.iloc[:middle]) <= budget else (low, middle)
    return shuffled.iloc[:low]


def csv_size(frame: pd.DataFrame) -> int:
    return best_compressed(frame.to_csv(index=False).encode())


def resampled_mlp(x: np.ndarray, y: np.ndarray, x_test: np.ndarray, size: int) -> np.ndarray:
    """MLP on the rows resampled to `size`, early-stopped on held-out distinct rows (best epoch kept)."""
    import copy
    from sklearn.neural_network import MLPClassifier
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))
    hold = max(2, len(y) // 10)
    held, kept = order[:hold], order[hold:]
    index = kept[rng.integers(0, len(kept), size)]
    model = MLPClassifier(hidden_layer_sizes=(32,), random_state=SEED)
    best, state, stale = np.inf, None, 0
    for _ in range(200):
        model.partial_fit(x[index], y[index], classes=np.array([0, 1]))
        loss = log_loss(y[held], np.clip(model.predict_proba(x[held])[:, 1], 1e-9, 1 - 1e-9), labels=[0, 1])
        if loss < best:
            best, state, stale = loss, (copy.deepcopy(model.coefs_), copy.deepcopy(model.intercepts_)), 0
        else:
            stale += 1
            if stale >= 15:
                break
    model.coefs_, model.intercepts_ = state
    return model.predict_proba(x_test)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path)
    args = parser.parse_args()
    train, test = read(args.dir / "train.csv"), read(args.dir / "test.csv")
    synthetic = read(args.synthetic or args.dir / "synthetic.csv")
    artifact_bytes = args.model.stat().st_size
    codes = Codes(train)
    subsets = {"csv": rows_in_budget(train, artifact_bytes, csv_size),
               "codes": codes.decode(rows_in_budget(train, artifact_bytes, codes.size))}
    arms = {"real rows": train, "DOPE synthetic": synthetic,
            **{f"real rows in {artifact_bytes} B ({name})": rows for name, rows in subsets.items()}}
    views = ml_view([*arms.values(), test], train)
    x_test, y_test = views[-1]
    prior = np.full(len(y_test), views[0][1].mean())
    null = log_loss(y_test, prior)
    results = {"artifact_bytes": artifact_bytes, "subset_rows": {name: len(rows) for name, rows in subsets.items()},
               "train_file": compressed_sizes(args.dir / "train.csv"), "null_log_loss": null, "arms": {}}
    for arm, (x, y) in zip(arms, views[:-1]):
        results["arms"][arm] = {}
        for name, make in learners().items():
            probabilities = [make().fit(x, y).predict_proba(x_test)[:, 1]]
            if name == "mlp" and len(y) < len(views[0][1]):
                probabilities.append(resampled_mlp(x, y, x_test, len(views[0][1])))
            scores = [{"log_loss": log_loss(y_test, np.clip(p, 1e-9, 1 - 1e-9)), "auc": roc_auc_score(y_test, p)}
                      for p in probabilities]
            results["arms"][arm][name] = min(scores, key=lambda score: score["log_loss"])
    real = results["arms"]["real rows"]
    print(f"artifact {artifact_bytes:,} B | train.csv {results['train_file']['csv']:,} B "
          f"(gzip {results['train_file']['gzip']:,} B) | equal-byte real subsets: {results['subset_rows']}")
    print(f"{'arm':28s} " + " ".join(f"{n:>22s}" for n in real))
    for arm, scores in results["arms"].items():
        cells = []
        for name, score in scores.items():
            retention = (null - score["log_loss"]) / (null - real[name]["log_loss"])
            score["retention"] = retention
            cells.append(f"ret {retention:5.3f} AUC {score['auc']:.3f}")
        print(f"{arm:28s} " + " ".join(f"{c:>22s}" for c in cells))
    (args.dir / "evaluation.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
