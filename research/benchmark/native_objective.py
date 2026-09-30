"""Validation likelihood KPI for the three already locked density baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .score import sha256


FLOOR = 1e-300


def _index(value: float, card: int) -> int:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("validation value outside common numeric support")
    return min(int(value * card), card - 1)


def _rows(path: Path):
    with path.open(newline="") as stream:
        width = None
        for row in csv.reader(stream):
            if width is None:
                width = len(row)
            if len(row) != width or not row:
                raise ValueError("malformed validation table")
            try:
                yield [float(value) for value in row]
            except ValueError:
                raise ValueError("nonnumeric validation table") from None


def _marginal_log_density(row: list[float], model: dict) -> float:
    if len(row) != len(model["columns"]):
        raise ValueError("validation width differs from marginal model")
    total = 0.0
    for value, column in zip(row, model["columns"]):
        if column["kind"] == "binary":
            if value not in (0.0, 1.0):
                raise ValueError("binary validation target is invalid")
            probability = column["p"] if value == 1.0 else 1 - column["p"]
        else:
            probability = model["bins"] * column["p"][_index(value, model["bins"])]
        total += math.log(max(FLOOR, probability))
    return total


def _chow_log_density(row: list[float], model: dict) -> float:
    cards = model["cards"]
    if len(row) != len(cards):
        raise ValueError("validation width differs from Chow-Liu model")
    codes = [_index(value, card) for value, card in zip(row, cards)]
    total = math.log(max(FLOOR, model["root_probs"][codes[0]]))
    for edge in model["edges"]:
        total += math.log(max(FLOOR, edge["conditional"][codes[edge["parent"]]][codes[edge["child"]]]))
    total += sum(math.log(card) for card in cards if card != 2)
    return total


def _log_density(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("native likelihood is nonfinite or negative")
    return math.log(max(FLOOR, value))


def score(method: str, artifact_dir: Path, validation: Path) -> dict:
    if method not in ("GaussianCopula", "independent_marginals", "Chow-Liu"):
        raise ValueError("method has no locked native likelihood KPI")
    artifact = artifact_dir / "model.json"
    model = json.loads(artifact.read_text())
    count = 0
    total = 0.0
    if method == "GaussianCopula":
        import pandas as pd
        from copulas.multivariate import GaussianMultivariate
        fitted = GaussianMultivariate.from_dict(model["model"])
        batch = []
        for row in _rows(validation):
            batch.append(row)
            if len(batch) == 1024:
                frame = pd.DataFrame(batch, columns=model["model"]["columns"])
                total += sum(_log_density(float(value))
                             for value in fitted.probability_density(frame))
                count += len(batch)
                batch.clear()
        if batch:
            frame = pd.DataFrame(batch, columns=model["model"]["columns"])
            total += sum(_log_density(float(value))
                         for value in fitted.probability_density(frame))
            count += len(batch)
    else:
        scorer = _marginal_log_density if method == "independent_marginals" else _chow_log_density
        for row in _rows(validation):
            total += scorer(row, model)
            count += 1
    if count == 0 or not math.isfinite(total):
        raise ValueError("native likelihood is undefined")
    return {"format": "dope-benchmark-native-validation-kpi", "version": 1,
            "method": method, "partition": "validation", "objective": "mean_log_density",
            "direction": "maximize", "rows": count, "value": total / count,
            "implementation_sha256": sha256(Path(__file__)),
            "artifact_sha256": sha256(artifact), "validation_sha256": sha256(validation)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method")
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.method, args.artifact_dir, args.validation), sort_keys=True))


if __name__ == "__main__":
    main()
