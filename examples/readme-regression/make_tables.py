#!/usr/bin/env python3
"""Write the README toy tables.

Headerless numeric CSV. The target is the last column. Every value is
already inside [0, 1], which is what ``compile`` requires at the default
tier. This toy is not the 97-lineage panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SEED = 20261008
TRAIN_ROWS = 400
TEST_ROWS = 100


def table(seed, count):
    """Four columns: x1, x2, an unused x3, and y.

    y = 0.55 x1 + 0.25 x2 + 0.10 u, with u uniform noise that is not stored.
    The coefficients keep y inside [0, 0.90].
    """
    generator = np.random.default_rng(seed)
    x1 = generator.random(count)
    x2 = generator.random(count)
    x3 = generator.random(count)
    noise = generator.random(count)
    target = 0.55 * x1 + 0.25 * x2 + 0.10 * noise
    values = np.column_stack((x1, x2, x3, target))
    if values.shape != (count, 4):
        raise RuntimeError("toy table has the wrong shape")
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise RuntimeError("toy table left the unit interval")
    return values


def write(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(directory / "train.csv", table(SEED, TRAIN_ROWS), delimiter=",", fmt="%.8f")
    np.savetxt(
        directory / "test.csv",
        table(SEED + 1, TEST_ROWS),
        delimiter=",",
        fmt="%.8f",
    )


def main():
    write(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
