import unittest
import numpy as np
import torch

from research.levels.multilevel_model import LEVEL_WIDTHS, MultiLevelLandmarkModel
from research.levels.scaling import FitOnlyTableTransform


def _mock_report(n_rows: int = 240, n_features: int = 8, task: str = "regression"):
    cells = []
    for idx in range(23):
        cells.append({
            "id": idx + 1,
            "name": f"lm_{idx+1}",
            "state": "complete",
            "gain": 0.15 + 0.025 * idx,
            "se": 0.01,
            "stable": True,
            "lambda_indices": [1, 1, 2, 1, 2],
            "train_rows": [180] * 5,
            "query_rows": 45,
        })
    return {
        "panel": "landmarks-native-v1",
        "kind": "real",
        "privacy": "private",
        "task": task,
        "classes": 2 if task == "classification" else 0,
        "rows": n_rows,
        "features": n_features,
        "cells": cells,
        "target_stats": [0.4, 1.2, None],
        "spectrum": [
            {"slot": j, "values": [0.4 / (j + 1)] * 9}
            for j in range(n_features)
        ],
    }


class MultiLevelLandmarkModelTests(unittest.TestCase):
    def test_encode_discard_train_and_unlimited_synthetic_generation(self):
        rng = np.random.default_rng(123)
        n_tr = 160
        X_raw = rng.normal(size=(n_tr, 6))
        y_raw = 1.5 * X_raw[:, 0] - 0.9 * np.tanh(X_raw[:, 1]) + 0.7 * (X_raw[:, 0] * X_raw[:, 2]) + rng.normal(0.0, 0.15, size=n_tr)

        scaler = FitOnlyTableTransform.fit(X_raw, y_raw, task="regression")
        X_tr, y_tr = scaler.transform(X_raw, y_raw)
        report = _mock_report(n_rows=n_tr, n_features=6, task="regression")

        model = MultiLevelLandmarkModel()
        for level, expected_w in LEVEL_WIDTHS.items():
            enc = model.encode_table(X_tr, y_tr, report, task="regression", level=level)
            self.assertEqual(enc["code_width"], expected_w)
            self.assertEqual(enc["code_fp16"].shape, (expected_w,))
            self.assertNotIn("X_train", enc)
            self.assertNotIn("y_train", enc)

            # Generate same-size N_syn = N_train and 3x N_train
            X_syn1, y_syn1 = model.generate_synthetic(enc, n_rows=n_tr, seed=7)
            X_syn3, y_syn3 = model.generate_synthetic(enc, n_rows=3 * n_tr, seed=7)
            self.assertEqual(X_syn1.shape, (n_tr, 6))
            self.assertEqual(y_syn1.shape, (n_tr,))
            self.assertEqual(X_syn3.shape, (3 * n_tr, 6))
            self.assertEqual(y_syn3.shape, (3 * n_tr,))
            self.assertTrue(np.all(np.isfinite(X_syn3)))
            self.assertTrue(np.all(np.isfinite(y_syn3)))

    def test_gradient_flow_and_convergence_across_all_three_levels(self):
        torch.manual_seed(11)
        rng = np.random.default_rng(11)
        X = rng.normal(size=(120, 5)).astype(np.float32)
        y_reg = (1.2 * X[:, 0] + 0.8 * (X[:, 1] ** 2) - 0.5 * X[:, 2]).astype(np.float32)
        report = _mock_report(n_rows=120, n_features=5, task="regression")

        model = MultiLevelLandmarkModel()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        X_tr = torch.tensor(X[:80])
        y_tr = torch.tensor(y_reg[:80])
        X_qr = torch.tensor(X[80:])
        y_qr = torch.tensor(y_reg[80:])

        initial_loss = None
        final_loss = None
        for step in range(8):
            opt.zero_grad()
            tot = torch.tensor(0.0)
            for lvl in ("medium", "high", "ultra"):
                vec140, rep_d, tab_d = model.build_conditioning_vector(report, X[:80], y_reg[:80], level=lvl, device=torch.device("cpu"))
                emp_d = torch.tensor(0.5 * rep_d["column_prior_importance"] + 0.5 * tab_d["empirical_driver_weights"])
                out = model.forward_conditioning(X_tr, y_tr, X_qr, y_qr, vec140, emp_d, "regression", lvl)
                tot = tot + out["total"] / 3.0
            if step == 0:
                initial_loss = float(tot.item())
            tot.backward()
            opt.step()
            final_loss = float(tot.item())

        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
