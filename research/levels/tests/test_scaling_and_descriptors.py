import json
import unittest
import numpy as np

from research.levels.scaling import FitOnlyTableTransform
from research.levels.descriptors import (
    FINE_ADDED_INDICES,
    LARGER_ADDED_INDICES,
    MACRO_INDICES,
    compute_table_descriptors,
    extract_report_descriptors,
    level_landmark_mask,
)


class ScalingAndDescriptorsTests(unittest.TestCase):
    def test_fit_only_table_transform_is_leak_free_and_anonymized(self):
        rng = np.random.default_rng(42)
        n_train = 120
        # Col 0: heavy-tailed lognormal
        c0_train = np.exp(rng.normal(0.0, 2.0, size=n_train))
        # Col 1: standard Gaussian with missing values
        c1_train = rng.normal(10.0, 3.0, size=n_train)
        c1_train[0] = np.nan
        # Col 2: categorical with frequent and rare levels
        cats = ["alpha"] * 50 + ["beta"] * 45 + ["gamma"] * 20 + ["rare_secret_id_999"] * 5
        y_train = (0.8 * np.log1p(c0_train) + 0.3 * np.nan_to_num(c1_train, nan=10.0) + rng.normal(0.0, 0.5, size=n_train))

        X_train = [[float(c0_train[i]), float(c1_train[i]), cats[i]] for i in range(n_train)]
        scaler = FitOnlyTableTransform.fit(
            X_train,
            y_train,
            kinds=("numeric", "numeric", "categorical"),
            task="regression",
            min_level_count=6,
        )

        # Verify heavy-tailed column got signed_log1p_hybrid
        self.assertEqual(scaler.numeric_scalers[0].transform_kind, "signed_log1p_hybrid")
        # Verify rare category (<6) was pooled into __other__
        self.assertNotIn("rare_secret_id_999", scaler.categorical_encoders[2].vocab_tokens)

        # Verify anonymous metadata contains zero raw strings
        meta_str = json.dumps(scaler.anonymous_metadata())
        self.assertNotIn("alpha", meta_str)
        self.assertNotIn("rare_secret_id_999", meta_str)

        # Verify test transform is row-independent and immune to other test rows
        X_test_clean = [[2.5, 11.0, "alpha"], [1.2, 9.5, "unseen_test_cat"]]
        X_test_poisoned = [[2.5, 11.0, "alpha"], [1e12, -1e12, "unseen_test_cat_2"]]
        Z_clean, _ = scaler.transform(X_test_clean, [1.0, 2.0])
        Z_poisoned, _ = scaler.transform(X_test_poisoned, [1.0, 999999.0])
        np.testing.assert_allclose(Z_clean[0], Z_poisoned[0], rtol=0, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(Z_poisoned)))
        self.assertLessEqual( float(np.max(np.abs(Z_poisoned))), 6.000001 )

    def test_three_level_landmark_and_descriptor_hierarchy(self):
        cells = []
        for idx in range(23):
            cells.append({
                "id": idx + 1,
                "name": f"lm_{idx+1}",
                "state": "complete",
                "gain": 0.10 + 0.02 * idx,
                "se": 0.01,
                "stable": True,
                "lambda_indices": [1, 2, 1, 2, 1],
                "train_rows": [100, 100, 100, 100, 100],
                "query_rows": 125,
            })
        report = {
            "panel": "landmarks-native-v1",
            "kind": "real",
            "privacy": "private",
            "task": "classification",
            "classes": 2,
            "rows": 500,
            "features": 6,
            "cells": cells,
            "target_stats": [0.95, 0.38, None],
            "spectrum": [
                {"slot": j, "values": [0.1 * (j + 1)] * 9}
                for j in range(6)
            ],
        }

        med = extract_report_descriptors(report, level="medium")
        high = extract_report_descriptors(report, level="high")
        ultra = extract_report_descriptors(report, level="ultra")

        self.assertEqual(int(np.sum(med["mask23"])), len(MACRO_INDICES))
        self.assertEqual(int(np.sum(high["mask23"])), len(MACRO_INDICES) + len(LARGER_ADDED_INDICES))
        self.assertEqual(int(np.sum(ultra["mask23"])), 23)

        # Fine indices must be zeroed at Medium and High
        for fine_idx in FINE_ADDED_INDICES:
            self.assertEqual(float(med["gains23"][fine_idx]), 0.0)
            self.assertEqual(float(high["gains23"][fine_idx]), 0.0)
            self.assertGreater(float(ultra["gains23"][fine_idx]), 0.0)

        # Larger indices must be zeroed at Medium, active at High and Ultra
        for larger_idx in LARGER_ADDED_INDICES:
            self.assertEqual(float(med["gains23"][larger_idx]), 0.0)
            self.assertGreater(float(high["gains23"][larger_idx]), 0.0)

        # Table descriptors respect level hierarchy
        rng = np.random.default_rng(7)
        X = rng.normal(size=(80, 6))
        y = X[:, 0] + 0.5 * (X[:, 1] ** 2) + rng.normal(scale=0.1, size=80)
        td_med = compute_table_descriptors(X, y, level="medium")
        td_high = compute_table_descriptors(X, y, level="high")
        td_ultra = compute_table_descriptors(X, y, level="ultra")
        self.assertEqual(td_med["table_vec24"].shape, (24,))
        self.assertTrue(np.allclose(td_med["table_vec24"][8:], 0.0))
        self.assertTrue(np.allclose(td_high["table_vec24"][16:], 0.0))
        self.assertGreater(float(np.linalg.norm(td_ultra["table_vec24"][16:])), 0.0)


if __name__ == "__main__":
    unittest.main()
