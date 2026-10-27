"""Unit tests for multiclass K>=33 / non-contiguous / fractional / absent-class handling and decoupled supervision_desc_vec."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from research.levels.budget_planner import deserialize_dope_v3_envelope, serialize_dope_v3_envelope
from research.levels.multiclass_head import fit_multiclass_target_bundle, predict_multiclass_from_bundle
from research.levels.multilevel_model import MultiLevelLandmarkModel
from research.levels.optional_landmarks import build_conditioning, observed_target_stats
from research.levels.scaling import FitOnlyTableTransform


class TestMulticlassAndDropoutSupervision(unittest.TestCase):
    def test_multiclass_cap32_default_and_lossless_k3_k32_k36(self) -> None:
        rng = np.random.default_rng(20261027)
        for k_classes in (3, 32, 36):
            n = k_classes * 4
            X = rng.normal(size=(n, 5))
            # Non-contiguous fractional & negative/positive raw class labels
            raw_vocab = np.array([float(i * 0.5 - 5.0) for i in range(k_classes)], dtype=np.float64)
            y_raw = np.repeat(raw_vocab, 4)

            # Default policy caps at 32 classes (mc_many legacy compatibility)
            tf_cap = FitOnlyTableTransform.fit(X, y_raw, task="multiclass")
            assert tf_cap.target_scaler is not None
            self.assertEqual(len(tf_cap.target_scaler.class_labels), min(k_classes, 32))

            # Lossless policy preserves all K>=33 classes
            tf = FitOnlyTableTransform.fit(X, y_raw, task="multiclass", max_multiclass_classes=None)
            assert tf.target_scaler is not None
            self.assertEqual(len(tf.target_scaler.class_labels), k_classes)
            np.testing.assert_allclose(np.asarray(tf.target_scaler.class_labels), raw_vocab)

            X_sc, y_sc = tf.transform(X, y_raw, strict_unknown=True)
            assert y_sc is not None
            self.assertEqual(len(np.unique(y_sc)), k_classes)
            _, y_inv = tf.inverse_transform(X_sc, y_sc)
            assert y_inv is not None
            np.testing.assert_allclose(y_inv, y_raw)

            # Strict unknown label check rejects unseen or NaN query labels
            with self.assertRaises(ValueError):
                tf.transform_target([999.0], strict_unknown=True)
            with self.assertRaises(ValueError):
                tf.transform_target([float("nan")], strict_unknown=True)

            # Fractional/close labels and label permutations preserve target_stats
            ts_orig = observed_target_stats(y_sc, "multiclass", y_raw=y_raw)
            self.assertAlmostEqual(float(ts_orig[0]), 1.0, places=6)
            self.assertAlmostEqual(float(ts_orig[1]), 1.0 / float(k_classes), places=6)
            y_perm_raw = np.repeat(raw_vocab[::-1], 4)
            tf_perm = FitOnlyTableTransform.fit(X, y_perm_raw, task="multiclass", max_multiclass_classes=None)
            _, y_perm_sc = tf_perm.transform(X, y_perm_raw, strict_unknown=True)
            assert y_perm_sc is not None
            ts_perm = observed_target_stats(y_perm_sc, "multiclass", y_raw=y_perm_raw)
            self.assertAlmostEqual(float(ts_perm[0]), float(ts_orig[0]), places=12)
            self.assertAlmostEqual(float(ts_perm[1]), float(ts_orig[1]), places=12)

            # Verify LR-VLOT multiclass target bundle and DML3/CSC1 round-trip preserve all K classes
            gates = {"monotone_w": 0.55, "hinge_w": 0.50, "pair_w": 0.50, "tree_w": 0.80, "ridge_alpha": 0.25, "noise_scale": 0.15}
            col_prior = np.full(5, 0.2, dtype=np.float64)
            tb = fit_multiclass_target_bundle(X_sc, y_sc, gates=gates, col_prior=col_prior, emp_drivers=col_prior, level="high")
            self.assertEqual(tb["k_classes"], k_classes)
            y_syn_idx = predict_multiclass_from_bundle(X_sc, tb, level="high", rng=np.random.default_rng(7))
            self.assertEqual(len(np.unique(y_syn_idx)), k_classes)
            _, y_syn_raw = tf.inverse_transform(X_sc, y_syn_idx)
            assert y_syn_raw is not None
            self.assertEqual(set(np.round(y_syn_raw, 4).tolist()), set(np.round(raw_vocab, 4).tolist()))

            model = MultiLevelLandmarkModel(max_cols=16, hidden_dim=64, latent_dim=128)
            enc = model.encode_table(X_sc, y_sc, task="multiclass", level="high", landmark_mode="none")
            wire = serialize_dope_v3_envelope(enc, scaler=tf)
            dec_enc, dec_tf = deserialize_dope_v3_envelope(wire)
            assert dec_tf is not None and dec_tf.target_scaler is not None
            self.assertEqual(dec_enc["target_bundle"]["k_classes"], k_classes)
            self.assertEqual(len(dec_tf.target_scaler.class_labels), k_classes)

    def test_multiclass_forward_conditioning_k33_and_absent_query_classes(self) -> None:
        torch.manual_seed(0)
        rng = np.random.default_rng(42)
        n_tr, n_qr, p = 99, 30, 6
        X_tr = torch.tensor(rng.normal(size=(n_tr, p)), dtype=torch.float32)
        X_qr = torch.tensor(rng.normal(size=(n_qr, p)), dtype=torch.float32)
        # 33 classes in train (0..32), only 5 classes present in query ({0, 5, 12, 24, 32})
        y_tr = torch.tensor(np.repeat(np.arange(33, dtype=np.float32), 3), dtype=torch.float32)
        y_qr = torch.tensor(np.tile(np.array([0, 5, 12, 24, 32], dtype=np.float32), 6), dtype=torch.float32)

        cond = build_conditioning(X_tr.numpy(), y_tr.numpy(), task="multiclass", mode="none")
        desc_vec = torch.tensor(cond["vec256"], dtype=torch.float32)
        col_lm = torch.tensor(cond["col_landmarks"], dtype=torch.float32)
        emp = torch.full((p,), 1.0 / p, dtype=torch.float32)

        model = MultiLevelLandmarkModel(max_cols=64, hidden_dim=128, latent_dim=128)
        out = model.forward_conditioning(
            X_tr, y_tr, X_qr, y_qr, desc_vec, emp, "multiclass", "high", col_landmarks=col_lm, k_categories=35,
        )
        self.assertTrue(torch.isfinite(out["total"]).item())
        out["total"].backward()
        grad_norm = sum(float(param.grad.norm().item()) for param in model.parameters() if param.grad is not None)
        self.assertGreater(grad_norm, 0.0)

        # Unknown query class labels (outside train vocab) must not resize K or raise out-of-bounds
        y_qr_unknown = y_qr.clone()
        y_qr_unknown[0] = 99.0
        y_qr_unknown[1] = -1.0
        out_unk = model.forward_conditioning(
            X_tr, y_tr, X_qr, y_qr_unknown, desc_vec, emp, "multiclass", "high", col_landmarks=col_lm,
        )
        self.assertTrue(torch.isfinite(out_unk["total"]).item())

    def test_supervision_desc_vec_decoupled_from_router_input(self) -> None:
        torch.manual_seed(11)
        rng = np.random.default_rng(11)
        X_tr = torch.tensor(rng.normal(size=(40, 4)), dtype=torch.float32)
        y_tr = torch.tensor(rng.normal(size=(40,)), dtype=torch.float32)
        X_qr = torch.tensor(rng.normal(size=(20, 4)), dtype=torch.float32)
        y_qr = torch.tensor(rng.normal(size=(20,)), dtype=torch.float32)
        emp = torch.full((4,), 0.25, dtype=torch.float32)

        cond_none = build_conditioning(X_tr.numpy(), y_tr.numpy(), task="regression", mode="none")
        v_none = torch.tensor(cond_none["vec256"], dtype=torch.float32)
        c_none = torch.tensor(cond_none["col_landmarks"], dtype=torch.float32)

        v_sup = v_none.clone()
        v_sup[:23] = torch.linspace(0.05, 0.85, 23)
        v_sup[23:46] = 1.0
        v_sup[188:204] = torch.linspace(-0.3, 0.6, 16)

        model = MultiLevelLandmarkModel(max_cols=64, hidden_dim=64, latent_dim=128)
        out_no_sup = model.forward_conditioning(
            X_tr, y_tr, X_qr, y_qr, v_none, emp, "regression", "high", col_landmarks=c_none,
        )
        out_with_sup = model.forward_conditioning(
            X_tr, y_tr, X_qr, y_qr, v_none, emp, "regression", "high", col_landmarks=c_none, supervision_desc_vec=v_sup,
        )
        self.assertEqual(out_no_sup["gates"], out_with_sup["gates"])
        self.assertNotAlmostEqual(
            float(out_no_sup["landmark_loss"].item()),
            float(out_with_sup["landmark_loss"].item()),
            places=5,
        )

    def test_trainer_full_partition_conditioning_parity(self) -> None:
        from research.levels.train_landmark_dropout import (
            ALL_TIERS, _fast_observed_conditioning, _mask_bundle, prepare_full_partition_cell
        )
        from research.levels.optional_landmarks import LandmarkBundle

        rng = np.random.default_rng(99)
        X_part = rng.normal(size=(60, 4)).astype(np.float64)
        y_part_raw = (rng.normal(size=(60,)) * 10.0 + 5.0).astype(np.float64)

        cells_rep = [{"id": i + 1, "state": "complete", "gain": 0.05 * (i + 1), "se": 0.01, "stable": True, "lambda_indices": [1, 1, 1, 1, 1]} for i in range(23)]
        spec_rep = [{"slot": j, "values": [0.1 * (j + 1)] * 9} for j in range(4)]
        for task_name, k_cls, y_raw_case in (
            ("regression", 0, y_part_raw),
            ("classification", 2, np.array([0.0 if i % 2 == 0 else 1.0 for i in range(60)], dtype=np.float64)),
            ("multiclass", 3, np.array([0.0, 0.5, 1.0] * 20, dtype=np.float64)),
        ):
            sc_case = FitOnlyTableTransform.fit(X_part, y_raw_case, task=task_name, max_multiclass_classes=None)
            X_sc, y_sc = sc_case.transform(X_part, y_raw_case, strict_unknown=(task_name == "multiclass"))
            X_sc = np.clip(np.nan_to_num(X_sc, nan=0.0, posinf=6.0, neginf=-6.0), -6.0, 6.0)
            y_sc = np.asarray(y_sc, dtype=np.float64)
            rep_task = "regression" if task_name == "regression" else "classification"
            rep_t_stats = [float(np.mean(y_raw_case)), float(np.std(y_raw_case)), None] if task_name == "regression" else [1.0, 1.0 / max(k_cls, 2), None]
            rep = {
                "panel": "landmarks-native-v1", "kind": "real", "privacy": "private",
                "task": rep_task, "classes": k_cls, "rows": 60, "features": 4,
                "cells": cells_rep, "target_stats": rep_t_stats,
                "spectrum": spec_rep,
            }
            bun = LandmarkBundle.from_report(rep, source_digest="a" * 64)
            cell = prepare_full_partition_cell(
                did=f"test_{task_name}", split=f"{task_name}-fit-75-v1", task=task_name,
                X_part_sc=X_sc, y_part_sc=y_sc, y_part_raw=y_raw_case,
                X_fit_ep=X_sc[:32].astype(np.float32), y_fit_ep=y_sc[:32].astype(np.float32),
                X_qry_ep=X_sc[32:].astype(np.float32), y_qry_ep=y_sc[32:].astype(np.float32),
                k_classes=k_cls, rep_full=rep, bun_full=bun,
            )
            if task_name == "multiclass":
                self.assertAlmostEqual(float(cell["target_stats"][0]), 1.0, places=6)
                self.assertAlmostEqual(float(cell["target_stats"][1]), 1.0 / 3.0, places=6)
            for lvl in ALL_TIERS:
                c_full_ref = build_conditioning(
                    X_sc, y_sc, task=task_name, bundle=bun, mode="full",
                    native_report=rep, level=lvl, y_train_raw=y_raw_case,
                )
                np.testing.assert_array_equal(cell[f"vec256_{lvl}"], c_full_ref["vec256"])
                np.testing.assert_array_equal(cell[f"col17_{lvl}"], c_full_ref["col_landmarks"])
                for view in ("none", "macro", "macro+larger", "random_sparse", "random_dense"):
                    b_m = _mask_bundle(bun, view, rng_seed=12345)
                    v_np, c_np = _fast_observed_conditioning(
                        b_m, full_vec256=cell[f"vec256_{lvl}"], obs_col17=cell[f"obs_col17_{lvl}"],
                        rows=cell["partition_rows"], features=cell["features"], classes=cell["classes"],
                        task=task_name, target_stats=cell["target_stats"], level=lvl,
                    )
                    mode_ref = "none" if view == "none" else "progressive"
                    c_ref = build_conditioning(
                        X_sc, y_sc, task=task_name, bundle=(None if view == "none" else b_m),
                        mode=mode_ref, level=lvl, y_train_raw=y_raw_case,
                    )
                    np.testing.assert_array_equal(v_np, c_ref["vec256"])
                    np.testing.assert_array_equal(c_np, c_ref["col_landmarks"])

    def test_numeric_scaler_ieee754_tanh_tail_parity(self) -> None:
        import math
        from research.levels.scaling import NumericColumnScaler

        scaler = NumericColumnScaler(
            slot=0,
            anonymous_name="f_000",
            transform_kind="linear",
            missing_rate=0.0,
            center=0.0,
            scale=1.0,
            post_mean=0.0,
            post_std=1.0,
            quantiles=(0.0,) * 9,
            skewness=0.0,
            kurtosis=0.0,
            blend_quantile_weight=0.35,
        )
        # u = z / 6.0 = 19.040037931519883 falls in [18.9903411, 19.0615475) where
        # numpy.tanh saturates to 1.0 early while IEEE-754 f64::tanh (and math.tanh)
        # stays strictly < 1.0, preserving exact rank/tie order with Rust dope-core.
        u = 19.040037931519883
        vals = np.array([-u * 6.0, 18.2 * 6.0, u * 6.0, 20.0 * 6.0], dtype=np.float64)
        scaled, _ = scaler.transform(vals)
        self.assertEqual(float(scaled[0]), 6.0 * math.tanh(-u))
        self.assertEqual(float(scaled[1]), 6.0 * math.tanh(18.2))
        self.assertEqual(float(scaled[2]), 6.0 * math.tanh(u))
        self.assertLess(abs(float(scaled[2])), 6.0)
        self.assertEqual(float(scaled[3]), 6.0)

        # Verify end-to-end none-mode conditioning spearman_r rank order when a column has
        # a cell at u=19.040037931519883 alongside saturated cells at u=22.0.
        n = 32
        x_raw = np.linspace(-2.0, 2.0, n, dtype=np.float64).reshape(n, 1)
        x_raw[-3:, 0] = 22.0 * 6.0
        x_raw[-4, 0] = u * 6.0
        x_sc, _ = scaler.transform(x_raw[:, 0])
        x_sc = x_sc.reshape(n, 1)
        y_sc = np.array([float(i % 4) for i in range(n)], dtype=np.float64)
        cond = build_conditioning(x_sc, y_sc, task="multiclass", bundle=None, mode="none")
        self.assertTrue(np.all(np.isfinite(cond["vec256"])))
        self.assertTrue(np.all(np.isfinite(cond["col_landmarks"])))
        # Row n-4 is strictly < 6.0 so its ordinal rank is n-4 (28), not tied with the top 3 rows.
        rx = np.argsort(np.argsort(x_sc[:, 0]))
        self.assertEqual(int(rx[-4]), n - 4)


if __name__ == "__main__":
    unittest.main()
