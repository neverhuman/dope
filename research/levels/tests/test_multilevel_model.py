import unittest
import zlib
import numpy as np
import torch

from research.levels.multilevel_model import LEVEL_WIDTHS, MultiLevelLandmarkModel
from research.levels.scaling import FitOnlyTableTransform
from research.levels.strata_audit import (
    AUDITORS,
    LEVELS,
    METHODS,
    deserialize_dope_latent_wire_bytes,
    estimate_scaler_sidecar_bytes,
    serialize_dope_latent_wire_bytes,
    summarize_cohort_records,
)
from research.levels.world_class_eval import _KumoTabularWrapper, _TabularKumoAttentionNet


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

    def test_strata_summary_and_local_attention_auditor_attribution(self):
        rng = np.random.default_rng(77)
        X_toy = rng.normal(size=(40, 4)).astype(np.float32)
        y_toy = (X_toy[:, 0] > 0.0).astype(np.int64)
        wrapper = _KumoTabularWrapper(task="classification", seed=77).fit(X_toy[:30], y_toy[:30], X_toy[30:])
        self.assertIsInstance(wrapper.model, _TabularKumoAttentionNet)
        self.assertEqual(wrapper.get_feature_importance().shape, (4,))
        self.assertAlmostEqual(float(wrapper.get_feature_importance().sum()), 1.0, places=5)

        mock_records = []
        for idx, is_val in enumerate((True, False, True, False)):
            cells = {}
            elo_cells = {}
            for lvl_idx, lvl in enumerate(LEVELS):
                sc = {}
                for m_idx, m in enumerate(METHODS):
                    r_val = 0.88 - 0.05 * m_idx + 0.02 * lvl_idx
                    d_val = 0.81 - 0.04 * m_idx + 0.01 * lvl_idx
                    btr_val = 100.0 * (0.75 * r_val + 0.25 * d_val)
                    cells[f"{m}_{lvl}"] = {
                        "R": r_val, "D": d_val, "rho": 0.65, "top3": 0.67, "btr": btr_val,
                        "acc": 0.78 if idx % 2 == 0 else None,
                        "bacc": 0.76 if idx % 2 == 0 else None,
                        "roc_auc": 0.83 if idx % 2 == 0 else None,
                        "r2": 0.54 if idx % 2 == 1 else None,
                    }
                    sc[m] = btr_val
                for aud in ("catboost", "lightgbm", "linear"):
                    elo_cells[f"{lvl}|{aud}"] = dict(sc)
            auds = {
                aud: {
                    "R": 0.89, "D": 0.82, "rho": 0.68, "top3": 0.67,
                    "real_score": 0.80, "syn_score": 0.74,
                    "acc_real": 0.79 if idx % 2 == 0 else None,
                    "acc_syn": 0.78 if idx % 2 == 0 else None,
                }
                for aud in AUDITORS
            }
            mock_records.append({
                "dataset_id": f"ds_{idx}",
                "task": "classification" if idx % 2 == 0 else "regression",
                "is_val_dataset": is_val,
                "cells": cells,
                "elo_cells": elo_cells,
                "auditors_ultra": auds,
                "expansion": {"n1_R": 0.85, "n2_R": 0.87, "n5_R": 0.89},
            })

        summary = summarize_cohort_records(mock_records, n_boot=16, seed=42)
        self.assertEqual(summary["datasets_evaluated"], 4)
        self.assertEqual(summary["elo_leaderboard_overall"][0]["method"], "dope_mlc_v2")
        self.assertGreater(summary["elo_leaderboard_overall"][0]["elo"], 1500.0)
        self.assertEqual(len(summary["cross_size_elo_by_artifact_bytes"]), len(METHODS))

    def test_wire_serialization_isolated_receiver_roundtrip_and_parity(self):
        rng = np.random.default_rng(2026)
        n_tr = 140
        X_raw = rng.normal(size=(n_tr, 7))
        y_raw = 1.4 * X_raw[:, 0] - 0.8 * (X_raw[:, 1] ** 2) + 0.5 * X_raw[:, 2] + rng.normal(0.0, 0.1, size=n_tr)

        scaler = FitOnlyTableTransform.fit(X_raw, y_raw, task="regression")
        X_tr, y_tr = scaler.transform(X_raw, y_raw)
        sidecar = estimate_scaler_sidecar_bytes(scaler, 7)
        self.assertGreater(sidecar["scaler_sidecar_raw_bytes"], 200)

        sender_model = MultiLevelLandmarkModel().eval()
        receiver_model = MultiLevelLandmarkModel().eval()
        receiver_model.load_state_dict(sender_model.state_dict())

        report = _mock_report(n_rows=n_tr, n_features=7, task="regression")
        for level in ("medium", "high", "ultra"):
            enc_mem = sender_model.encode_table(X_tr, y_tr, report, task="regression", level=level)
            raw_wire = serialize_dope_latent_wire_bytes(enc_mem)
            z_wire = zlib.compress(raw_wire, level=9)
            dec_wire = deserialize_dope_latent_wire_bytes(zlib.decompress(z_wire))

            # Verify bit-exact roundtrip of serialized wire bytes
            self.assertEqual(serialize_dope_latent_wire_bytes(dec_wire), raw_wire)

            X_mem, y_mem = sender_model.generate_synthetic(enc_mem, n_rows=n_tr, seed=20261023)
            X_dec, y_dec = receiver_model.generate_synthetic(dec_wire, n_rows=n_tr, seed=20261023)

            # X_syn depends only on fp16/bool tensors already quantized in encode_table -> bit-exact match
            self.assertEqual(float(np.max(np.abs(X_mem - X_dec))), 0.0)
            # y_syn differs only by fp16 quantization of 11 target_bundle scalars & tree thresholds
            corr_y = float(np.corrcoef(y_mem, y_dec)[0, 1])
            self.assertGreater(corr_y, 0.999)


if __name__ == "__main__":
    unittest.main()
