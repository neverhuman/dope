import unittest
import numpy as np
import torch

from research.levels.budget_planner import (
    TIER_BYTE_CAPS,
    TIER_COORD_WIDTHS,
    deserialize_dope_v3_envelope,
    serialize_dope_v3_envelope,
)
from research.levels.building_block_landmarks import (
    build_v3_conditioning_bundle,
    compute_building_block_landmarks,
    compute_row_summary_features,
)
from research.levels.multilevel_model import MultiLevelLandmarkModel
from research.levels.scaling import FitOnlyTableTransform
from research.levels.tests.test_multilevel_model import _mock_report


class DopeMlcV3Tests(unittest.TestCase):
    def test_building_block_landmarks_and_inter_landmark_summaries_hierarchy(self):
        rng = np.random.default_rng(2026)
        n, p = 180, 10
        X = rng.normal(size=(n, p))
        # Inject heavy-tailed asinh signal on col 0, step signal on col 1, periodic on col 2, row-mean signal
        y = (
            1.4 * np.arcsinh(np.exp(X[:, 0]))
            + 1.1 * (X[:, 1] > 0.2).astype(np.float64)
            + 0.8 * np.sin(1.5 * X[:, 2])
            + 0.9 * np.mean(X, axis=1)
            + rng.normal(0.0, 0.1, size=n)
        )
        report = _mock_report(n_rows=n, n_features=p, task="regression")

        b_small = build_v3_conditioning_bundle(report, X, y, level="small")
        b_med = build_v3_conditioning_bundle(report, X, y, level="medium")
        b_high = build_v3_conditioning_bundle(report, X, y, level="high")
        b_ultra = build_v3_conditioning_bundle(report, X, y, level="ultra")

        for b in (b_small, b_med, b_high, b_ultra):
            self.assertEqual(b["vec140"].shape, (140,))
            self.assertEqual(b["vec256"].shape, (256,))
            self.assertEqual(b["inter_landmark_vec48"].shape, (48,))
            self.assertEqual(b["building_block_vec68"].shape, (68,))
            self.assertEqual(b["col_landmarks24"].shape, (p, 24))
            self.assertEqual(b["v3_driver_weights"].shape, (p,))
            self.assertTrue(np.all(np.isfinite(b["vec256"])))
            self.assertTrue(np.all(np.isfinite(b["col_landmarks24"])))
            self.assertAlmostEqual(float(np.sum(b["v3_driver_weights"])), 1.0, places=5)

        # Verify level visibility masks on building-block probes (indices 9..15 zeroed at small/medium)
        self.assertTrue(np.allclose(b_small["bb_desc"]["bb_gains16"][6:], 0.0))
        self.assertTrue(np.allclose(b_med["bb_desc"]["bb_gains16"][9:], 0.0))
        self.assertTrue(np.allclose(b_high["bb_desc"]["bb_gains16"][14:], 0.0))
        self.assertGreater(float(np.linalg.norm(b_ultra["bb_desc"]["bb_gains16"][9:14])), 0.0)

        # Verify row summary features only depend on X
        r_feats = compute_row_summary_features(X)
        self.assertEqual(r_feats.shape, (n, 8))
        self.assertTrue(np.all(np.isfinite(r_feats)))

    def test_col_gate_head_active_in_loss_encode_and_generate(self):
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        n, p = 120, 8
        X = rng.normal(size=(n, p)).astype(np.float32)
        y = (1.5 * np.arcsinh(X[:, 0]) - 0.9 * (X[:, 1] > 0.0).astype(np.float32) + 0.6 * X[:, 2]).astype(np.float32)
        report = _mock_report(n_rows=n, n_features=p, task="regression")

        model = MultiLevelLandmarkModel()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        X_tr, y_tr = torch.tensor(X[:80]), torch.tensor(y[:80])
        X_qr, y_qr = torch.tensor(X[80:]), torch.tensor(y[80:])

        vec256, rep_d, tab_d = model.build_conditioning_vector(report, X[:80], y[:80], level="ultra", device=torch.device("cpu"))
        col_lm = torch.tensor(tab_d["col_landmarks24"])
        emp_d = torch.tensor(tab_d["v3_driver_weights"])

        opt.zero_grad()
        out = model.forward_conditioning(X_tr, y_tr, X_qr, y_qr, vec256, emp_d, "regression", "ultra", col_landmarks=col_lm)
        out["total"].backward()

        # Verify non-zero gradients flow into col_gate_head AND InterLandmarkAttentionBlock
        self.assertIsNotNone(model.target_net.col_gate_head.weight.grad)
        self.assertGreater(float(model.target_net.col_gate_head.weight.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(model.router.inter_attn.out_proj.weight.grad)
        self.assertGreater(float(model.router.inter_attn.out_proj.weight.grad.abs().sum().item()), 0.0)
        opt.step()

        # After 1 step, col_gate_head has non-zero weights and actively modulates encode & generate
        gates_encode = model.target_net.compute_column_gates(torch.zeros(128), col_lm, X_pad=torch.zeros((80, 64)))[:p]
        self.assertFalse(torch.allclose(gates_encode, torch.ones_like(gates_encode)))

    def test_four_levels_and_multiclass_exact_wire_caps(self):
        rng = np.random.default_rng(99)
        n_tr, p = 200, 48  # Wide table (p=48) to stress Small (2 KiB) and Medium (4 KiB) caps
        X_raw = rng.normal(size=(n_tr, p))
        # 6-class multiclass target encoded on [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        logits = np.column_stack([
            1.6 * X_raw[:, 0] - 0.8 * X_raw[:, 1],
            -1.2 * X_raw[:, 0] + 1.4 * X_raw[:, 2],
            1.3 * np.tanh(X_raw[:, 3]) + 0.9 * X_raw[:, 4],
            -1.1 * X_raw[:, 2] - 1.0 * X_raw[:, 5],
            0.9 * (X_raw[:, 0] * X_raw[:, 1]),
            -0.7 * X_raw[:, 3] + 1.2 * X_raw[:, 6],
        ])
        y_raw = np.argmax(logits, axis=1).astype(np.float64) / 5.0

        scaler = FitOnlyTableTransform.fit(X_raw, y_raw, task="multiclass")
        X_tr, y_tr = scaler.transform(X_raw, y_raw)
        self.assertEqual(len(scaler.target_scaler.class_labels), 6)

        sender = MultiLevelLandmarkModel().eval()
        receiver = MultiLevelLandmarkModel().eval()
        receiver.load_state_dict(sender.state_dict())
        report = _mock_report(n_rows=n_tr, n_features=p, task="classification")

        for level in ("small", "medium", "high", "ultra"):
            enc = sender.encode_table(
                X_tr, y_tr, report, task="multiclass", level=level,
                adaptive_budget=True, include_v3_primitives=True,
            )
            self.assertEqual(enc["code_width"], TIER_COORD_WIDTHS[level])
            wire = serialize_dope_v3_envelope(enc, scaler=scaler, compress=True)
            self.assertLessEqual(
                len(wire), TIER_BYTE_CAPS[level],
                msg=f"Level {level} wire size {len(wire)} exceeded cap {TIER_BYTE_CAPS[level]}",
            )
            dec_enc, dec_scaler = deserialize_dope_v3_envelope(wire)
            self.assertIsNotNone(dec_scaler)
            X_syn_s, y_syn_idx = receiver.generate_synthetic(dec_enc, n_rows=n_tr, seed=20261024)
            y_syn_orig = dec_scaler.target_scaler.inverse_transform(y_syn_idx)
            self.assertEqual(X_syn_s.shape, (n_tr, p))
            # All 6 classes must be preserved in synthetic data
            self.assertEqual(len(np.unique(np.round(y_syn_orig, 4))), 6)

    def test_ledger_replay_retention_flags_and_receiver_raw_inverse(self):
        from research.levels.v3_ledger_replay import (
            _compute_retention_stats,
            evaluate_multiclass_auditors,
            score_multiclass_syn_against_real,
        )

        # Verify denominator floor and clipping flags on _compute_retention_stats
        st_floor = _compute_retention_stats(0.26, 0.28, chance=0.25, floor=0.03)
        self.assertTrue(st_floor["denom_floor_applied"])
        self.assertAlmostEqual(st_floor["unclipped_R"], 1.0, places=5)
        self.assertFalse(st_floor["clip_high_applied"])

        st_clip = _compute_retention_stats(0.65, 0.85, chance=0.25, floor=0.03)
        self.assertFalse(st_clip["denom_floor_applied"])
        self.assertTrue(st_clip["clip_high_applied"])
        self.assertGreater(st_clip["unclipped_R"], 1.25)
        self.assertAlmostEqual(st_clip["clipped_R"], 1.25, places=5)

        # Verify FitOnlyTableTransform.inverse_transform and dual-domain multiclass auditor scoring
        rng = np.random.default_rng(7)
        X_raw = rng.normal(loc=10.0, scale=3.0, size=(90, 6))
        y_raw = np.array([0.0, 0.5, 1.0] * 30, dtype=np.float64)
        X_raw[:, 0] += 4.0 * y_raw
        scaler = FitOnlyTableTransform.fit(X_raw[:60], y_raw[:60], task="multiclass")
        X_tr_s, y_tr_s = scaler.transform(X_raw[:60], y_raw[:60])
        X_te_s, y_te_s = scaler.transform(X_raw[60:], y_raw[60:])
        X_inv, y_inv = scaler.inverse_transform(X_tr_s, y_tr_s)
        self.assertEqual(X_inv.shape, (60, 6))
        self.assertTrue(np.allclose(y_inv, y_raw[:60]))

        real_info = evaluate_multiclass_auditors(X_tr_s, y_tr_s, X_te_s, y_te_s, k_classes=3)
        syn_res = score_multiclass_syn_against_real(real_info, X_tr_s, y_tr_s, X_te_s, y_te_s, k_classes=3)
        self.assertEqual(len(syn_res["per_auditor"]), 6)
        self.assertIn("task2354_2aud_rf_lr", syn_res)
        self.assertIn("sklearn_6aud_probe", syn_res)
        self.assertEqual(syn_res["class_support_coverage"], 1.0)

    def test_preserved_task2354_source_and_scope_addendum_if_present(self):
        import hashlib
        import json
        from pathlib import Path

        src_path = Path("research/levels/v3_ablation_and_eval.py")
        self.assertTrue(src_path.exists())
        self.assertEqual(
            hashlib.sha256(src_path.read_bytes()).hexdigest(),
            "02d2e0a028d1ba4dae165620979ec42644cc0957a37217c160e19d48bae0bb82",
        )
        addendum_path = Path(
            "target/s3-multilevel-training-20261023/v3/v3_task2354_task2502_scope_and_audit_addendum.json"
        )
        if addendum_path.exists():
            addendum = json.loads(addendum_path.read_text())
            self.assertEqual(addendum["schema"], "dope.v3.scope_and_audit_addendum.v1")
            parents = addendum["immutable_parent_receipts_and_artifacts"]
            for key, entry in parents.items():
                if "path" in entry:
                    p = Path(entry["path"])
                    self.assertTrue(p.exists(), f"Missing {key} at {p}")
                    self.assertEqual(
                        hashlib.sha256(p.read_bytes()).hexdigest(),
                        entry["sha256"],
                        f"Hash mismatch for {key}",
                    )
            scope = addendum["explicit_measurement_scope_disclosures"]
            self.assertFalse(scope["task2502_scope"]["is_recovered_original_task2354_raw_measurements"])
            self.assertEqual(scope["task2502_scope"]["measured_dataset_tier_cells"], 724)
            self.assertEqual(scope["task2502_scope"]["measured_unique_datasets"], 181)
            self.assertTrue(scope["task2502_scope"]["raw_receiver_evaluation_scope"]["covers_multiclass_cells_only"])
            self.assertFalse(
                scope["task2502_scope"]["raw_receiver_evaluation_scope"]["csc1_is_bit_exact_float64_scaler_parity"]
            )
            self.assertFalse(addendum["task2569_fast_audit_finding_disclosure"]["is_passing_scientific_or_release_gate"])

    def test_canonical_6auditor_roster_equal_wire_comparators_and_csc1_categorical(self):
        from research.levels.budget_planner import pack_compact_scaler_sidecar, unpack_compact_scaler_sidecar
        from research.levels.v3_canonical_roster_eval import (
            CANONICAL_6_AUDITORS,
            evaluate_canonical_oracle,
            pack_copula_complete_wire,
            pack_rows_complete_wire,
            score_synthetic_against_oracle,
            unpack_copula_complete_wire,
            unpack_rows_complete_wire,
        )
        from research.levels.v3_canonical_runner import summarize_canonical_ledger_and_elo

        # 1. Verify CSC1 preserves categorical frequency codes (k_id = 5) alongside numeric columns
        rng = np.random.default_rng(123)
        u = rng.normal(size=60)
        X_mixed = [[str(10.0 + float(u[i])), "cat_A" if i % 2 == 0 else "cat_B", str(0.5 * float(u[i]))] for i in range(60)]
        y_reg = [1.5 * float(u[i]) + (2.0 if i % 2 == 0 else -2.0) for i in range(60)]
        sc_mix = FitOnlyTableTransform.fit(X_mixed, y_reg, task="regression")
        X_s, y_s = sc_mix.transform(X_mixed, y_reg)
        sc_bytes = pack_compact_scaler_sidecar(sc_mix, 3, level="medium")
        sc_dec = unpack_compact_scaler_sidecar(sc_bytes)
        X_inv_exact, _ = sc_mix.inverse_transform(X_s, y_s)
        X_inv_csc1, _ = sc_dec.inverse_transform(X_s, y_s)
        # Column 1 is categorical and must be preserved identically by CSC1
        self.assertTrue(np.allclose(X_inv_csc1[:, 1], X_inv_exact[:, 1]))

        # 2. Verify rows_wire and copula_wire complete packets respect tier byte caps and unpack cleanly
        for lvl in ("small", "medium", "high", "ultra"):
            w_r = pack_rows_complete_wire(X_s, y_s, sc_mix, lvl, seed=42)
            self.assertLessEqual(len(w_r), TIER_BYTE_CAPS[lvl])
            Xr, yr, sc_r = unpack_rows_complete_wire(w_r, 60, seed=42)
            self.assertEqual(Xr.shape, (60, 3))
            self.assertEqual(yr.shape, (60,))

            w_c = pack_copula_complete_wire(X_s, y_s, sc_mix, lvl, task="regression")
            self.assertLessEqual(len(w_c), TIER_BYTE_CAPS[lvl])
            Xc, yc, sc_c = unpack_copula_complete_wire(w_c, 60, seed=42)
            self.assertEqual(Xc.shape, (60, 3))
            self.assertEqual(yc.shape, (60,))

        # 3. Verify canonical 6-auditor oracle + synthetic scoring + LocalGatedAttentionAuditor identity
        oracle = evaluate_canonical_oracle(X_s[:40], y_s[:40], X_s[40:], y_s[40:], task="regression")
        self.assertEqual(set(oracle.keys()), set(CANONICAL_6_AUDITORS))
        for aud, info in oracle.items():
            self.assertFalse(info["is_official_kumo_or_tabpfn"])

        syn_eval = score_synthetic_against_oracle(
            oracle, X_s[:40], y_s[:40], X_s[40:], y_s[40:], task="regression", compute_drivers=True
        )
        self.assertEqual(syn_eval["completed_auditors"], 6)
        self.assertEqual(syn_eval["expected_auditors_denom"], 6)
        self.assertEqual(syn_eval["missing_or_error_rate"], 0.0)
        self.assertGreater(syn_eval["complete_case_mean_clipped_R"], 0.8)
        self.assertGreater(syn_eval["mean_driver_fidelity_D"], 0.5)
        self.assertEqual(
            syn_eval["per_auditor"]["local_gated_attention"]["auditor_display_name"],
            "LocalGatedAttentionNet",
        )
        self.assertFalse(syn_eval["per_auditor"]["local_gated_attention"]["is_official_kumo_or_tabpfn"])

        # 4. Verify summarize_canonical_ledger_and_elo accounts for unexecuted cells honestly
        unexec_row = {
            "dataset_id": "ds_unexec",
            "role": "eval_gradient_disjoint_multiclass_panel",
            "task": "multiclass",
            "level": "all",
            "cell_status": "unexecuted_wall_budget_reached",
        }
        red = summarize_canonical_ledger_and_elo([unexec_row], total_planned_mc=69, total_planned_v2_slice=80, n_boot=10)
        self.assertEqual(red["coverage"]["completed_multiclass_datasets"], 0)
        self.assertEqual(red["coverage"]["remaining_unexecuted_multiclass_datasets"], 69)
        self.assertEqual(red["coverage"]["unexecuted_or_interrupted_records"], 1)

    def test_canonical_auditor_label_integrity_and_overwrite_refusal(self) -> None:
        import hashlib
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from research.levels.v3_canonical_roster_eval import (
            CANONICAL_6_AUDITORS,
            CANONICAL_V3_AUDITOR_PROTOCOL_SPEC,
            fit_and_score_canonical_auditor,
            score_synthetic_against_oracle,
        )
        from research.levels.v3_canonical_runner import run_bounded_canonical_eval

        rng = np.random.default_rng(20261024)
        X_tr = rng.normal(size=(36, 4))
        X_te = rng.normal(size=(16, 4))

        # 1. Constant regression, single-class binary, single-class multiclass, and missing-class multiclass:
        #    check that y_train and y_test bytes are never mutated and no synthetic class members are invented.
        cases = [
            ("regression", np.full(36, 3.14159, dtype=np.float64), rng.normal(size=16), 2, "constant_target_train_constant_predictor"),
            ("classification", np.zeros(36, dtype=np.float64), np.array([0, 1] * 8, dtype=np.float64), 2, "single_class_train_constant_predictor"),
            ("multiclass", np.full(36, 2.0, dtype=np.float64), np.array([0.0, 1.0, 2.0, 3.0] * 4, dtype=np.float64), 4, "single_class_train_constant_predictor"),
            ("multiclass", np.array([0.0, 3.0] * 18, dtype=np.float64), np.array([0.0, 1.0, 2.0, 3.0, 4.0] * 3 + [0.0], dtype=np.float64), 5, "partial_class_support_invertible_bijection"),
        ]
        for task, y_tr, y_te, n_cls, expected_status in cases:
            y_tr_sha = hashlib.sha256(np.ascontiguousarray(y_tr).tobytes()).hexdigest()
            y_te_sha = hashlib.sha256(np.ascontiguousarray(y_te).tobytes()).hexdigest()
            for aud in CANONICAL_6_AUDITORS:
                res = fit_and_score_canonical_auditor(
                    aud, X_tr, y_tr, X_te, y_te, task=task, n_classes=n_cls, compute_importance=True
                )
                self.assertEqual(hashlib.sha256(np.ascontiguousarray(y_tr).tobytes()).hexdigest(), y_tr_sha)
                self.assertEqual(hashlib.sha256(np.ascontiguousarray(y_te).tobytes()).hexdigest(), y_te_sha)
                self.assertTrue(res["label_sha256_verified"])
                self.assertEqual(res["train_label_status"], expected_status)
                if expected_status == "partial_class_support_invertible_bijection":
                    self.assertEqual(res["present_train_classes"], 2)
                    self.assertEqual(res["missing_train_classes_count"], 3)

        # 2. Verify complete-case vs zero-imputed sensitivity separation when an auditor fails or is missing
        partial_oracle = {
            "linear": {
                "status": "completed",
                "real_score": 0.6,
                "chance": 0.0,
                "floor": 0.05,
                "importance": np.ones(4) / 4.0,
                "is_official_kumo_or_tabpfn": False,
            }
        }
        sc_res = score_synthetic_against_oracle(
            partial_oracle,
            X_tr,
            X_tr[:, 0] + 0.1 * rng.normal(size=36),
            X_te,
            X_te[:, 0] + 0.1 * rng.normal(size=16),
            task="regression",
            auditors=("linear", "catboost"),
            compute_drivers=False,
        )
        self.assertEqual(sc_res["expected_auditors_denom"], 2)
        self.assertEqual(sc_res["completed_auditors"], 1)
        self.assertEqual(sc_res["failed_auditors"], 1)
        self.assertAlmostEqual(sc_res["missing_or_error_rate"], 0.5)
        self.assertIsNotNone(sc_res["complete_case_mean_clipped_R"])
        self.assertAlmostEqual(
            sc_res["sensitivity_zero_imputed_all_denom_mean_clipped_R"],
            0.5 * float(sc_res["complete_case_mean_clipped_R"]),
            places=3,
        )

        # 3. Verify CANONICAL_V3_AUDITOR_PROTOCOL_SPEC drives constructor parameters and overwrite refusal works
        self.assertEqual(CANONICAL_V3_AUDITOR_PROTOCOL_SPEC["auditors"]["lightgbm"]["multiclass"]["n_estimators"], 24)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            existing_run = tmp_out / "runs" / "duplicate_run_id"
            existing_run.mkdir(parents=True, exist_ok=True)
            with patch("research.levels.v3_canonical_runner.V3_OUT_DIR", tmp_out):
                with self.assertRaises(FileExistsError):
                    run_bounded_canonical_eval(max_v2_slice=1, workers_count=1, wall_budget_s=1.0, run_id="duplicate_run_id")


if __name__ == "__main__":
    unittest.main()
