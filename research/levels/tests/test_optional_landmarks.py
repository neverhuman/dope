"""Unit tests for Optional & Progressive Landmarks (LandmarkBundle v1 and MLC-v3 modes)."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np

from research.levels import cli, landmark_features
from research.levels.descriptors import extract_report_descriptors  # noqa: F401
from research.levels.multilevel_model import (
    MultiLevelLandmarkModel, build_conditioning, build_conditioning_inputs, build_v3_conditioning_bundle,  # noqa: F401
)
from research.levels.optional_landmarks import (  # noqa: F401
    FINE_INDICES, LANDMARK_NAMES, LARGER_INDICES, FULL_REPORT_RECIPE, LandmarkBundle,
    MACRO_INDICES, OBSERVED_RECIPE, bundle_to_report, effective_conditioning_digest,
    observed_report_descriptors, observed_target_stats, progressive_bundles, progressive_steps,
    squash_f32, stratified_subsample, table_target_stats, unsquash,
)
from research.levels.scaling import FitOnlyTableTransform

GOLDEN_ZERO_DIGEST = "3b80a6af433fd746af93fd3c65fe0cfd85f5b887a1a9acecdb3d2748c6658ed5"
GOLDEN_MACRO_PARTIAL_300_DIGEST = "4eaedd68a5239fdf3542e93cec1d7bffa4a7143e7c23f53be5a3c8716a53806b"
FIXTURE_PATH = Path(__file__).resolve().parents[3] / "engine" / "dope-core" / "tests" / "fixtures" / "landmark_bundle_v1.json"


def _make_full_report(
    *, rows: int = 120, features: int = 6, task: str = "regression", classes: int = 0,
    target_stats: list[float | None] | None = None, uninformative_slots: tuple[int, ...] = (),
    failed_slots: tuple[int, ...] = (),
) -> dict:
    cells, tr_rows = [], [max(1, rows - max(1, rows // 5))] * 5
    for idx, name in enumerate(LANDMARK_NAMES):
        cid = idx + 1
        qr_rows = landmark_features.expected_query_rows(cid, rows, landmark_features.OPTIONS)
        if idx in failed_slots:
            cells.append({"id": cid, "name": name, "state": "failed", "gain": None, "se": None, "stable": False, "lambda_indices": [], "train_rows": [], "query_rows": 0})
        elif idx in uninformative_slots:
            cells.append({"id": cid, "name": name, "state": "uninformative", "gain": 0.0, "se": 0.01, "stable": False, "lambda_indices": [1], "train_rows": list(tr_rows), "query_rows": qr_rows})
        else:
            cells.append({"id": cid, "name": name, "state": "complete", "gain": 0.04 * (idx + 1), "se": 0.005 * ((idx % 4) + 1), "stable": (idx % 2 == 0), "lambda_indices": [idx % 5, (idx + 2) % 5], "train_rows": list(tr_rows), "query_rows": qr_rows})
    return {
        "panel": landmark_features.OPTIONS["panel"], "kind": "real", "privacy": "private",
        "options": dict(landmark_features.OPTIONS), "rows": rows, "features": features,
        "task": task, "classes": classes, "cells": cells,
        "spectrum": [{"slot": j, "values": [0.05 * (k + 1) + 0.02 * j for k in range(9)]} for j in range(features)],
        "target_stats": [0.42, 1.25, None] if target_stats is None else list(target_stats),
    }


class TestOptionalLandmarks(unittest.TestCase):
    def test_golden_digests_and_fixture_parity(self) -> None:
        z = LandmarkBundle.zero()
        self.assertTrue(z.is_zero())
        self.assertEqual(z.status, "none")
        self.assertEqual(z.conditioning_digest(), GOLDEN_ZERO_DIGEST)
        np.testing.assert_array_equal(z.features46(), np.zeros(46, dtype=np.float32))

        raw = [float("nan")] * 23
        for i in MACRO_INDICES:
            raw[i] = 0.05 * (i + 1)
        p = LandmarkBundle.from_raw_gains(
            raw,
            subsample_n=300,
            source_digest="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            provenance="native-v1",
        )
        self.assertFalse(p.is_zero())
        self.assertEqual(p.status, {"partial": 9})
        self.assertEqual(p.conditioning_digest(), GOLDEN_MACRO_PARTIAL_300_DIGEST)

        self.assertTrue(FIXTURE_PATH.is_file(), f"Missing golden fixture at {FIXTURE_PATH}")
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(fixture["schema"], "dope.landmark-bundle-fixtures/v1")
        cases = {c["name"]: c for c in fixture["cases"]}

        zero_case = cases["zero"]
        self.assertEqual(zero_case["conditioning_digest"], GOLDEN_ZERO_DIGEST)
        z_rt = LandmarkBundle.from_dict(zero_case["bundle"])
        self.assertEqual(z_rt, z)
        self.assertEqual(z_rt.to_dict(), zero_case["bundle"])
        self.assertEqual(LandmarkBundle.from_json(z_rt.to_json()).to_dict(), zero_case["bundle"])
        self.assertEqual(z_rt.conditioning_digest(), GOLDEN_ZERO_DIGEST)

        macro_case = cases["macro"]
        self.assertEqual(macro_case["conditioning_digest"], GOLDEN_MACRO_PARTIAL_300_DIGEST)
        p_rt = LandmarkBundle.from_dict(macro_case["bundle"])
        self.assertEqual(p_rt, p)
        self.assertEqual(p_rt.to_dict(), macro_case["bundle"])
        self.assertEqual(LandmarkBundle.from_json(p_rt.to_json()).to_dict(), macro_case["bundle"])
        self.assertEqual(p_rt.conditioning_digest(), GOLDEN_MACRO_PARTIAL_300_DIGEST)

    def test_sanitation_raw_gains_overflow_and_signed_zero(self) -> None:
        raw: list[float | None | bool] = [None] * 23
        raw[0] = -0.0
        raw[1] = 0.0
        raw[2] = float("nan")
        raw[3] = float("inf")
        raw[4] = float("-inf")
        raw[5] = True
        raw[6] = 1e300
        raw[7] = -1e300
        raw[8] = 0.25

        bundle = LandmarkBundle.from_raw_gains(raw, subsample_n=200, source_digest="scope1")
        self.assertTrue(bundle.mask[0])
        self.assertEqual(bundle.values[0], 0.0)
        self.assertEqual(math.copysign(1.0, bundle.values[0]), 1.0)
        self.assertTrue(bundle.mask[1])
        self.assertEqual(bundle.values[1], 0.0)
        self.assertEqual(math.copysign(1.0, bundle.values[1]), 1.0)
        for absent_slot in (2, 3, 4, 5):
            self.assertFalse(bundle.mask[absent_slot])
            self.assertEqual(bundle.values[absent_slot], 0.0)
            self.assertEqual(math.copysign(1.0, bundle.values[absent_slot]), 1.0)
        # Finite f64 values exceeding f32 max must NOT overflow to +-Inf and drop mask:
        self.assertTrue(bundle.mask[6])
        self.assertEqual(bundle.values[6], 1.0)
        self.assertTrue(bundle.mask[7])
        self.assertEqual(bundle.values[7], -1.0)
        self.assertTrue(bundle.mask[8])
        self.assertAlmostEqual(bundle.values[8], float(np.float32(0.25) / np.float32(1.25)), places=7)
        self.assertEqual(bundle.status, {"partial": 5})

        # Also check report with 1e300 gain
        rep = _make_full_report()
        rep["cells"][0]["gain"] = 1e300
        rep_bundle = LandmarkBundle.from_report(rep)
        self.assertTrue(rep_bundle.mask[0])
        self.assertEqual(rep_bundle.values[0], 1.0)
        self.assertEqual(rep_bundle.status, "full")

        opt_f46 = landmark_features.feature_tensor_optional(
            landmark_features.report_bundle(rep, rows=120, task="regression"),
            device="cpu",
        )
        self.assertEqual(float(opt_f46[0]), 1.0)
        self.assertEqual(float(opt_f46[23]), 1.0)

    def test_sanitation_on_squashed_json_and_merge(self) -> None:
        vals = [0.0] * 23
        mask = [False] * 23
        vals[0], mask[0] = -0.0, True
        vals[1], mask[1] = 1.5, True       # out of [-1, 1] -> sanitized to absent
        vals[2], mask[2] = float("nan"), True
        vals[3], mask[3] = float("-inf"), True
        vals[4], mask[4] = 0.75, False     # masked-out non-zero -> zeroed
        vals[5], mask[5] = 0.5, True

        b = LandmarkBundle.from_squashed(vals, mask, subsample_n=250, source_digest="s")
        self.assertEqual(b.mask[0], True)
        self.assertEqual(math.copysign(1.0, b.values[0]), 1.0)
        for bad in (1, 2, 3, 4):
            self.assertFalse(b.mask[bad])
            self.assertEqual(b.values[bad], 0.0)
        self.assertTrue(b.mask[5])
        self.assertEqual(b.values[5], 0.5)
        self.assertEqual(b.status, {"partial": 2})

        # Deserialization with stale/forged status and unsanitized payload
        dirty_payload = {
            "values": [0.5, float("nan"), 2.0, -0.0] + [0.25] * 19,
            "mask": [True, True, True, True] + [False] * 19,
            "status": "full",  # stale/wrong! Only 2 valid slots remain after sanitation.
            "subsample_n": 250,
            "source_digest": "s",
            "provenance": "test",
        }
        b_from_dict = LandmarkBundle.from_dict(dirty_payload)
        self.assertEqual(b_from_dict.status, {"partial": 2})
        self.assertEqual(b_from_dict.mask[:4], (True, False, False, True))
        self.assertEqual(b_from_dict.values[1], 0.0)
        self.assertEqual(b_from_dict.values[2], 0.0)
        self.assertTrue( all(v == 0.0 for v in b_from_dict.values[4:]) )

        b_from_json = LandmarkBundle.from_json(json.dumps(dirty_payload))
        self.assertEqual(b_from_json, b_from_dict)

        # Merge must sanitize incoming so an unvalidated NaN with mask=True never clobbers base
        base = LandmarkBundle.from_squashed(
            [0.25 if i == 0 else 0.0 for i in range(23)],
            [i == 0 for i in range(23)],
            subsample_n=250,
            source_digest="s",
        )
        corrupted = LandmarkBundle(
            values=(float("nan"), 0.5) + (0.0,) * 21,
            mask=(True, True) + (False,) * 21,
            subsample_n=250,
            source_digest="s",
        )
        merged = base.merge(corrupted)
        self.assertTrue(merged.mask[0])
        self.assertEqual(merged.values[0], 0.25)
        self.assertTrue(merged.mask[1])
        self.assertEqual(merged.values[1], 0.5)
        self.assertEqual(merged.status, {"partial": 2})

    def test_zero_sentinel_vs_scoped_empty_and_merge_laws(self) -> None:
        z = LandmarkBundle.zero()
        self.assertTrue(z.is_zero())

        scoped_empty = LandmarkBundle.from_raw_gains(
            [None] * 23, subsample_n=300, source_digest="table_a"
        )
        self.assertEqual(scoped_empty.status, "none")
        self.assertEqual(scoped_empty.subsample_n, 300)
        self.assertEqual(scoped_empty.source_digest, "table_a")
        self.assertFalse(scoped_empty.is_zero())

        # Scoped empty round-trips through JSON preserving scope
        scoped_empty_rt = LandmarkBundle.from_json(scoped_empty.to_json())
        self.assertEqual(scoped_empty_rt, scoped_empty)
        self.assertFalse(scoped_empty_rt.is_zero())

        # Scoped bundle for table_a
        macro_a = LandmarkBundle.from_raw_gains(
            [0.1 * (i + 1) if i in MACRO_INDICES else None for i in range(23)],
            subsample_n=300,
            source_digest="table_a",
        )
        larger_a = LandmarkBundle.from_raw_gains(
            [0.1 * (i + 1) if i in LARGER_INDICES else None for i in range(23)],
            subsample_n=300,
            source_digest="table_a",
        )
        fine_a = LandmarkBundle.from_raw_gains(
            [0.1 * (i + 1) if i in FINE_INDICES else None for i in range(23)],
            subsample_n=300,
            source_digest="table_a",
        )

        # Zero merges universally with any scope
        self.assertEqual(z.merge(macro_a), macro_a)
        self.assertEqual(macro_a.merge(z), macro_a)
        self.assertEqual(z.merge(scoped_empty), scoped_empty)
        self.assertEqual(scoped_empty.merge(z), scoped_empty)

        # Scoped empty merges with same scope, rejects different scope
        self.assertEqual(scoped_empty.merge(macro_a), macro_a)
        self.assertEqual(macro_a.merge(scoped_empty), macro_a)

        scoped_empty_b = LandmarkBundle.from_raw_gains(
            [None] * 23, subsample_n=300, source_digest="table_b"
        )
        scoped_empty_500 = LandmarkBundle.from_raw_gains(
            [None] * 23, subsample_n=500, source_digest="table_a"
        )
        with self.assertRaises(ValueError):
            scoped_empty.merge(macro_a.from_raw_gains(
                [0.1 if i == 0 else None for i in range(23)],
                subsample_n=300,
                source_digest="table_b",
            ))
        with self.assertRaises(ValueError):
            scoped_empty.merge(scoped_empty_b)
        with self.assertRaises(ValueError):
            scoped_empty.merge(scoped_empty_500)

        # Invalid subsample_n rejected
        with self.assertRaises(ValueError):
            LandmarkBundle.from_raw_gains([None] * 23, subsample_n=0)
        with self.assertRaises(ValueError):
            LandmarkBundle.from_raw_gains([None] * 23, subsample_n=-10)
        with self.assertRaises(ValueError):
            LandmarkBundle.from_raw_gains([None] * 23, subsample_n=True)

        # Idempotence and order independence across disjoint progressive groups
        self.assertEqual(macro_a.merge(macro_a), macro_a)
        order_1 = macro_a.merge(larger_a).merge(fine_a)
        order_2 = fine_a.merge(macro_a).merge(larger_a)
        order_3 = larger_a.merge(fine_a).merge(macro_a)
        self.assertEqual(order_1.status, "full")
        self.assertEqual(order_1, order_2)
        self.assertEqual(order_1, order_3)
        self.assertEqual(order_1.conditioning_digest(), order_2.conditioning_digest())

    def test_from_report_measured_zero_uninformative_vs_complete_only(self) -> None:
        rep = _make_full_report(uninformative_slots=(2, 7), failed_slots=(11,))

        # Default: include_uninformative=True keeps measured 0.0 (uninformative) present
        b_default = LandmarkBundle.from_report(rep, subsample_n=300)
        self.assertTrue(b_default.mask[2])
        self.assertEqual(b_default.values[2], 0.0)
        self.assertTrue(b_default.mask[7])
        self.assertEqual(b_default.values[7], 0.0)
        self.assertFalse(b_default.mask[11])
        self.assertEqual(b_default.status, {"partial": 22})

        # Strict legacy FEATURE_RECIPE parity: complete_only=True / from_report_complete
        b_complete = LandmarkBundle.from_report_complete(rep, subsample_n=300)
        self.assertFalse(b_complete.mask[2])
        self.assertFalse(b_complete.mask[7])
        self.assertFalse(b_complete.mask[11])
        self.assertEqual(b_complete.status, {"partial": 20})
        self.assertEqual(
            b_complete,
            LandmarkBundle.from_report(rep, subsample_n=300, complete_only=True),
        )

    def test_full_bundle_bit_identity_with_strict_landmark_features_and_legacy_v3(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(64, 6))
        y = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.2 * (X[:, 2] ** 2) + rng.normal(scale=0.1, size=64)
        rep = _make_full_report(rows=64, features=6, task="regression", classes=0)

        # 1. features46 bit-identity with landmark_features.feature_tensor
        strict_t = landmark_features.feature_tensor(
            landmark_features.report_features(rep, rows=64, task="regression"),
            device="cpu",
        )
        bundle_full = LandmarkBundle.from_report(rep)
        np.testing.assert_array_equal(bundle_full.features46(), strict_t.numpy())
        np.testing.assert_array_equal(bundle_full.features46(device="cpu").numpy(), strict_t.numpy())

        # 2. Conditioning bit-identity on unsampled full mode vs build_v3_conditioning_bundle
        legacy_cond = build_v3_conditioning_bundle(rep, X, y, level="high")
        new_cond = build_conditioning(
            X, y, task="regression", bundle=None, mode="full", native_report=rep, level="high"
        )
        self.assertEqual(new_cond["recipe"], FULL_REPORT_RECIPE)
        np.testing.assert_array_equal(new_cond["vec256"], legacy_cond["vec256"])
        np.testing.assert_array_equal(new_cond["cond"]["col_landmarks24"], legacy_cond["col_landmarks24"])
        np.testing.assert_array_equal(new_cond["col_landmarks"][:6], legacy_cond["col_landmarks24"][:6, :17])

        # 3. encode_table bit-identity on unsampled full mode
        import torch
        torch.manual_seed(0)
        model = MultiLevelLandmarkModel()
        model.eval()
        legacy_enc = model.encode_table(X, y, rep, task="regression", level="high")
        new_enc = model.encode_table(
            X, y, rep, task="regression", level="high", landmarks=bundle_full, landmark_mode="full"
        )
        np.testing.assert_array_equal(new_enc["code_fp16"], legacy_enc["code_fp16"])
        np.testing.assert_array_equal(new_enc["task_latent"], legacy_enc["task_latent"])
        np.testing.assert_array_equal(new_enc["dist_latent"], legacy_enc["dist_latent"])

    def test_sampled_full_mode_overrides_base_shape_and_target_stats(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 5))
        y_raw = np.exp(0.8 * X[:, 0] + rng.normal(scale=0.3, size=200))
        tf = FitOnlyTableTransform.fit(X, y_raw, task="regression")
        X_scaled, y_scaled = tf.transform(X, y_raw)
        rep_n80 = _make_full_report(rows=80, features=5, task="regression", classes=0, target_stats=[0.91, 0.55, None])
        rep_n140 = _make_full_report(rows=140, features=5, task="regression", classes=0, target_stats=[1.44, 2.80, None])

        cond_none = build_conditioning(X_scaled, y_scaled, task="regression", mode="none", y_train_raw=y_raw)
        cond_prog = build_conditioning(X_scaled, y_scaled, task="regression", mode="progressive", bundle=LandmarkBundle.from_report(rep_n80, subsample_n=80), y_train_raw=y_raw)
        cond_full_80 = build_conditioning(X_scaled, y_scaled, task="regression", mode="full", native_report=rep_n80, subsample_n=80, y_train_raw=y_raw)
        cond_full_140 = build_conditioning(X_scaled, y_scaled, task="regression", mode="full", native_report=rep_n140, subsample_n=140, y_train_raw=y_raw)

        np.testing.assert_allclose(cond_full_80["vec256"][85:91], cond_none["vec256"][85:91], rtol=0, atol=1e-6)
        np.testing.assert_allclose(cond_full_80["vec256"][85:91], cond_prog["vec256"][85:91], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(cond_full_80["vec256"][85:91], cond_full_140["vec256"][85:91])
        self.assertTrue(np.any(cond_full_80["vec256"][:23] != 0.0))
        self.assertTrue(np.any(cond_full_80["col_landmarks"][:5, :9] != 0.0))

        rep_partial = _make_full_report(rows=80, features=5, task="regression", classes=0, uninformative_slots=(1, 3), failed_slots=(18, 22))
        b_inc = LandmarkBundle.from_report(rep_partial, subsample_n=80)
        b_comp = LandmarkBundle.from_report_complete(rep_partial, subsample_n=80)
        self.assertEqual(b_inc.status, {"partial": 21})
        self.assertEqual(b_comp.status, {"partial": 19})
        out_inc = build_conditioning(X_scaled, y_scaled, task="regression", mode="full", bundle=b_inc, native_report=rep_partial, subsample_n=80, y_train_raw=y_raw)
        out_comp = build_conditioning(X_scaled, y_scaled, task="regression", mode="full", bundle=b_comp, native_report=rep_partial, subsample_n=80, y_train_raw=y_raw)
        np.testing.assert_array_equal(out_inc["vec256"], out_comp["vec256"])

    def test_none_fast_and_progressive_modes(self) -> None:
        rng = np.random.default_rng(123)
        X = rng.normal(size=(90, 4))
        y_cls = np.array([0] * 50 + [1] * 30 + [2] * 10, dtype=np.float64)

        cond_none = build_conditioning(X, y_cls, task="multiclass", mode="none")
        cond_fast = build_conditioning(X, y_cls, task="multiclass", mode="fast")
        self.assertEqual(cond_none["recipe"], OBSERVED_RECIPE)
        np.testing.assert_array_equal(cond_none["vec256"], cond_fast["vec256"])
        np.testing.assert_array_equal(cond_none["col_landmarks"], cond_fast["col_landmarks"])
        self.assertEqual(cond_none["digest"], cond_fast["digest"])
        np.testing.assert_array_equal(cond_none["vec256"][0:85], np.zeros(85, dtype=np.float32))
        np.testing.assert_array_equal(cond_none["vec256"][91:116], np.zeros(25, dtype=np.float32))
        self.assertGreater(float(cond_none["vec256"][85]), 0.0)
        self.assertGreater(float(cond_none["vec256"][86]), 0.0)
        self.assertTrue(np.any(cond_none["vec256"][87:91] != 0.0))
        self.assertTrue(np.any(cond_none["vec256"][116:140] != 0.0))
        np.testing.assert_array_equal(cond_none["col_landmarks"][:, :9], np.zeros((cond_none["col_landmarks"].shape[0], 9), dtype=np.float32))
        self.assertTrue(np.any(cond_none["col_landmarks"][:4, 9:17] != 0.0))

        y_reg_raw = np.exp(X[:, 0])
        y_reg_scaled = 8.0 * np.tanh((y_reg_raw - np.mean(y_reg_raw)) / np.std(y_reg_raw) / 8.0)
        reg_with_raw = build_conditioning(X, y_reg_scaled, task="regression", mode="none", y_train_raw=y_reg_raw)
        reg_scaled_only = build_conditioning(X, y_reg_scaled, task="regression", mode="none")
        self.assertTrue(np.any(reg_with_raw["vec256"][85:87] != 0.0))
        np.testing.assert_array_equal(reg_scaled_only["vec256"][85:87], np.zeros(2, dtype=np.float32))

        macro_b = LandmarkBundle.from_raw_gains([0.2 if i in MACRO_INDICES else None for i in range(23)], subsample_n=90)
        with self.assertRaises(ValueError):
            build_conditioning(X, y_cls, task="multiclass", mode="none", bundle=macro_b)
        with self.assertRaises(ValueError):
            build_conditioning(X, y_cls, task="multiclass", mode="fast", bundle=macro_b)

        full_rep = _make_full_report(rows=90, features=4, task="multiclass", classes=3)

        def fake_runner(slots: tuple[int, ...]) -> dict:
            allowed = {s + 1 for s in slots}
            return dict(full_rep, cells=[c for c in full_rep["cells"] if int(c["id"]) in allowed])

        steps = list(progressive_steps(fake_runner, subsample_n=90, source_digest="prog_scope"))
        self.assertEqual([name for _, name, _, _ in steps], ["zero", "macro", "larger", "fine"])
        self.assertEqual(steps[0][2].status, "none")
        self.assertTrue(steps[0][2].is_zero())
        self.assertEqual(steps[1][2].status, {"partial": 9})
        self.assertEqual(steps[2][2].status, {"partial": 18})
        self.assertEqual(steps[3][2].status, "full")

        def empty_first_runner(slots: tuple[int, ...]) -> dict:
            return dict(full_rep, cells=[]) if set(slots) == set(MACRO_INDICES) else fake_runner(slots)

        steps_empty_first = list(progressive_steps(empty_first_runner, subsample_n=90, source_digest="prog_scope"))
        self.assertEqual(steps_empty_first[1][2].status, "none")
        self.assertFalse(steps_empty_first[1][2].is_zero())
        self.assertEqual(steps_empty_first[1][2].subsample_n, 90)
        self.assertEqual(steps_empty_first[1][2].source_digest, "prog_scope")
        self.assertEqual(steps_empty_first[2][2].status, {"partial": 9})
        self.assertEqual(steps_empty_first[3][2].status, {"partial": 14})

    def test_stratified_subsample_determinism_and_row_id_permutation_invariance(self) -> None:
        rng = np.random.default_rng(99)
        y_cls = np.array([0] * 500 + [1] * 250 + [2] * 50)
        idx1 = stratified_subsample(y_cls, "multiclass", 160, seed=42)
        idx2 = stratified_subsample(y_cls, "multiclass", 160, seed=42)
        np.testing.assert_array_equal(idx1, idx2)
        self.assertEqual(idx1.size, 160)
        for c in (0, 1, 2):
            self.assertGreaterEqual(int(np.sum(y_cls[idx1] == c)), 2)

        perm = rng.permutation(y_cls.size)
        row_ids = np.arange(y_cls.size, dtype=np.int64)
        sel_orig = stratified_subsample(y_cls, "multiclass", 160, seed=42, row_ids=row_ids)
        sel_perm = stratified_subsample(y_cls[perm], "multiclass", 160, seed=42, row_ids=row_ids[perm])
        np.testing.assert_array_equal(np.sort(row_ids[sel_orig]), np.sort(row_ids[perm][sel_perm]))

        y_reg = np.linspace(-10.0, 100.0, 600) + rng.normal(scale=0.01, size=600)
        perm_r = rng.permutation(y_reg.size)
        rids_r = np.arange(y_reg.size, dtype=np.int64)
        sel_r_orig = stratified_subsample(y_reg, "regression", 150, seed=17, row_ids=rids_r)
        sel_r_perm = stratified_subsample(y_reg[perm_r], "regression", 150, seed=17, row_ids=rids_r[perm_r])
        np.testing.assert_array_equal(np.sort(rids_r[sel_r_orig]), np.sort(rids_r[perm_r][sel_r_perm]))

    def test_cli_encode_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path, rep_path, weights_path = tmp / "table.csv", tmp / "report.json", tmp / "dummy.safetensors"
            rng = np.random.default_rng(11)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["f0", "f1", "f2", "target"])
                for _ in range(40):
                    x0, x1, x2 = float(rng.normal()), float(rng.normal()), float(rng.normal())
                    w.writerow([x0, x1, x2, float(np.exp(0.5 * x0 - 0.3 * x1))])
            rep_path.write_text(json.dumps(_make_full_report(rows=25, features=3, task="regression", classes=0)), encoding="utf-8")
            weights_path.write_bytes(b"dummy")
            from unittest import mock
            import torch

            class _DummyEncoder(torch.nn.Module):
                def forward(self, vec256: torch.Tensor, col_lm: torch.Tensor) -> torch.Tensor:
                    out = torch.zeros((vec256.shape[0], 629), dtype=torch.float32, device=vec256.device)
                    out[:, :256], out[:, 256:273] = vec256, col_lm[:, 0, :]
                    return out

            with mock.patch("research.levels.encoder.DopeEncoder629.from_pretrained", return_value=_DummyEncoder()):
                buf_none, buf_full = io.StringIO(), io.StringIO()
                with mock.patch("sys.stdout", buf_none):
                    self.assertEqual(cli.main(["encode", "--csv", str(csv_path), "--target", "target", "--task", "regression", "--weights", str(weights_path), "--landmark-mode", "none", "--json"]), 0)
                rec_none = json.loads(buf_none.getvalue().strip())
                self.assertEqual((rec_none["status"], rec_none["conditioning_digest"], rec_none["recipe"]), ("none", GOLDEN_ZERO_DIGEST, OBSERVED_RECIPE))
                with mock.patch("sys.stdout", buf_full):
                    self.assertEqual(cli.main(["encode", "--csv", str(csv_path), "--target", "target", "--task", "regression", "--weights", str(weights_path), "--landmark-mode", "full", "--report-json", str(rep_path), "--landmark-subsample", "25", "--json"]), 0)
                rec_full = json.loads(buf_full.getvalue().strip())
                self.assertEqual((rec_full["status"], rec_full["recipe"]), ("full", FULL_REPORT_RECIPE))
                self.assertNotEqual(rec_full["features629_sha256"], rec_none["features629_sha256"])


if __name__ == "__main__":
    unittest.main()
