"""Unit tests for dynamic per-batch resplitting, augmentation, and DopeFidelityV5 Main-KPI training."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.torch import save_file
import torch

from research.levels.s3_dynamic_augment import (
    AugmentConfig,
    S3CorpusStore,
    sample_dynamic_episode,
)
from research.levels.s3_kpi_evaluator import evaluate_checkpoint
from research.levels.v5_fidelity_net import DopeFidelityV5, compute_main_kpi_loss


def _make_mock_table(task: str, n: int = 180, p: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    # Inject exact duplicate rows to verify Rule A-1 grouping
    for i in range(0, 20, 2):
        X[i + 1] = X[i]
    dup = np.arange(n, dtype=np.int32)
    for i in range(0, 20, 2):
        dup[i + 1] = dup[i]
    signal = 1.6 * X[:, 0] - 1.1 * X[:, 1] + 0.7 * (X[:, 0] * X[:, 2])
    if task == "regression":
        y = (signal + 0.25 * rng.standard_normal(n)).astype(np.float32)
        n_classes = 0
    elif task == "binary":
        y = (signal > float(np.median(signal))).astype(np.float32)
        n_classes = 2
    else:
        q = np.quantile(signal, [0.25, 0.50, 0.75])
        y = np.digitize(signal, q).astype(np.float32)
        n_classes = 4
    return {
        "dataset_hash": f"mock_{task}_{seed}",
        "family_stem": f"fam_{task}_{seed}",
        "task": task,
        "n_classes": n_classes,
        "X": X,
        "y": y,
        "dup_groups": dup,
    }


class TestS3V5DynamicTraining(unittest.TestCase):
    def test_fresh_splits_rule_a1_row_lengths_and_synthetic_columns(self) -> None:
        tbl = _make_mock_table("regression", n=220, p=8, seed=11)
        cfg = AugmentConfig(
            min_fit_rows=32,
            max_fit_rows=140,
            min_query_rows=24,
            max_query_rows=70,
            aug_prob=1.0,
            max_noise_cols=4,
            max_derived_cols=3,
            multilabel_view_prob=0.0,
        )
        sigs, fit_lens, noise_counts = set(), set(), []
        for step in range(12):
            ep = sample_dynamic_episode(tbl, global_seed=7, step=step, rank=0, cfg=cfg)
            sigs.add(ep["split_sig"])
            fit_lens.add(ep["aug_stats"]["n_fit"])
            noise_counts.append(ep["aug_stats"]["n_noise_cols"])
            self.assertGreaterEqual(ep["aug_stats"]["n_noise_cols"], 1)
            self.assertEqual(int(ep["distractor_mask"].sum()), ep["aug_stats"]["n_noise_cols"])
            self.assertTrue(np.isfinite(ep["X_fit"]).all())
            self.assertTrue(np.isfinite(ep["X_qry"]).all())
        self.assertGreaterEqual(len(sigs), 10)
        self.assertGreaterEqual(len(fit_lens), 4)
        self.assertGreater(sum(noise_counts), 12)

    def test_all_four_tasks_forward_backward_finite_and_permutation_equivariant(self) -> None:
        torch.manual_seed(42)
        model = DopeFidelityV5(d_model=64, n_layers=2, n_heads=4, latent_dim=32, copula_rank=8)
        dev = torch.device("cpu")
        for idx, (task, ml_prob) in enumerate(
            [("regression", 0.0), ("binary", 0.0), ("multiclass", 0.0), ("regression", 1.0)]
        ):
            tbl = _make_mock_table(task, n=180, p=8, seed=idx + 1)
            cfg = AugmentConfig(
                min_fit_rows=48,
                max_fit_rows=96,
                min_query_rows=32,
                max_query_rows=64,
                aug_prob=1.0,
                multilabel_view_prob=ml_prob,
            )
            ep = sample_dynamic_episode(tbl, global_seed=99, step=idx, rank=0, cfg=cfg)
            expected_task = "multilabel" if ml_prob == 1.0 else task
            self.assertEqual(ep["task"], expected_task)
            model.zero_grad(set_to_none=True)
            out = compute_main_kpi_loss(model, ep, dev, step_seed=100 + idx)
            self.assertTrue(bool(torch.isfinite(out["total"]).all()))
            out["total"].backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertGreater(len(grads), 5)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))

        # Verify column-permutation invariance of table latent and equivariance of driver weights
        model.eval()
        X_t = torch.randn(64, 6)
        y_t = 1.5 * X_t[:, 0] - 0.8 * X_t[:, 2]
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        with torch.no_grad():
            e1 = model.encode_table(X_t, y_t, "regression", 0)
            e2 = model.encode_table(X_t[:, perm], y_t, "regression", 0)
        self.assertLess(float((e1["task_latent"] - e2["task_latent"]).abs().max().item()), 1e-4)
        self.assertLess(float((e1["driver_weights"][perm] - e2["driver_weights"]).abs().max().item()), 1e-4)

    def test_corpus_store_and_checkpoint_kpi_evaluator_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tables").mkdir()
            rows = []
            for task in ("regression", "binary", "multiclass"):
                tbl = _make_mock_table(task, n=160, p=6, seed=5)
                rel = f"tables/{tbl['dataset_hash']}.npz"
                np.savez_compressed(
                    root / rel,
                    X=tbl["X"],
                    y=tbl["y"],
                    dup_groups=tbl["dup_groups"],
                    n_classes=np.int32(tbl["n_classes"]),
                    task=np.array(task),
                )
                rows.append(
                    {
                        "dataset_hash": tbl["dataset_hash"],
                        "canonical_sha256": "0" * 64,
                        "family_stem": tbl["family_stem"],
                        "task": task,
                        "raw_task": task,
                        "n_rows": 160,
                        "n_features": 6,
                        "n_classes": tbl["n_classes"],
                        "split": "val",
                        "npz_relpath": rel,
                    }
                )
            (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            store = S3CorpusStore(root, split="val", cache_cap=16)
            model = DopeFidelityV5(d_model=64, n_layers=2, n_heads=4, latent_dim=32)
            ckpt = root / "test.safetensors"
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(ckpt))
            res = evaluate_checkpoint(
                ckpt,
                {"step": 10, "d_model": 64, "n_layers": 2, "latent_dim": 32},
                store,
                torch.device("cpu"),
                max_per_task=1,
            )
            self.assertEqual(res["formal_dp_state"], "not_claimed")
            self.assertGreaterEqual(res["tables_evaluated"], 2)
            self.assertIsNotNone(res["main_kpi_retention_median"])


if __name__ == "__main__":
    unittest.main()
