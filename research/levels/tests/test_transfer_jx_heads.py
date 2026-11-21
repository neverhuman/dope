"""Unit tests for DOPE-JX nested prefix heads, preference reversals, and invariance losses."""
import math
import unittest

import torch

from research.levels.transfer_jx_heads import (
    LATENT_WIDTH,
    PREFIXES,
    SCHEMA,
    JXHeadError,
    JXPrefixHeads,
    matryoshka_prefix_loss,
    preference_reversal_loss,
    split_invariance_infonce,
    target_swap_disentangle_loss,
    task_additive_offset_control,
)


class TransferJXHeadsTests(unittest.TestCase):
    def test_prefix_heads_isolate_task_prefix_and_dist_latent(self):
        torch.manual_seed(101)
        heads = JXPrefixHeads(program_slots=3, pair_noise_dim=2)
        task = torch.randn((4, LATENT_WIDTH), dtype=torch.float32, requires_grad=True)
        dist = torch.randn((4, LATENT_WIDTH), dtype=torch.float32, requires_grad=True)

        out32 = heads(task, dist, prefix=32)
        self.assertEqual(set(out32.keys()), {"rank_utility", "calibration", "validity", "pair_noise"})
        self.assertEqual(out32["rank_utility"].shape, (4, 3))
        self.assertEqual(out32["pair_noise"].shape, (4, 2))

        altered_tail = task.detach().clone()
        altered_tail[:, 32:] = altered_tail[:, 32:] + 25.0
        out32_tail = heads(altered_tail, dist.detach(), prefix=32)
        for key in out32:
            torch.testing.assert_close(out32[key], out32_tail[key], rtol=0.0, atol=0.0)

        altered_dist = dist.detach() * -7.0 + 3.0
        out32_dist = heads(task.detach(), altered_dist, prefix=32)
        for task_only_key in ("rank_utility", "calibration", "validity"):
            torch.testing.assert_close(
                out32[task_only_key], out32_dist[task_only_key], rtol=0.0, atol=0.0
            )
        self.assertGreater(float((out32["pair_noise"] - out32_dist["pair_noise"]).abs().max()), 1e-4)

        rank_sum = out32["rank_utility"].sum()
        t_grad, d_grad = torch.autograd.grad(
            rank_sum, (task, dist), retain_graph=True, allow_unused=True
        )
        self.assertGreater(float(t_grad[:, :32].abs().sum()), 1e-5)
        self.assertEqual(float(t_grad[:, 32:].abs().max()), 0.0)
        self.assertIsNone(d_grad)

        pair_sum = out32["pair_noise"].sum()
        t_pair_grad, d_pair_grad = torch.autograd.grad(pair_sum, (task, dist))
        self.assertGreater(float(t_pair_grad[:, :32].abs().sum()), 1e-5)
        self.assertEqual(float(t_pair_grad[:, 32:].abs().max()), 0.0)
        self.assertGreater(float(d_pair_grad[:, :32].abs().sum()), 1e-5)
        self.assertGreater(float(d_pair_grad[:, 32:].abs().sum()), 1e-5)

    def test_matryoshka_prefix_loss_masks_nonfinite_and_nests_gradients(self):
        torch.manual_seed(202)
        heads = JXPrefixHeads(program_slots=2, pair_noise_dim=1)
        task = torch.randn((3, LATENT_WIDTH), dtype=torch.float32, requires_grad=True)
        dist = torch.randn((3, LATENT_WIDTH), dtype=torch.float32, requires_grad=True)

        targets = {
            "rank_utility": torch.tensor([[0.8, 0.2], [0.1, float("nan")], [0.4, 0.9]], dtype=torch.float32),
            "calibration": torch.tensor([[0.7, 0.3], [0.2, 0.5], [0.6, 0.8]], dtype=torch.float32),
            "validity": torch.tensor([[1.0, 0.0], [1.0, 99.0], [0.0, 1.0]], dtype=torch.float32),
            "pair_noise": torch.tensor([[0.15], [0.25], [float("inf")]], dtype=torch.float32),
        }
        masks = {
            "rank_utility": torch.tensor([[True, True], [True, False], [True, True]]),
            "calibration": torch.ones((3, 2), dtype=torch.bool),
            "validity": torch.tensor([[True, True], [True, False], [True, True]]),
            "pair_noise": torch.tensor([[True], [True], [False]]),
        }
        loss32_only, rec32 = matryoshka_prefix_loss(
            heads, task, dist, targets, masks, weights={32: 1.0, 64: 0.0, 96: 0.0, 128: 0.0}
        )
        self.assertEqual(rec32["schema"], SCHEMA)
        self.assertEqual(rec32["prefixes"], PREFIXES)
        t_grad32, d_grad32 = torch.autograd.grad(loss32_only, (task, dist))
        self.assertGreater(float(t_grad32[:, :32].abs().sum()), 1e-5)
        self.assertEqual(float(t_grad32[:, 32:].abs().max()), 0.0)
        self.assertGreater(float(d_grad32[:, 32:].abs().sum()), 1e-5)
        for bad_weights in ([True, 0.0, 0.0, 0.0], ["bad", 0.0, 0.0, 0.0]):
            with self.assertRaises(JXHeadError) as w_err:
                matryoshka_prefix_loss(heads, task, dist, targets, masks, weights=bad_weights)
            self.assertEqual(str(w_err.exception), "jx_head_weights")

        full_loss, full_rec = matryoshka_prefix_loss(heads, task, dist, targets, masks)
        self.assertTrue(torch.isfinite(full_loss))
        self.assertEqual(len(full_rec["per_prefix_loss"]), 4)
        t_grad_full, _ = torch.autograd.grad(full_loss, (task, dist))
        self.assertGreater(float(t_grad_full[:, 96:128].abs().sum()), 1e-6)

        bad_masks = dict(masks, rank_utility=torch.ones((3, 2), dtype=torch.bool))
        with self.assertRaises(JXHeadError) as nan_err:
            matryoshka_prefix_loss(heads, task, dist, targets, bad_masks)
        self.assertEqual(str(nan_err.exception), "jx_head_nonfinite")

        bad_val_masks = dict(masks, validity=torch.ones((3, 2), dtype=torch.bool))
        with self.assertRaises(JXHeadError) as lbl_err:
            matryoshka_prefix_loss(heads, task, dist, targets, bad_val_masks)
        self.assertEqual(str(lbl_err.exception), "jx_head_label")
        self.assertNotIn("99", str(lbl_err.exception))

        frozen_heads = JXPrefixHeads(program_slots=2, pair_noise_dim=1)
        empty_masks = {key: torch.zeros_like(val) for key, val in masks.items()}
        zero_loss, zero_rec = matryoshka_prefix_loss(
            frozen_heads, task.detach(), dist.detach(), targets, empty_masks
        )
        self.assertTrue(zero_loss.requires_grad)
        self.assertEqual(zero_rec["active_entries"], 0)
        zero_loss.backward()
        for param in frozen_heads.parameters():
            self.assertIsNotNone(param.grad)
            self.assertEqual(float(param.grad.abs().max()), 0.0)

    def test_preference_reversal_loss_and_additive_offset_control(self):
        outcomes_a = torch.tensor([0.90, 0.20, 0.50], dtype=torch.float64)
        outcomes_b = torch.tensor([0.15, 0.85, 0.50], dtype=torch.float64)
        mask_a = torch.tensor([True, True, True])
        mask_b = torch.tensor([True, True, True])

        z_a = torch.tensor([1.5, -1.5], dtype=torch.float64)
        z_b = torch.tensor([-1.5, 1.5], dtype=torch.float64)
        program_vectors = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64
        )
        ctrl = task_additive_offset_control(
            z_a,
            z_b,
            program_vectors,
            outcomes_a,
            outcomes_b,
            mask_a,
            mask_b,
            offset_a=3.2,
            offset_b=-4.7,
            margin=0.05,
        )
        self.assertEqual(ctrl["schema"], SCHEMA)
        self.assertGreaterEqual(ctrl["reversal_pairs"], 1)
        self.assertAlmostEqual(ctrl["additive_loss"], math.log(2.0), places=12)
        self.assertEqual(ctrl["additive_advantage"], 0.0)
        self.assertEqual(ctrl["additive_offset_grad_norm"], 0.0)
        self.assertEqual(ctrl["static_prior_grad_norm"], 0.0)
        self.assertLess(ctrl["bilinear_loss"], ctrl["additive_loss"])
        self.assertGreater(ctrl["bilinear_advantage"], 1.0)
        self.assertGreater(ctrl["bilinear_grad_norm"], 1e-4)

        sa_zero = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
        sb_zero = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
        same_order_u = torch.tensor([0.9, 0.1], dtype=torch.float32)
        zero_loss, zero_rec = preference_reversal_loss(
            sa_zero,
            sb_zero,
            same_order_u,
            same_order_u,
            torch.tensor([True, True]),
            torch.tensor([True, True]),
        )
        self.assertEqual(zero_rec["reversal_pairs"], 0)
        self.assertEqual(float(zero_loss), 0.0)
        grads = torch.autograd.grad(zero_loss, (sa_zero, sb_zero))
        self.assertEqual(float(grads[0].abs().max()), 0.0)
        self.assertEqual(float(grads[1].abs().max()), 0.0)

    def test_split_invariance_infonce_masks_same_family_siblings(self):
        z_a = torch.tensor(
            [[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        z_b = torch.tensor(
            [[1.0, 0.0, 0.0], [0.92, 0.08, 0.0], [0.0, 0.9, 0.1]],
            dtype=torch.float64,
            requires_grad=True,
        )
        families = ("fam-alpha", "fam-alpha", "fam-beta")
        loss, rec = split_invariance_infonce(z_a, z_b, families, temperature=0.2)
        self.assertEqual(rec["schema"], SCHEMA)
        self.assertEqual(rec["masked_sibling_pairs"], 2)
        self.assertEqual(rec["eligible_negative_pairs"], 4)
        self.assertTrue(torch.isfinite(loss))

        # Verify sibling z_b[1] is completely masked out of anchor 0's forward contrast:
        # perturbing z_b[1] must only affect row 1's own positive term and row 2's negative denominator,
        # whereas under distinct families the loss increases sharply because sibling 1 is penalized as a negative!
        loss_unmasked, rec_unmasked = split_invariance_infonce(
            z_a, z_b, ("fam-1", "fam-2", "fam-3"), temperature=0.2
        )
        self.assertEqual(rec_unmasked["masked_sibling_pairs"], 0)
        self.assertLess(float(loss), float(loss_unmasked))

        grad_a, grad_b = torch.autograd.grad(loss, (z_a, z_b))
        self.assertTrue(torch.isfinite(grad_a).all())
        self.assertTrue(torch.isfinite(grad_b).all())

        with self.assertRaises(JXHeadError):
            split_invariance_infonce(torch.zeros((2, 4)), z_b[:2].float(), ("a", "b"))

        za_single = z_a.detach().clone().requires_grad_(True)
        zb_single = z_b.detach().clone().requires_grad_(True)
        single_loss, single_rec = split_invariance_infonce(
            za_single, zb_single, ("fam-same", "fam-same", "fam-same"), temperature=0.2
        )
        self.assertEqual(single_rec["eligible_negative_pairs"], 0)
        self.assertEqual(single_rec["masked_sibling_pairs"], 6)
        self.assertTrue(single_loss.requires_grad)
        self.assertEqual(float(single_loss), 0.0)
        single_loss.backward()
        self.assertEqual(float(za_single.grad.abs().max()), 0.0)
        self.assertEqual(float(zb_single.grad.abs().max()), 0.0)

    def test_target_swap_disentangle_pulls_dist_and_pushes_task(self):
        task_a = torch.tensor([[1.0, 0.1, 0.0], [0.2, 1.0, 0.0]], dtype=torch.float32, requires_grad=True)
        task_b = torch.tensor([[0.98, 0.12, 0.0], [0.22, 0.96, 0.0]], dtype=torch.float32, requires_grad=True)
        dist_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32, requires_grad=True)
        dist_b = torch.tensor([[-0.5, 0.8, 0.0], [0.7, -0.4, 0.0]], dtype=torch.float32, requires_grad=True)

        loss, rec = target_swap_disentangle_loss(task_a, dist_a, task_b, dist_b, margin=0.5)
        self.assertEqual(rec["schema"], SCHEMA)
        self.assertGreater(rec["dist_pull"], 0.5)
        self.assertGreater(rec["task_push"], 0.4)
        grads = torch.autograd.grad(loss, (task_a, dist_a, task_b, dist_b))
        for g in grads:
            self.assertGreater(float(g.norm()), 1e-4)

        step = 0.25
        ta2 = task_a.detach() - step * grads[0]
        da2 = dist_a.detach() - step * grads[1]
        tb2 = task_b.detach() - step * grads[2]
        db2 = dist_b.detach() - step * grads[3]
        loss_after, rec_after = target_swap_disentangle_loss(ta2, da2, tb2, db2, margin=0.5)
        self.assertLess(float(loss_after), float(loss))
        self.assertLess(rec_after["dist_pull"], rec["dist_pull"])
        self.assertLess(rec_after["task_push"], rec["task_push"])


if __name__ == "__main__":
    unittest.main()
