"""Analytic decision-loss fixtures. No training run and no weight update."""
import inspect
import math
import unittest

import torch
from torch.nn import functional as F

from research.levels import losses as baseline
from research.levels.transfer_losses import (
    NONDIFFERENTIABLE,
    TSTR_INTEGRATION,
    DecisionLossError,
    coupled_tstr_loss,
    masked_calibration,
    pairwise_logistic,
    reduce_tasks,
    teacher_distillation,
    top_choice_indicator,
    validity_bce,
)


def _scores(*values):
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


class TransferLossTests(unittest.TestCase):
    def test_pairwise_sign_mask_and_denominator(self):
        scores = _scores(0.0, 0.0, 5.0)
        outcomes = torch.tensor([1.0, 0.0, 100.0], dtype=torch.float64)
        mask = torch.tensor([True, True, False])
        loss, record = pairwise_logistic(scores, outcomes, mask, higher_is_better=True)
        self.assertEqual(record["denominator"], 1)
        self.assertAlmostEqual(float(loss), math.log(2), places=12)
        grads = torch.autograd.grad(loss, scores)[0]
        self.assertLess(float(grads[0]), 0.0)
        self.assertGreater(float(grads[1]), 0.0)
        self.assertEqual(float(grads[2]), 0.0)

    def test_lower_is_better_flips_the_preferred_score(self):
        scores = _scores(0.0, 0.0)
        outcomes = torch.tensor([1.0, 4.0], dtype=torch.float64)
        mask = torch.tensor([True, True])
        loss, record = pairwise_logistic(scores, outcomes, mask, higher_is_better=False)
        self.assertEqual(record["denominator"], 1)
        grads = torch.autograd.grad(loss, scores)[0]
        self.assertLess(float(grads[0]), 0.0)
        self.assertGreater(float(grads[1]), 0.0)

    def test_query_values_are_refused_without_echo(self):
        scores = _scores(0.0, 1.0)
        outcomes = torch.tensor([1.0, 0.0], dtype=torch.float64)
        mask = torch.tensor([True, True])
        with self.assertRaises(DecisionLossError) as caught:
            pairwise_logistic(
                scores,
                outcomes,
                mask,
                higher_is_better=True,
                query_values=torch.tensor([12345.6]),
            )
        text = str(caught.exception)
        self.assertIn("decision_loss_query", text)
        self.assertNotIn("12345", text)

    def test_validity_and_calibration_use_the_mask_denominator(self):
        logits = _scores(0.0, 4.0)
        labels = torch.tensor([1.0, 7.0], dtype=torch.float64)
        mask = torch.tensor([True, False])
        loss, record = validity_bce(logits, labels, mask)
        self.assertEqual(record["denominator"], 1)
        self.assertTrue(torch.isfinite(loss))
        bad = torch.tensor([7.0], dtype=torch.float64)
        with self.assertRaises(DecisionLossError) as caught:
            validity_bce(_scores(0.0), bad, torch.tensor([True]))
        self.assertNotIn("7", str(caught.exception))
        scores = _scores(0.0, 9.0)
        outcomes = torch.tensor([1.0, -5.0], dtype=torch.float64)
        calibrated, counted = masked_calibration(
            scores, outcomes, torch.tensor([True, False]), higher_is_better=True
        )
        self.assertEqual(counted["denominator"], 1)
        self.assertAlmostEqual(float(calibrated), 0.5, places=12)
        wide = _scores(0.0)
        wide_loss, _wide_record = masked_calibration(
            wide, torch.tensor([4.0], dtype=torch.float64), torch.tensor([True]), higher_is_better=True
        )
        self.assertAlmostEqual(float(wide_loss), 3.5, places=12)

    def test_teacher_gradient_stops_and_top_choice_is_labeled(self):
        student = _scores(0.0, 0.0)
        teacher = torch.tensor([1.0, 3.0], dtype=torch.float64, requires_grad=True)
        loss, record = teacher_distillation(student, teacher, torch.tensor([True, True]))
        self.assertEqual(record["tstr_integration"], "opt_in_v4_objectives")
        self.assertEqual(record["nondifferentiable"], NONDIFFERENTIABLE)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        choice = top_choice_indicator(
            torch.tensor([0.2, 5.0, 9.0]), torch.tensor([True, True, False])
        )
        self.assertFalse(choice["differentiable"])
        self.assertEqual(choice["component"], "top_choice_indicator")
        self.assertEqual(choice["choice"], 1)

    def test_task_reductions_keep_separate_denominators(self):
        losses = torch.tensor([1.0, 3.0], dtype=torch.float64)
        uniform, uniform_record = reduce_tasks(losses, [1, 3], mode="uniform")
        weighted, weighted_record = reduce_tasks(losses, [1, 3], mode="pair_weighted")
        self.assertAlmostEqual(float(uniform), 2.0, places=12)
        self.assertEqual(uniform_record["denominator"], 2)
        self.assertAlmostEqual(float(weighted), 2.5, places=12)
        self.assertEqual(weighted_record["denominator"], 4)
        sparse = torch.tensor([float("nan"), 1.0, 9.0], dtype=torch.float64, requires_grad=True)
        uniform_sparse, uniform_sparse_record = reduce_tasks(sparse, [0, 1, 1], mode="uniform")
        self.assertEqual(uniform_sparse_record["denominator"], 2)
        self.assertAlmostEqual(float(uniform_sparse), 5.0, places=12)
        sparse_grad = torch.autograd.grad(uniform_sparse, sparse)[0]
        self.assertEqual(float(sparse_grad[0]), 0.0)
        balanced, balanced_record = reduce_tasks(
            torch.tensor([1.0, 3.0, 9.0], dtype=torch.float64),
            [1, 1, 1],
            mode="family_balanced",
            family_ids=("alpha", "alpha", "beta"),
        )
        self.assertEqual(balanced_record["denominator"], 2)
        self.assertAlmostEqual(float(balanced), 5.5, places=12)
        frac_small, frac_small_rec = reduce_tasks(losses, [0.4, 0.4], mode="pair_weighted")
        self.assertAlmostEqual(float(frac_small), 2.0, places=12)
        self.assertAlmostEqual(float(frac_small_rec["denominator"]), 0.8, places=12)
        frac_over, frac_over_rec = reduce_tasks(losses, [0.6, 0.6], mode="pair_weighted")
        self.assertAlmostEqual(float(frac_over), 2.0, places=12)
        self.assertAlmostEqual(float(frac_over_rec["denominator"]), 1.2, places=12)
        for bad_counts in ([float("nan"), 1.0], [float("inf"), 1.0]):
            with self.assertRaises(DecisionLossError) as nf_err:
                reduce_tasks(losses, bad_counts, mode="pair_weighted")
            self.assertEqual(str(nf_err.exception), "decision_loss_nonfinite")
        with self.assertRaises(DecisionLossError) as neg_err:
            reduce_tasks(losses, [-0.5, 1.0], mode="pair_weighted")
        self.assertEqual(str(neg_err.exception), "decision_loss_weights")

    def test_masked_nonfinite_stays_out_of_the_graph(self):
        scores = torch.tensor([0.0, 0.0, float("nan")], dtype=torch.float64, requires_grad=True)
        outcomes = torch.tensor([1.0, 0.0, 8.0], dtype=torch.float64)
        mask = torch.tensor([True, True, False])
        loss, record = pairwise_logistic(scores, outcomes, mask, higher_is_better=True)
        self.assertEqual(record["denominator"], 1)
        self.assertAlmostEqual(float(loss), math.log(2), places=12)
        grads = torch.autograd.grad(loss, scores)[0]
        self.assertEqual(float(grads[2]), 0.0)
        active = torch.tensor([float("nan"), 0.0], dtype=torch.float64, requires_grad=True)
        with self.assertRaises(DecisionLossError) as caught:
            pairwise_logistic(
                active,
                torch.tensor([1.0, 0.0], dtype=torch.float64),
                torch.tensor([True, True]),
                higher_is_better=True,
            )
        self.assertIn("decision_loss_nonfinite", str(caught.exception))
        self.assertNotIn("nan", str(caught.exception).lower())

    def test_common_logit_shift_has_no_class_probability_gradient(self):
        labels = torch.arange(4) % 3
        shift = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        logits = torch.zeros(4, 3, dtype=torch.float64)
        F.cross_entropy(logits + shift, labels).backward()
        self.assertLess(abs(float(shift.grad)), 1e-12)
        class_bias = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        F.cross_entropy(logits + class_bias, labels).backward()
        self.assertGreater(float(class_bias.grad.abs().sum()), 1e-8)
        self.assertGreater(float(class_bias.grad.max() - class_bias.grad.min()), 1e-8)

    def test_coupled_tstr_gradients_and_query_isolation(self):
        gen = torch.Generator().manual_seed(17)
        fit_x = torch.randn((24, 3), generator=gen, dtype=torch.float32)
        true_w = torch.tensor([1.5, -0.8, 0.4], dtype=torch.float32)
        fit_y = fit_x @ true_w + 0.2
        query_x = torch.randn((16, 3), generator=gen, dtype=torch.float32)
        query_y = (query_x @ true_w + 0.2).requires_grad_(True)

        def encode_fit(fx, fy):
            return torch.cat([fx.mean(dim=0), fx.std(dim=0, correction=0), fy.mean().unsqueeze(0)])

        fit_summary_before = encode_fit(fit_x, fit_y)
        perturbed_query_y = (query_y.detach() * -9.0 + 42.0).requires_grad_(True)
        fit_summary_after = encode_fit(fit_x, fit_y)
        self.assertTrue(torch.equal(fit_summary_before, fit_summary_after))

        synth_x = fit_x.clone().detach().requires_grad_(True)
        synth_y = (0.25 * fit_y).clone().detach().requires_grad_(True)
        loss, record = coupled_tstr_loss(
            fit_x, fit_y, query_x, query_y, synth_x, synth_y, ridge_lambda=1e-2
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(record["denominator"], 16)
        self.assertEqual(record["kind"], "coupled_tstr")
        self.assertEqual(record["tstr_integration"], "opt_in_v4_objectives")
        loss.backward()
        self.assertIsNone(query_y.grad)
        self.assertIsNotNone(synth_x.grad)
        self.assertIsNotNone(synth_y.grad)
        self.assertGreater(float(synth_x.grad.abs().sum()), 1e-5)
        self.assertGreater(float(synth_y.grad.abs().sum()), 1e-5)

        updated_y = (synth_y.detach() - 0.8 * synth_y.grad.detach()).requires_grad_(True)
        loss_after, _ = coupled_tstr_loss(
            fit_x, fit_y, query_x, perturbed_query_y.detach(), synth_x.detach(), updated_y
        )
        loss_aligned, _ = coupled_tstr_loss(
            fit_x, fit_y, query_x, query_y.detach(), synth_x.detach(), updated_y
        )
        self.assertLess(float(loss_aligned), float(loss))
        self.assertTrue(torch.isfinite(loss_after))

        with self.assertRaises(DecisionLossError) as alias_err:
            coupled_tstr_loss(fit_x, fit_y, fit_x, query_y[:24] if len(query_y) >= 24 else fit_y, synth_x, synth_y)
        self.assertIn("decision_loss_", str(alias_err.exception))
        with self.assertRaises(DecisionLossError):
            coupled_tstr_loss(fit_x[:0], fit_y[:0], query_x, query_y, synth_x, synth_y)
        bad_synth = synth_x.detach().clone()
        bad_synth[0, 0] = float("nan")
        with self.assertRaises(DecisionLossError) as nan_err:
            coupled_tstr_loss(fit_x, fit_y, query_x, query_y, bad_synth, synth_y)
        self.assertEqual(str(nan_err.exception), "decision_loss_nonfinite")
        with self.assertRaises(DecisionLossError):
            coupled_tstr_loss(fit_x, fit_y, query_x, query_y, synth_x, synth_y, ridge_lambda=0.0)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            bf_sx = fit_x.detach().to(torch.bfloat16).requires_grad_(True)
            bf_sy = (0.25 * fit_y).detach().to(torch.bfloat16).requires_grad_(True)
            bf_loss, _ = coupled_tstr_loss(
                fit_x.to(torch.bfloat16),
                fit_y.to(torch.bfloat16),
                query_x.to(torch.bfloat16),
                query_y.detach().to(torch.bfloat16),
                bf_sx,
                bf_sy,
            )
        self.assertEqual(bf_loss.dtype, torch.float32)
        bf_loss.backward()
        self.assertIsNotNone(bf_sx.grad)
        self.assertTrue(torch.isfinite(bf_sx.grad).all())

        collinear_s = torch.tensor([[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]], dtype=torch.float32)
        with self.assertRaises(DecisionLossError) as sing_err:
            coupled_tstr_loss(
                fit_x,
                fit_y,
                query_x,
                query_y.detach(),
                collinear_s,
                torch.tensor([0.5, 0.5], dtype=torch.float32),
                ridge_lambda=1e-46,
            )
        self.assertEqual(str(sing_err.exception), "decision_loss_singular")

    def test_baseline_objective_is_unchanged(self):
        source = inspect.getsource(baseline.objective)
        self.assertIn("def objective", source)
        self.assertEqual(TSTR_INTEGRATION, "opt_in_v4_objectives")
        self.assertNotIn("transfer_losses", source)
        self.assertEqual(inspect.getsource(baseline.objective), source)
