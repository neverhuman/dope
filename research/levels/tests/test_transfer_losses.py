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
        self.assertEqual(record["tstr_integration"], "not_connected")
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

    def test_baseline_objective_is_unchanged(self):
        source = inspect.getsource(baseline.objective)
        self.assertIn("def objective", source)
        self.assertEqual(TSTR_INTEGRATION, "not_connected")
        self.assertNotIn("transfer_losses", source)
        self.assertEqual(inspect.getsource(baseline.objective), source)
