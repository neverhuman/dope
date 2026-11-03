import hashlib
import random
import tempfile
import unittest
from pathlib import Path
TEST_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "target" / "v4-training-tests"
TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


import numpy as np
import torch

from research.levels.v4_trainer import (
    BatchCell,
    IdentityMismatch,
    NonFiniteUpdate,
    RoleViolation,
    TrainingIdentity,
    V4Trainer,
    V4TrainerConfig,
    build_shape_batches,
    collate_cells,
)


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def identity(seed="fixture"):
    return TrainingIdentity(*(digest(seed + part) for part in ("source", "wire", "norm", "fit", "objective", "software")))


def cells():
    result = []
    for i, (rows, cols, qrows) in enumerate(((3, 2, 2), (5, 3, 1), (2, 2, 3), (4, 3, 2))):
        x = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) / 10 + i
        q = torch.arange(qrows * cols, dtype=torch.float32).reshape(qrows, cols) / 7 + i / 3
        result.append(BatchCell(
            cell_id=f"cell-{i}", x_fit=x, y_fit=x.sum(1) * 0.2,
            x_query=q, y_query=q.sum(1) * 0.2,
            schema={"columns": cols}, native_report={"fixture": i},
            fit_row_ids=tuple(f"family-{i}-fit-{j}" for j in range(rows)),
            query_row_ids=tuple(f"family-{i}-query-{j}" for j in range(qrows)),
            keep_mask=torch.ones(23, dtype=torch.bool), family_id=f"family-{i}",
        ))
    return result


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def make_loss(scale=1.0, inject_nan=False):
    def loss(model, batch):
        pieces = []
        for i in range(len(batch.cell_ids)):
            nf = int(batch.fit_row_mask[i].sum())
            nq = int(batch.query_row_mask[i].sum())
            cols = int(batch.fit_column_mask[i].sum())
            x = batch.x_query[i, :nq, :cols]
            y = batch.y_query[i, :nq]
            padded = torch.zeros((nq, 3), dtype=x.dtype)
            padded[:, :cols] = x
            pieces.append(torch.nn.functional.mse_loss(model(padded), y))
        value = scale * torch.stack(pieces).mean()
        if inject_nan:
            value = value + _FiniteForwardBadBackward.apply(model.linear.weight.sum())
        # Exercise all three RNGs so their checkpoint restoration is covered.
        jitter = random.random() + float(np.random.random()) + float(torch.rand(()))
        return value + value.new_tensor(jitter * 1e-6)
    return loss


class _FiniteForwardBadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.new_zeros(())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * torch.tensor(float("nan"))


def trainer(seed=7, ident=None, inject_nan=False):
    torch.manual_seed(991)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.2)
    return V4Trainer(
        model, optimizer, cells(), make_loss(inject_nan=inject_nan), ident or identity(),
        V4TrainerConfig(batch_size=1, row_bucket=4, column_bucket=2, seed=seed),
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8),
    )


def weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class V4TrainingTests(unittest.TestCase):
    def test_shape_buckets_padding_masks_and_no_truncation(self):
        data = cells()
        config = V4TrainerConfig(batch_size=2, row_bucket=4, column_bucket=2, max_padded_cells=1000, seed=31)
        first = build_shape_batches(data, config)
        self.assertEqual(first, build_shape_batches(data, config))
        self.assertEqual(sorted(x for batch in first for x in batch), sorted(c.cell_id for c in data))
        batch = collate_cells([data[0], data[1]], config)
        self.assertEqual(tuple(batch.x_fit.shape), (2, 8, 4))
        self.assertEqual(batch.fit_row_mask.sum(1).tolist(), [3, 5])
        self.assertEqual(batch.query_row_mask.sum(1).tolist(), [2, 1])
        self.assertEqual(batch.fit_column_mask.sum(1).tolist(), [2, 3])
        self.assertTrue(torch.equal(batch.keep_masks[0], torch.ones(23, dtype=torch.bool)))
        self.assertFalse(batch.fit_row_mask[0, 3])
        self.assertFalse(batch.fit_column_mask[0, 2:].any())
        self.assertEqual(float(batch.x_fit[0, 0, 2]), 0.0)
        with self.assertRaisesRegex(Exception, "padded_allocation_limit"):
            collate_cells(data, V4TrainerConfig(batch_size=4, max_padded_cells=1))

    def test_fit_and_gradient_query_memberships_must_be_disjoint_and_role_safe(self):
        base = cells()[0]
        with self.assertRaises(RoleViolation):
            BatchCell(**{**base.__dict__, "query_role": "test"}).validate()
        with self.assertRaises(RoleViolation):
            BatchCell(**{**base.__dict__, "query_row_ids": base.fit_row_ids[:2]}).validate()

    def test_uninterrupted_and_checkpoint_resume_match_weights_cursor_and_rng(self):
        random.seed(17)
        np.random.seed(18)
        torch.manual_seed(19)
        uninterrupted = trainer()
        uninterrupted.fit(2)
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT_ROOT) as tmp:
            uninterrupted.save_checkpoint(tmp)
            uninterrupted.fit(2)
            expected = weights(uninterrupted.model)
            expected_cursor = uninterrupted.cursor
            expected_rng = torch.rand(3)

            random.seed(17)
            np.random.seed(18)
            torch.manual_seed(19)
            resumed = trainer()
            resumed.load_checkpoint(tmp)
            resumed.fit(2)
            self.assertEqual(resumed.cursor, expected_cursor)
            for key, tensor in expected.items():
                self.assertTrue(torch.equal(tensor, resumed.model.state_dict()[key]), key)
            self.assertTrue(torch.equal(expected_rng, torch.rand(3)))

    def test_checkpoint_identity_and_model_shapes_are_strict(self):
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT_ROOT) as tmp:
            source = trainer()
            source.fit(1)
            source.save_checkpoint(tmp)
            other = trainer(ident=identity("changed"))
            with self.assertRaises(IdentityMismatch):
                other.load_checkpoint(tmp)
            differently_shaped = TinyModel()
            differently_shaped.linear = torch.nn.Linear(4, 1)
            optimizer = torch.optim.SGD(differently_shaped.parameters(), lr=0.03, momentum=0.2)
            target = V4Trainer(differently_shaped, optimizer, cells(), make_loss(), identity(), scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 2))
            with self.assertRaises(IdentityMismatch):
                target.load_checkpoint(tmp)

    def test_nonfinite_gradient_refuses_update_and_keeps_cursor(self):
        subject = trainer(inject_nan=True)
        before = weights(subject.model)
        with self.assertRaises(NonFiniteUpdate):
            subject.fit_step()
        self.assertEqual(subject.update_count, 0)
        self.assertEqual(subject.cursor, 0)
        for key, tensor in before.items():
            self.assertTrue(torch.equal(tensor, subject.model.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
