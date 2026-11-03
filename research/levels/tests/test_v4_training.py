import hashlib
import copy
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


import numpy as np
import torch

from research.levels.v4_trainer import (
    BatchCell,
    BoundedPrefetcher,
    IdentityMismatch,
    NonFiniteUpdate,
    RoleViolation,
    TrainerError,
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
        with tempfile.TemporaryDirectory() as tmp:
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
        with tempfile.TemporaryDirectory() as tmp:
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
        self.assertEqual(subject.attempt_count, 1)
        self.assertEqual(subject.failed_update_count, 1)
        self.assertEqual(subject.cursor, 0)
        self.assertEqual(subject.counters()["accumulation_steps"], 1)
        self.assertEqual(subject.counters()["failed_update_count"], 1)
        for key, tensor in before.items():
            self.assertTrue(torch.equal(tensor, subject.model.state_dict()[key]))

    def test_accumulation_is_explicitly_single_step_only(self):
        self.assertEqual(V4TrainerConfig().accumulation_steps, 1)
        with self.assertRaisesRegex(Exception, "accumulation_steps_other_than_one"):
            V4TrainerConfig(accumulation_steps=2)
        with self.assertRaisesRegex(Exception, "accumulation_steps_other_than_one"):
            V4TrainerConfig(accumulation_steps=True)

    def test_callback_and_mutating_optimizer_failures_roll_back_update_state(self):
        subject = trainer()
        before_model = weights(subject.model)
        before_optimizer = copy.deepcopy(subject.optimizer.state_dict())
        before_rng = torch.get_rng_state().clone()
        def bad_loss(model, batch):
            torch.rand(10)
            with torch.no_grad():
                next(model.parameters()).add_(7)
            raise RuntimeError("fixture_callback_failure")
        subject.loss_fn = bad_loss
        with self.assertRaisesRegex(RuntimeError, "fixture_callback_failure"):
            subject.fit_step()
        self.assertEqual(subject.cursor, 0)
        self.assertEqual(subject.update_count, 0)
        self.assertEqual(subject.attempt_count, 1)
        self.assertEqual(subject.failed_update_count, 1)
        self.assertEqual(subject.optimizer.state_dict(), before_optimizer)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        for key, tensor in before_model.items():
            self.assertTrue(torch.equal(tensor, subject.model.state_dict()[key]))

        subject.loss_fn = make_loss()
        before_model = weights(subject.model)
        before_optimizer = copy.deepcopy(subject.optimizer.state_dict())
        before_rng = torch.get_rng_state().clone()
        def mutating_failure():
            with torch.no_grad():
                for parameter in subject.model.parameters():
                    parameter.add_(13)
            subject.optimizer.param_groups[0]["lr"] = 999.0
            raise RuntimeError("fixture_optimizer_failure")
        subject.optimizer.step = mutating_failure
        with self.assertRaisesRegex(RuntimeError, "fixture_optimizer_failure"):
            subject.fit_step()
        self.assertEqual(subject.cursor, 0)
        self.assertEqual(subject.update_count, 0)
        self.assertEqual(subject.attempt_count, 2)
        self.assertEqual(subject.failed_update_count, 2)
        self.assertEqual(subject.optimizer.state_dict(), before_optimizer)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        for key, tensor in before_model.items():
            self.assertTrue(torch.equal(tensor, subject.model.state_dict()[key]))

    def test_partial_optimizer_failure_rolls_back_all_state_and_accounts_once(self):
        subject = trainer()
        for parameter in subject.model.parameters():
            parameter.grad = torch.full_like(parameter, 0.125)
        mutated_parameter = next(subject.model.parameters())
        subject.optimizer.state[mutated_parameter]["momentum_buffer"] = torch.full_like(mutated_parameter, 0.25)
        before_grads = [parameter.grad.detach().clone() for parameter in subject.model.parameters()]
        before_model = weights(subject.model)
        before_optimizer = copy.deepcopy(subject.optimizer.state_dict())
        before_scheduler = copy.deepcopy(subject.scheduler.state_dict())
        before_python_rng = random.getstate()
        before_numpy_rng = np.random.get_state()
        before_torch_rng = torch.get_rng_state().clone()
        before_compute = subject.compute_seconds
        original_error = RuntimeError("fixture_partial_optimizer_failure")
        def partial_step():
            with torch.no_grad():
                mutated_parameter.add_(13)
            subject.optimizer.param_groups[0]["lr"] = 999.0
            subject.optimizer.state[mutated_parameter]["momentum_buffer"].fill_(77.0)
            subject.scheduler.last_epoch += 1
            random.random()
            np.random.random()
            torch.rand(5)
            raise original_error
        subject.optimizer.step = partial_step
        with patch("research.levels.v4_trainer.time.perf_counter", side_effect=[100.0, 102.5]):
            with self.assertRaisesRegex(RuntimeError, "fixture_partial_optimizer_failure") as caught:
                subject.fit_step()
        self.assertIs(caught.exception, original_error)
        for key, tensor in before_model.items():
            self.assertTrue(torch.equal(tensor, subject.model.state_dict()[key]), key)
        for parameter, expected_grad in zip(subject.model.parameters(), before_grads):
            self.assertTrue(torch.equal(parameter.grad, expected_grad))
        parameter_id = next(iter(before_optimizer["state"]))
        restored_momentum = subject.optimizer.state[mutated_parameter]["momentum_buffer"]
        self.assertTrue(torch.equal(restored_momentum, before_optimizer["state"][parameter_id]["momentum_buffer"]))
        self.assertEqual(subject.optimizer.param_groups[0]["lr"], before_optimizer["param_groups"][0]["lr"])
        self.assertEqual(subject.scheduler.state_dict(), before_scheduler)
        self.assertEqual(random.getstate(), before_python_rng)
        after_numpy_rng = np.random.get_state()
        self.assertEqual(before_numpy_rng[0], after_numpy_rng[0])
        np.testing.assert_array_equal(before_numpy_rng[1], after_numpy_rng[1])
        self.assertEqual(before_numpy_rng[2:], after_numpy_rng[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch_rng))
        self.assertEqual((subject.attempt_count, subject.failed_update_count, subject.update_count, subject.cursor),
                         (1, 1, 0, 0))
        self.assertEqual(subject.compute_seconds, before_compute + 2.5)

    def test_rollback_exception_is_wrapped_and_failure_accounted_once(self):
        class OriginalUpdateError(Exception):
            pass
        class RollbackFault(Exception):
            pass
        subject = trainer()
        original_error = OriginalUpdateError("fixture_original_update_failure")
        rollback_error = RollbackFault("fixture_rollback_failure")
        def fail_during_update(model, batch):
            raise original_error
        def fail_during_rollback(*args, **kwargs):
            raise rollback_error
        subject.loss_fn = fail_during_update
        subject.model.load_state_dict = fail_during_rollback
        before_compute = subject.compute_seconds
        with patch("research.levels.v4_trainer.time.perf_counter", side_effect=[10.0, 13.25]):
            with self.assertRaises(TrainerError) as caught:
                subject.fit_step()
        self.assertIs(caught.exception.__cause__, rollback_error)
        self.assertEqual((subject.attempt_count, subject.failed_update_count, subject.update_count, subject.cursor),
                         (1, 1, 0, 0))
        self.assertEqual(subject.compute_seconds, before_compute + 3.25)

    def test_successful_transaction_does_not_restore_snapshot(self):
        subject = trainer()
        calls = []
        original_load_state_dict = subject.model.load_state_dict
        def spy_load_state_dict(*args, **kwargs):
            calls.append(None)
            return original_load_state_dict(*args, **kwargs)
        subject.model.load_state_dict = spy_load_state_dict
        subject.fit_step()
        self.assertEqual(calls, [])
        self.assertEqual((subject.attempt_count, subject.failed_update_count, subject.update_count, subject.cursor),
                         (1, 0, 1, 1))

    def test_base_exception_keeps_previous_nonrollback_behavior(self):
        class ControlStop(BaseException):
            pass
        subject = trainer()
        before_model = weights(subject.model)
        before_rng = torch.get_rng_state().clone()
        before_compute = subject.compute_seconds
        def stop_after_mutation(model, batch):
            with torch.no_grad():
                next(model.parameters()).add_(5)
            torch.rand(5)
            raise ControlStop("fixture_control_stop")
        subject.loss_fn = stop_after_mutation
        with self.assertRaises(ControlStop):
            subject.fit_step()
        self.assertFalse(torch.equal(before_model["linear.weight"], subject.model.state_dict()["linear.weight"]))
        self.assertFalse(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual((subject.attempt_count, subject.failed_update_count, subject.update_count, subject.cursor),
                         (1, 0, 0, 0))
        self.assertEqual(subject.compute_seconds, before_compute)

    def test_checkpoint_restores_and_validates_counters(self):
        source = trainer(inject_nan=True)
        with self.assertRaises(NonFiniteUpdate):
            source.fit_step()
        source.loss_fn = make_loss()
        report = source.fit_step()
        self.assertEqual((report["attempt_count"], report["failed_update_count"], report["update_count"]), (2, 1, 1))
        self.assertEqual((source.attempt_count, source.failed_update_count, source.update_count), (2, 1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            source.save_checkpoint(tmp)
            target = trainer()
            target.load_checkpoint(tmp)
            self.assertEqual((target.attempt_count, target.failed_update_count, target.update_count), (2, 1, 1))
            self.assertEqual(target.setup_seconds, source.setup_seconds)
            self.assertEqual(target.compute_seconds, source.compute_seconds)

            pointer = json.loads((Path(tmp) / "CURRENT").read_text())
            state_path = Path(tmp) / pointer["state"]
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            state["counters"]["failed_update_count"] = 0
            torch.save(state, state_path)
            invalid_target = trainer()
            before = weights(invalid_target.model)
            before_rng = torch.get_rng_state().clone()
            with self.assertRaisesRegex(TrainerError, "checkpoint_counters_invalid"):
                invalid_target.load_checkpoint(tmp)
            self.assertEqual((invalid_target.attempt_count, invalid_target.failed_update_count, invalid_target.update_count), (0, 0, 0))
            self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
            for key, tensor in before.items():
                self.assertTrue(torch.equal(tensor, invalid_target.model.state_dict()[key]))

    def test_checkpoint_read_boundary_translates_ordinary_fault_and_preserves_state(self):
        class DecoderFailure(Exception):
            pass
        class DecoderAbort(BaseException):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            source = trainer()
            source.save_checkpoint(tmp)
            target = trainer()
            model_before = weights(target.model)
            optimizer_before = copy.deepcopy(target.optimizer.state_dict())
            scheduler_before = copy.deepcopy(target.scheduler.state_dict())
            cursor_before = (target.permutation[:], target.cursor, target.update_count,
                             target.attempt_count, target.failed_update_count)
            python_rng_before = random.getstate()
            numpy_rng_before = np.random.get_state()
            torch_rng_before = torch.get_rng_state().clone()
            time_before = (target.setup_seconds, target.compute_seconds)

            def assert_unchanged():
                for key, value in model_before.items():
                    self.assertTrue(torch.equal(value, target.model.state_dict()[key]), key)
                self.assertEqual(target.optimizer.state_dict(), optimizer_before)
                self.assertEqual(target.scheduler.state_dict(), scheduler_before)
                self.assertEqual((target.setup_seconds, target.compute_seconds), time_before)
                self.assertEqual((target.permutation, target.cursor, target.update_count,
                                  target.attempt_count, target.failed_update_count), cursor_before)
                self.assertEqual(random.getstate(), python_rng_before)
                numpy_rng_after = np.random.get_state()
                self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
                np.testing.assert_array_equal(numpy_rng_after[1], numpy_rng_before[1])
                self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng_before))

            decoder_failure = DecoderFailure("arbitrary decoder failure")
            with patch("research.levels.v4_trainer.torch.load", side_effect=decoder_failure):
                with self.assertRaisesRegex(TrainerError, "checkpoint_missing_or_invalid") as raised:
                    target.load_checkpoint(tmp)
            self.assertIs(raised.exception.__cause__, decoder_failure)
            self.assertEqual(str(raised.exception), "checkpoint_missing_or_invalid")
            assert_unchanged()

            abort = DecoderAbort("process-control exception")
            with patch("research.levels.v4_trainer.torch.load", side_effect=abort):
                with self.assertRaises(DecoderAbort) as raised_abort:
                    target.load_checkpoint(tmp)
            self.assertIs(raised_abort.exception, abort)
            assert_unchanged()

    def test_malformed_checkpoint_pointer_uses_read_boundary_without_state_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = trainer()
            before = weights(target.model)
            rng_before = torch.get_rng_state().clone()
            Path(tmp, "CURRENT").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(TrainerError, "checkpoint_missing_or_invalid") as raised:
                target.load_checkpoint(tmp)
            self.assertIsInstance(raised.exception.__cause__, json.JSONDecodeError)
            self.assertEqual(target.cursor, 0)
            self.assertEqual(target.update_count, 0)
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            for key, value in before.items():
                self.assertTrue(torch.equal(value, target.model.state_dict()[key]), key)

    def test_checkpoint_ignores_orphan_generation_and_missing_pointer_preserves_target(self):
        source = trainer()
        source.fit_step()
        with tempfile.TemporaryDirectory() as tmp:
            source.save_checkpoint(tmp)
            current_before = (Path(tmp) / "CURRENT").read_bytes()
            orphan = "f" * 32
            Path(tmp, f"model-{orphan}.safetensors.tmp").write_bytes(b"partial model")
            Path(tmp, f"state-{orphan}.pt.tmp").write_bytes(b"partial metadata")
            target = trainer()
            target.load_checkpoint(tmp)
            self.assertEqual(target.update_count, 1)
            self.assertEqual((Path(tmp) / "CURRENT").read_bytes(), current_before)

            pointer = json.loads(current_before)
            pointer["state"] = f"state-{'e' * 32}.pt"
            (Path(tmp) / "CURRENT").write_text(json.dumps(pointer))
            untouched = trainer(ident=identity("target"))
            before = weights(untouched.model)
            before_rng = torch.get_rng_state().clone()
            with self.assertRaises(TrainerError):
                untouched.load_checkpoint(tmp)
            self.assertEqual(untouched.update_count, 0)
            self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
            for key, tensor in before.items():
                self.assertTrue(torch.equal(tensor, untouched.model.state_dict()[key]))

    def test_prefetch_ring_ownership_and_exception_cleanup_are_bounded(self):
        ring = BoundedPrefetcher(capacity=1, device="cpu")
        slot, payload = ring.transfer({"x": torch.tensor([1.0])})
        self.assertTrue(torch.equal(ring.consume(slot)["x"], payload["x"]))
        with self.assertRaisesRegex(TrainerError, "still_owned"):
            ring.transfer({"x": torch.tensor([2.0])})
        ring.release(slot)
        slot, _ = ring.transfer({"x": torch.tensor([3.0])})
        ring.close()
        with self.assertRaisesRegex(TrainerError, "closed"):
            ring.transfer({"x": torch.tensor([4.0])})

        interrupted = BoundedPrefetcher(capacity=1, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
            with interrupted:
                interrupted.transfer({"x": torch.tensor([5.0])})
                raise RuntimeError("fixture interruption")
        with self.assertRaisesRegex(TrainerError, "closed"):
            interrupted.transfer({"x": torch.tensor([6.0])})


if __name__ == "__main__":
    unittest.main()
