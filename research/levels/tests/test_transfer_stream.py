"""Indexed stream, single-device updates, and exact resume. No campaign run."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from research.levels.transfer_stream import (
    SCHEMA,
    ExposureLedger,
    IndexedStream,
    StreamError,
    StreamRecord,
    accumulated_gradient,
    apply_update,
    capture,
    read_state,
    resolve_device,
    restore,
    support,
    write_state,
)


def _model() -> nn.Linear:
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -0.25]]))
        model.bias.copy_(torch.tensor([0.125]))
    return model


def _rows():
    features = torch.tensor([[1.0, 2.0], [0.0, -1.0], [3.0, 0.5], [-2.0, 1.0]])
    target = torch.tensor([[0.5], [-0.25], [1.0], [0.0]])
    return features, target


class TransferStreamTests(unittest.TestCase):
    def test_window_stays_bounded_as_the_manifest_grows(self):
        def open_stream(count):
            reads = {"n": 0}

            def read(index):
                reads["n"] += 1
                return StreamRecord(index, (2,), index)

            stream = IndexedStream(count, read, prefetch=2, batch=4)
            self.assertEqual(reads["n"], 0)
            self.assertEqual(stream.resident(), 0)
            first = next(iter(stream))
            self.assertLessEqual(stream.resident(), 2)
            self.assertLess(reads["n"], count)
            self.assertLessEqual(len(first), 4)
            self.assertTrue(all(record.shape == (2,) for record in first))
            consumed = len(first)
            for batch in stream:
                consumed += len(batch)
                self.assertLessEqual(stream.resident(), 2)
            self.assertEqual(consumed, count)
            self.assertEqual(reads["n"], count)
            return reads["n"]

        self.assertEqual(open_stream(40), 40)
        self.assertEqual(open_stream(400), 400)

    def test_batches_do_not_mix_shapes(self):
        shapes = ((2,), (2,), (2,), (4,), (4,))

        def read(index):
            return StreamRecord(index, shapes[index], None)

        batches = list(IndexedStream(len(shapes), read, prefetch=2, batch=8))
        self.assertEqual([len(batch) for batch in batches], [3, 2])
        self.assertEqual(batches[0][0].shape, (2,))
        self.assertEqual(batches[1][0].shape, (4,))

    def test_batched_and_unbatched_gradients_agree(self):
        features, target = _rows()
        batched = accumulated_gradient(_model(), features, target, microbatches=1)
        single = accumulated_gradient(_model(), features, target, microbatches=4)
        paired = accumulated_gradient(_model(), features, target, microbatches=2)
        for left, right, pairs in zip(batched, single, paired):
            self.assertTrue(torch.equal(left, right))
            self.assertTrue(torch.equal(left, pairs))

    def test_cpu_update_and_cuda_report_match_the_runtime(self):
        report = support()
        self.assertEqual(report["schema"], SCHEMA)
        self.assertTrue(report["cpu"])
        self.assertFalse(report["ddp"])
        self.assertEqual(report["cuda"], torch.cuda.is_available())
        with self.assertRaises(StreamError) as refused:
            resolve_device("secret-device")
        self.assertEqual(str(refused.exception), "stream_device")
        self.assertNotIn("secret-device", str(refused.exception))
        with patch("research.levels.transfer_stream.torch.cuda.is_available", return_value=False):
            self.assertFalse(support()["cuda"])
            with self.assertRaises(StreamError) as missing:
                resolve_device("cuda")
            self.assertEqual(str(missing.exception), "stream_device")
            self.assertNotIn("cuda", str(missing.exception))
        features = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([[1.0]])
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        before = model.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        apply_update(model, features, target, optimizer, device=resolve_device("cpu"))
        self.assertFalse(torch.equal(model.weight.detach().cpu(), before))
        quiet = nn.Linear(2, 1)
        with torch.no_grad():
            quiet.weight.zero_()
            quiet.bias.zero_()
        quiet_optimizer = torch.optim.AdamW(quiet.parameters(), lr=0.1)
        apply_update(
            quiet,
            torch.zeros(1, 2),
            torch.zeros(1, 1),
            quiet_optimizer,
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(quiet.weight.grad)
        self.assertEqual(int(torch.count_nonzero(quiet.weight.grad)), 0)
        if torch.cuda.is_available():
            gpu = nn.Linear(2, 1)
            with torch.no_grad():
                gpu.weight.zero_()
                gpu.bias.zero_()
            gpu_optimizer = torch.optim.AdamW(gpu.parameters(), lr=0.1)
            apply_update(gpu, features, target, gpu_optimizer, device=resolve_device("cuda"))
            self.assertEqual(gpu.weight.device.type, "cuda")
            self.assertFalse(torch.equal(gpu.weight.detach().cpu(), before))

    def test_coverage_refuses_a_duplicate_without_echo(self):
        ledger = ExposureLedger(4)
        for index in range(4):
            ledger.commit(index)
        self.assertTrue(ledger.complete())
        self.assertEqual(ledger.missing(), 0)
        with self.assertRaises(StreamError) as caught:
            ledger.commit(2)
        self.assertEqual(str(caught.exception), "stream_coverage")
        self.assertNotIn("2", str(caught.exception))
        with self.assertRaises(StreamError) as wide:
            IndexedStream(1, lambda index: StreamRecord(index, (1,), None), prefetch=9, batch=1)
        self.assertEqual(str(wide.exception), "stream_limit")
        self.assertNotIn("9", str(wide.exception))

    def test_prefetched_uncommitted_records_replay_from_start(self):
        count = 6
        reads = []

        def read(index):
            reads.append(index)
            return StreamRecord(index, (1,), None)

        stream = IndexedStream(count, read, prefetch=2, batch=1, start=0)
        first = next(iter(stream))
        self.assertEqual(first[0].index, 0)
        self.assertEqual(stream.resident(), 2)
        self.assertEqual(reads, [0, 1, 2])
        resumed_reads = []

        def read_again(index):
            resumed_reads.append(index)
            return StreamRecord(index, (1,), None)

        resumed = IndexedStream(count, read_again, prefetch=2, batch=1, start=1)
        indexes = [record.index for batch in resumed for record in batch]
        self.assertEqual(indexes, [1, 2, 3, 4, 5])
        self.assertEqual(resumed_reads[:2], [1, 2])
        ledger = ExposureLedger(count)
        ledger.commit(0)
        for index in indexes:
            ledger.commit(index)
        self.assertTrue(ledger.complete())
        with self.assertRaises(StreamError) as caught:
            IndexedStream(count, read, prefetch=2, batch=1, start=9)
        self.assertEqual(str(caught.exception), "stream_limit")
        self.assertNotIn("9", str(caught.exception))

    def test_adamw_replay_matches_and_a_failed_microbatch_keeps_the_checkpoint(self):
        features, target = _rows()

        def train(stop, payload=None):
            model = nn.Linear(2, 1)
            with torch.no_grad():
                model.weight.zero_()
                model.bias.zero_()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
            if payload is None:
                torch.manual_seed(7)
                ledger = ExposureLedger(4)
                start = 0
            else:
                ledger = restore(model, optimizer, payload)
                start = payload["step"]
            for index in range(start, stop):
                noise = torch.randn(1, 2)
                apply_update(
                    model,
                    features[index:index + 1] + noise,
                    target[index:index + 1],
                    optimizer,
                    device=torch.device("cpu"),
                )
                ledger.commit(index)
            return model, optimizer, capture(model, ledger, stop, optimizer)

        finished, finished_optimizer, full = train(4)
        _, _, midway = train(2)
        self.assertIn("optimizer", midway)
        self.assertIn("rng", midway)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.json"
            write_state(path, midway)
            self.assertFalse(Path(str(path) + ".part").exists())
            loaded = read_state(path)
            resumed, resumed_optimizer, replay = train(4, loaded)
            self.assertTrue(torch.equal(resumed.weight, finished.weight))
            self.assertTrue(torch.equal(resumed.bias, finished.bias))
            self.assertTrue(torch.equal(torch.get_rng_state(), _rng_from(full)))
            left_state = finished_optimizer.state_dict()["state"]
            right_state = resumed_optimizer.state_dict()["state"]
            self.assertEqual(set(left_state), set(right_state))
            for key, group in left_state.items():
                self.assertEqual(set(group), set(right_state[key]))
                for name, value in group.items():
                    self.assertTrue(torch.equal(value, right_state[key][name]))
            self.assertEqual(replay["committed"], full["committed"])
            checked = nn.Linear(2, 1)
            held = restore(checked, torch.optim.AdamW(checked.parameters(), lr=0.1), replay)
            self.assertTrue(held.complete())
            saved = path.read_bytes()
            failed = nn.Linear(1, 1)
            with torch.no_grad():
                failed.weight.fill_(1e20)
                failed.bias.zero_()
            failed_optimizer = torch.optim.AdamW(failed.parameters(), lr=0.1)
            kept = failed.weight.detach().clone()
            ledger = ExposureLedger(2, (0,))
            with self.assertRaises(StreamError) as caught:
                apply_update(
                    failed,
                    torch.tensor([[1e-20], [1.0]]),
                    torch.zeros(2, 1),
                    failed_optimizer,
                    device=torch.device("cpu"),
                    microbatches=2,
                )
            self.assertEqual(str(caught.exception), "stream_nonfinite")
            self.assertNotIn("inf", str(caught.exception))
            self.assertIsNone(failed.weight.grad)
            self.assertIsNone(failed.bias.grad)
            self.assertTrue(torch.equal(failed.weight, kept))
            self.assertEqual(ledger.committed(), (0,))
            self.assertEqual(path.read_bytes(), saved)


def _rng_from(payload):
    values = payload["rng"]["values"]
    return torch.tensor(values, dtype=torch.uint8)


if __name__ == "__main__":
    unittest.main()
