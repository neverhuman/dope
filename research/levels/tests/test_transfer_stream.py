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
        apply_update(model, features, target, device=resolve_device("cpu"), lr=0.1)
        self.assertFalse(torch.equal(model.weight.detach().cpu(), before))
        if torch.cuda.is_available():
            gpu = nn.Linear(2, 1)
            with torch.no_grad():
                gpu.weight.zero_()
                gpu.bias.zero_()
            apply_update(gpu, features, target, device=resolve_device("cuda"), lr=0.1)
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

    def test_interrupted_replay_finishes_the_same_update(self):
        features, target = _rows()

        def train(stop, payload=None):
            model = nn.Linear(2, 1)
            with torch.no_grad():
                model.weight.zero_()
                model.bias.zero_()
            if payload is None:
                ledger = ExposureLedger(4)
                start = 0
            else:
                ledger = restore(model, payload)
                start = payload["step"]
            for index in range(start, stop):
                apply_update(
                    model,
                    features[index:index + 1],
                    target[index:index + 1],
                    device=torch.device("cpu"),
                    lr=0.1,
                )
                ledger.commit(index)
            return model, capture(model, ledger, stop)

        finished, full = train(4)
        _, midway = train(2)
        self.assertEqual(midway["step"], 2)
        self.assertEqual(len(midway["committed"]), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.json"
            write_state(path, midway)
            self.assertFalse(Path(str(path) + ".tmp").exists())
            loaded = read_state(path)
        resumed, replay = train(4, loaded)
        self.assertTrue(torch.equal(resumed.weight, finished.weight))
        self.assertTrue(torch.equal(resumed.bias, finished.bias))
        self.assertEqual(replay["committed"], full["committed"])
        self.assertTrue(restore(nn.Linear(2, 1), replay).complete())
        with self.assertRaises(StreamError) as caught:
            restore(nn.Linear(2, 1), replay).commit(1)
        self.assertEqual(str(caught.exception), "stream_coverage")
        self.assertNotIn("1", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
