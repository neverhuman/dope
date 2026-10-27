"""Offline contract tests; DOPE_ENCODER_WEIGHTS requires the released checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load, save

from research.levels.encoder import (
    CANONICAL_GATE_KEYS_14, DOPE_ACTIVE_ENCODER_PARAMETERS,
    DOPE_ACTIVE_ENCODER_TENSORS, DopeEncoder629,
    PINNED_MULTITASK_SAFETENSORS_SHA256,
)
from research.levels.multilevel_model import MultiLevelLandmarkModel

BLOCKS = ((0, 256), (256, 384), (384, 512), (512, 526), (526, 549), (549, 565), (565, 629))


def micro_features(model, vec, columns):
    # Micro python/hpc/captured_router.py::_capture_stage1a_gpu_graph,
    # also used by its uncaptured Stage 1A. Intentionally compute each row.
    result = []
    for v, c in zip(vec, columns):
        routed = model.router(v, level="high")
        gates = torch.stack([routed["gates"][k] for k in sorted(routed["gates"])])
        col_gates = model.target_net.compute_column_gates(routed["task_latent"], col_landmarks=c)
        result.append(torch.nan_to_num(torch.cat([
            v, routed["task_latent"], routed["dist_latent"], gates,
            routed["landmark_pred"], routed["bb_pred"], col_gates,
        ]), nan=0.0, posinf=0.0, neginf=0.0))
    return torch.stack(result)


class Composite(torch.nn.Module):
    """Representative frozen projection/ranker; downstream weights are separate pins."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projection = torch.nn.Linear(629, 48)
        self.ranker = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(48, 3))

    def forward(self, vec, columns):
        return self.ranker(self.projection(self.encoder(vec, columns)))


class DopeEncoder629Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(629)
        root = Path("target/encoder-629")
        root.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=root)
        cls.path = Path(cls.tmp.name) / "model.safetensors"
        cls.full = MultiLevelLandmarkModel().eval()
        # Exercise gates that initialize to zero in a fresh model.
        with torch.no_grad():
            for p in cls.full.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        cls.raw = save(cls.full.state_dict())
        cls.path.write_bytes(cls.raw)
        cls.digest = hashlib.sha256(cls.raw).hexdigest()
        cls.encoder = DopeEncoder629.from_pretrained(cls.path, expected_sha256=cls.digest)
        cls.evidence = {"torch": torch.__version__, "performance": "not measured"}

    @classmethod
    def tearDownClass(cls):
        Path("target/encoder-629/verification.json").write_text(json.dumps(cls.evidence, indent=2) + "\n")
        cls.tmp.cleanup()

    def inputs(self, batch):
        vec = torch.randn(batch, 256)
        # Micro copies up to 64 columns and the first 17 of 24 statistics;
        # all remaining column rows are zero, not neutral output gates.
        columns = torch.zeros(batch, 64, 17)
        columns[:, :18] = torch.randn(batch, 18, 24)[:, :, :17]
        return vec, columns

    def assert_blocks(self, got, expected):
        diffs = []
        for start, end in BLOCKS:
            torch.testing.assert_close(got[:, start:end], expected[:, start:end], rtol=1e-5, atol=1e-6)
            diffs.append(float((got[:, start:end] - expected[:, start:end]).abs().max()))
        return diffs

    def test_inventory_and_micro_parity(self):
        enc = self.encoder
        self.assertEqual(len(enc.state_dict()), DOPE_ACTIVE_ENCODER_TENSORS)
        self.assertEqual(sum(p.numel() for p in enc.parameters()), DOPE_ACTIVE_ENCODER_PARAMETERS)
        self.assertFalse(enc.training)
        self.assertTrue(all(not p.requires_grad for p in enc.parameters()))
        self.assertEqual(tuple(sorted(CANONICAL_GATE_KEYS_14)), CANONICAL_GATE_KEYS_14)
        with torch.no_grad():
            for batch in (1, 4):
                vec, columns = self.inputs(batch)
                got = enc(vec, columns)
                self.assertEqual(got.shape, (batch, 629))
                self.assert_blocks(got, micro_features(self.full, vec, columns))

    def test_released_weights(self):
        path = Path(os.environ.get("DOPE_ENCODER_WEIGHTS", "target/encoder-629/dope_mlc_v3_multitask_best.safetensors"))
        if not path.is_file() and "DOPE_ENCODER_WEIGHTS" not in os.environ:
            self.skipTest("release checkpoint absent; set DOPE_ENCODER_WEIGHTS to require it")
        self.assertEqual(path.stat().st_size, 1_781_800)
        enc = DopeEncoder629.from_pretrained(path)  # canonical digest is mandatory by default
        full = MultiLevelLandmarkModel().eval()
        state = load(path.read_bytes())
        full.load_state_dict(state, strict=True)
        evidence = {}
        with torch.no_grad():
            for batch in (1, 4):
                vec, columns = self.inputs(batch)
                expected = micro_features(full, vec, columns)
                evidence[str(batch)] = self.assert_blocks(enc(vec, columns), expected)
                # Persist canonical Micro reference for the pure Rust test/example.
                if batch == 4:
                    Path("target/encoder-629/micro-reference.json").write_text(json.dumps({
                        "vec": vec.tolist(), "columns": columns.tolist(), "expected": expected.tolist(),
                    }))
            for target in (vec, columns):
                target.flatten()[0:3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
            self.assert_blocks(enc(vec, columns), micro_features(full, vec, columns))
        self.evidence["released_micro_max_abs_by_block"] = evidence
        if torch.cuda.is_available():
            with torch.no_grad():
                gpu = DopeEncoder629.from_pretrained(path, device="cuda")
                gfull = full.cuda()
                v, c = self.inputs(4)
                v, c = v.cuda(), c.cuda()
                self.evidence["released_cuda_micro_max_abs_by_block"] = self.assert_blocks(
                    gpu(v, c), micro_features(gfull, v, c))
                self.check_capture(gpu, v, c, "released_cuda_graph")

    def test_malformed_inputs_are_private(self):
        vec, columns = self.inputs(1)
        bad = [
            (vec[0], columns), (vec[:, :255], columns), (torch.ones(1, 257), columns),
            (vec[:0], columns[:0]), (vec, columns[0]), (vec, columns[:, :63]),
            (vec, torch.ones(1, 64, 18)), (vec, columns.expand(4, -1, -1)),
            (vec.double(), columns), (vec, columns.half()), (vec.long(), columns),
            (vec, None), (None, columns), (vec.to_sparse(), columns),
            (vec, columns.to("meta")),
        ]
        if torch.cuda.is_available():
            bad += [(vec.cuda(), columns), (vec.cuda(), columns.cuda())]
        for v, c in bad:
            with self.subTest(case=len(str(type(v)))):
                with self.assertRaises(ValueError) as err:
                    self.encoder(v, c)
                self.assertNotIn("tensor(", str(err.exception))
        with torch.autocast("cpu", dtype=torch.bfloat16), self.assertRaisesRegex(ValueError, "autocast"):
            self.encoder(vec, columns)
        with self.assertRaisesRegex(ValueError, "parameters"):
            DopeEncoder629().double()(vec, columns)

    def test_checksum_precedes_parsing(self):
        with patch("safetensors.torch.load") as parse:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                DopeEncoder629.from_pretrained(self.path)
            parse.assert_not_called()
        for digest in (None, "", "x" * 64, "0" * 63):
            with self.assertRaisesRegex(ValueError, "hex digest"):
                DopeEncoder629.from_pretrained(self.path, expected_sha256=digest)
        with patch("safetensors.torch.load") as parse:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                DopeEncoder629.from_pretrained(self.path, expected_sha256="0" * 64)
            parse.assert_not_called()

    def test_checkpoint_schema_drift(self):
        state = load(self.raw)
        key = "target_net.fc1.weight"  # even a pruned tensor must have the full schema
        cases = [dict(state, private_column_name=torch.zeros(1))]
        for replacement in (torch.zeros(1), state[key].double(), torch.full_like(state[key], float("nan"))):
            cases.append(dict(state, **{key: replacement}))
        missing = dict(state)
        del missing[key]
        cases += [missing, {k: t for k, t in state.items() if k.startswith("router.")}]
        for sd in cases:
            raw = save(sd)
            p = Path(self.tmp.name) / "bad.safetensors"
            p.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "full FP32 model schema") as err:
                DopeEncoder629.from_pretrained(p, expected_sha256=hashlib.sha256(raw).hexdigest())
            self.assertNotIn("private_column_name", str(err.exception))
        raw = b"invalid safetensors"
        p.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "full FP32 model schema"):
            DopeEncoder629.from_pretrained(p, expected_sha256=hashlib.sha256(raw).hexdigest())

    def test_cpu_export_and_composition(self):
        path = Path(os.environ.get("DOPE_ENCODER_WEIGHTS", "target/encoder-629/dope_mlc_v3_multitask_best.safetensors"))
        encoder = DopeEncoder629.from_pretrained(path) if path.is_file() else self.encoder
        composite = Composite(encoder).requires_grad_(False).eval()
        vec, columns = self.inputs(4)
        batch_dim = torch.export.Dim("batch", min=1, max=8)
        with torch.no_grad():
            exported = torch.export.export(composite, (vec, columns), dynamic_shapes=(
                {0: batch_dim}, {0: batch_dim},
            ))
            output = Path("target/encoder-629/composite.pt2")
            torch.export.save(exported, output)
            restored = torch.export.load(output).module()
            diffs = {}
            for batch in (1, 4):
                v, c = self.inputs(batch)
                torch.testing.assert_close(restored(v, c), composite(v, c), rtol=1e-5, atol=1e-6)
                diffs[str(batch)] = float((restored(v, c) - composite(v, c)).abs().max())
            self.evidence["cpu_export_composition_max_abs"] = diffs

    def check_capture(self, encoder, v, c, label):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                encoder(v, c)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = encoder(v, c)
        diffs = []
        for _ in range(3):
            v.copy_(torch.randn_like(v))
            c.copy_(torch.randn_like(c))
            expected = encoder(v, c)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, rtol=1e-5, atol=1e-6)
            diffs.append(float((captured - expected).abs().max()))
        self.evidence[label] = diffs

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_capture(self):
        encoder = DopeEncoder629.from_pretrained(self.path, expected_sha256=self.digest, device="cuda")
        v, c = self.inputs(1)
        with torch.no_grad():
            self.check_capture(Composite(encoder).cuda().eval(), v.cuda(), c.cuda(), "cuda_composite_graph")

    def test_package_is_lightweight(self):
        result = subprocess.run([sys.executable, "-B", "-c", (
            "import sys; import research.levels; import research.levels.budget_planner; "
            "assert 'torch' not in sys.modules"
        )], check=False, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
