import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.benchmark import adapters
from research.benchmark.score import sha256


ROOT = Path(__file__).resolve().parents[3]


class ChowLiuTests(unittest.TestCase):
    def test_artifact_sampling_preserves_strong_pair_dependence(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            values = np.array([[bit, bit, bit] for bit in (0.0, 1.0) for _ in range(100)])
            np.savetxt(root / "train.csv", values, delimiter=",")
            artifact = root / "artifact"
            files = adapters.fit("Chow-Liu", root / "train.csv", {"task": "binary"},
                                 {"bins": 8, "laplace_alpha": 1.0}, 11, artifact)
            self.assertEqual(files, ["model.json"])
            adapters.sample("Chow-Liu", artifact, 3000, 101, root / "sample.csv")
            adapters.sample("Chow-Liu", artifact, 3000, 101, root / "repeat.csv")
            self.assertEqual(sha256(root / "sample.csv"), sha256(root / "repeat.csv"))
            sample = np.loadtxt(root / "sample.csv", delimiter=",")
            self.assertGreater(np.corrcoef(sample[:, 0], sample[:, 1])[0, 1], 0.7)


if __name__ == "__main__":
    unittest.main()
