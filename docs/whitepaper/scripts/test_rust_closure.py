"""The architecture macros read the fitter's declared include closure, so a module split cannot break or fake them."""
import tempfile
import unittest
from pathlib import Path

import paper_emit
import rust_closure

PROFILE = '        "features12_steps2048" => (12, 16, 2048),\n'
OPTIMIZER = '        .build(&store, 2e-3)\n'
PRODUCTION = '    Ok((12, feature_count.min(12).clamp(4, 12), 96))\n'
LOCK = '{"split": "official_test_grouped_training_80_20"}\n'


def tree(root: Path, files: dict) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


OLD = {
    "rust/compiler/target_fitting.rs": 'include!("target_fitting/part_01.rs");\ninclude!("target_fitting/part_02.rs");\n'
                                       'include!("target_fitting/part_03.rs");\n',
    "rust/compiler/target_fitting/part_01.rs": "// linear\n",
    "rust/compiler/target_fitting/part_02.rs": PROFILE + OPTIMIZER + PRODUCTION,
    "rust/compiler/target_fitting/part_03.rs": "// tail\n",
    "research/benchmark/results/s3-data.lock.json": LOCK,
}
SPLIT = {
    "rust/compiler/target_fitting.rs": 'include!("target_fitting/part_01.rs");\ninclude!("target_fitting/research_training.rs");\n'
                                       'include!("target_fitting/compact_neural_training.rs");\ninclude!("target_fitting/part_03.rs");\n',
    "rust/compiler/target_fitting/part_01.rs": "// linear\n",
    "rust/compiler/target_fitting/research_training.rs": PROFILE + OPTIMIZER + PRODUCTION,
    "rust/compiler/target_fitting/compact_neural_training.rs": OPTIMIZER,
    "rust/compiler/target_fitting/part_03.rs": "// tail\n",
    "research/benchmark/results/s3-data.lock.json": LOCK,
}


def macros(root: Path) -> list[str]:
    return paper_emit._architecture_commands(lambda name, value: f"{name}={value}", root=root)


class ArchitectureSourceClosureTests(unittest.TestCase):
    def test_old_and_split_layouts_give_the_same_eight_macros(self):
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as split:
            before = macros(tree(Path(old), OLD))
            after = macros(tree(Path(split), SPLIT))
        self.assertEqual(before, after)
        self.assertEqual(before, ["ArchSplitMajor=80", "ArchSplitMinor=20", "ArchWidth=16", "ArchSteps=2{,}048",
                                  "ArchLr=0.002", "ProductionWidthLo=4", "ProductionWidthHi=12", "ProductionSteps=96"])

    def test_the_repository_closure_gives_the_published_constants(self):
        values = dict(item.split("=", 1) for item in paper_emit._architecture_commands(lambda n, v: f"{n}={v}"))
        self.assertEqual((values["ArchWidth"], values["ArchSteps"], values["ArchLr"]), ("16", "2{,}048", "0.002"))
        self.assertEqual((values["ProductionWidthLo"], values["ProductionWidthHi"], values["ProductionSteps"]),
                         ("4", "12", "96"))

    def test_a_missing_declared_file_cannot_default(self):
        files = dict(SPLIT)
        del files["rust/compiler/target_fitting/research_training.rs"]
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "missing Rust source in the include closure"):
                macros(tree(Path(root), files))

    def test_a_missing_or_conflicting_constant_raises(self):
        missing = dict(SPLIT, **{"rust/compiler/target_fitting/research_training.rs": PROFILE + PRODUCTION,
                                 "rust/compiler/target_fitting/compact_neural_training.rs": "// none\n"})
        conflicting = dict(SPLIT, **{"rust/compiler/target_fitting/compact_neural_training.rs": "        .build(&store, 1e-3)\n"})
        for files, message in ((missing, "is missing"), (conflicting, "is ambiguous")):
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaisesRegex(ValueError, message):
                    macros(tree(Path(root), files))

    def test_an_include_outside_the_rust_tree_or_a_symlink_is_refused(self):
        escaping = dict(SPLIT, **{"rust/compiler/target_fitting.rs": 'include!("../../research/benchmark/results/s3-data.lock.json");\n'})
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "escapes the rust/ tree"):
                macros(tree(Path(root), escaping))
        with tempfile.TemporaryDirectory() as root:
            base = tree(Path(root), SPLIT)
            target = base / "rust/compiler/target_fitting/research_training.rs"
            real = base / "elsewhere.rs"
            real.write_text(target.read_text())
            target.unlink()
            target.symlink_to(real)
            with self.assertRaisesRegex(ValueError, "missing Rust source"):
                macros(base)

    def test_the_closure_lists_every_declared_file_in_order(self):
        with tempfile.TemporaryDirectory() as root:
            closure = rust_closure.include_closure(tree(Path(root), SPLIT), "rust/compiler/target_fitting.rs")
        self.assertEqual([name for name, _ in closure], [
            "rust/compiler/target_fitting.rs", "rust/compiler/target_fitting/part_01.rs",
            "rust/compiler/target_fitting/research_training.rs", "rust/compiler/target_fitting/compact_neural_training.rs",
            "rust/compiler/target_fitting/part_03.rs"])


if __name__ == "__main__":
    unittest.main()
