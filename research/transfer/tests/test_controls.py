"""Matched teacher, predictor, coreset, and distillation controls."""

import json
from pathlib import Path
import unittest

from research.transfer.adapters.generators.distillation import TabularCondensation
from research.transfer.controls import (
    CONTROLS, ByteMatchedCoreset, ControlError, DirectPredictorControl, MatchedTeacherFactorial, Teacher,
)

ROOT = Path(__file__).resolve().parents[3]
PLAN = json.loads((ROOT / "research/transfer/ablations/matched_teacher.json").read_text())
DRAWS = {"independent": ((0, 0), (1, 0), (2, 0), (3, 1)), "learned": ((1, 0), (1, 1), (2, 0), (4, 1))}


def factorial():
    teacher = Teacher(PLAN["teacher"]["form"], PLAN["teacher"]["parameters"],
                      PLAN["teacher"]["noise_scale"], PLAN["precision"])
    return MatchedTeacherFactorial(teacher, PLAN["search_budget"], PLAN["byte_budget"], PLAN["controls"])


def cell(factor, control, x_mode="independent", form="nonlinear"):
    draws = tuple(item[0] for item in DRAWS[x_mode])
    noises = tuple(item[1] for item in DRAWS[x_mode])
    return factor.cell(form, x_mode, control, draws, noises), draws, noises


class ControlTests(unittest.TestCase):
    def test_plan_names_every_control_and_cells_share_one_budget(self):
        self.assertEqual(set(PLAN["controls"]), CONTROLS)
        factor = factorial()
        cells = [cell(factor, name)[0] for name in PLAN["controls"]]
        self.assertEqual({item["search_budget"] for item in cells}, {PLAN["search_budget"]})
        self.assertEqual({item["byte_budget"] for item in cells}, {PLAN["byte_budget"]})
        self.assertTrue(all(item["base_rows"] == 4 for item in cells))

    def test_x_contrast_keeps_the_teacher_and_changes_the_draws(self):
        factor = factorial()
        independent = cell(factor, "real_n", "independent")[0]
        learned = cell(factor, "real_n", "learned")[0]
        self.assertEqual(independent["parameter_sha256"], learned["parameter_sha256"])
        self.assertEqual(independent["noise_sha256"], learned["noise_sha256"])
        self.assertNotEqual(independent["draw_sha256"], learned["draw_sha256"])
        linear = cell(factor, "real_n", form="linear")[0]
        self.assertNotEqual(linear["parameter_sha256"], independent["parameter_sha256"])
        self.assertEqual(linear["noise_sha256"], independent["noise_sha256"])

    def test_noise_label_quantization_and_bootstrap_controls(self):
        factor = factorial()
        base = cell(factor, "real_n")[0]
        quiet = cell(factor, "no_noise")[0]
        self.assertNotEqual(quiet["noise_sha256"], base["noise_sha256"])
        shuffled = cell(factor, "shuffled_label")[0]
        self.assertEqual(shuffled["support_sha256"], base["support_sha256"])
        self.assertNotEqual(shuffled["label_sha256"], base["label_sha256"])
        self.assertTrue(shuffled["retained_real_rows"])
        post = cell(factor, "postquantization")[0]
        pre = cell(factor, "prequantization")[0]
        self.assertNotEqual(post["value_sha256"], pre["value_sha256"])
        bootstrap = cell(factor, "real_bootstrap_4n")[0]
        self.assertEqual(bootstrap["stored_rows"], 16)
        self.assertTrue(bootstrap["retained_real_rows"])

    def test_direct_predictor_matches_the_real_x_diagnostic(self):
        factor = factorial()
        _, draws, noises = cell(factor, "direct_predictor")
        teacher = factor.teacher_for("direct_predictor", "nonlinear")
        predictor = DirectPredictorControl(teacher, draws, noises)
        self.assertEqual(predictor.output_sha256, predictor.empirical_diagnostic(draws, noises))
        self.assertFalse(predictor.retained_real_rows)
        self.assertGreater(predictor.charged_bytes, 0)

    def test_coreset_keeps_real_rows_inside_the_byte_budget(self):
        draws, noises = (0, 1, 2, 3), (0, 0, 0, 1)
        coreset = ByteMatchedCoreset(draws, noises, 16)
        self.assertEqual(len(coreset.rows), 2)
        self.assertEqual(coreset.charged_bytes, 16)
        self.assertTrue(coreset.retained_real_rows)

    def test_distillation_charges_encoder_decoder_init_and_recipe(self):
        dependencies = {"encoder": 8, "decoder": 8, "init": 8, "recipe": 8}
        condensed = TabularCondensation((0, 1), dependencies, PLAN["byte_budget"], PLAN["search_budget"])
        self.assertEqual(condensed.charged_bytes, 48)
        self.assertFalse(condensed.retained_real_rows)
        self.assertEqual(condensed.adaptations, ("synthetic_prototype",))
        missing = dict(dependencies)
        del missing["recipe"]
        with self.assertRaises(ControlError) as raised:
            TabularCondensation((0, 1), missing, PLAN["byte_budget"], PLAN["search_budget"])
        self.assertEqual(str(raised.exception), "uncharged_dependency")
        self.assertNotIn("99991", str(raised.exception))

    def test_parameter_mutation_cannot_leave_the_parameter_hash_stale(self):
        teacher = Teacher("nonlinear", {"quadratic": 1, "linear": 1, "bias": 1}, 0, "float64")
        before_hash = teacher.parameter_sha256
        before_value = teacher.value(2, 0)
        with self.assertRaises(TypeError):
            teacher.parameters["bias"] = 99991
        self.assertEqual(teacher.parameter_sha256, before_hash)
        self.assertEqual(teacher.value(2, 0), before_value)

    def test_shuffled_constant_labels_preserve_the_label_multiset(self):
        teacher = Teacher("linear", {"quadratic": 0, "linear": 0, "bias": 5}, 0, "float64")
        factor = MatchedTeacherFactorial(
            teacher, PLAN["search_budget"], PLAN["byte_budget"], PLAN["controls"],
        )
        draws, noises = (0, 1, 2), (0, 0, 0)
        base = factor.cell("linear", "independent", "real_n", draws, noises)
        shuffled = factor.cell("linear", "independent", "shuffled_label", draws, noises)
        self.assertEqual(shuffled["label_sha256"], base["label_sha256"])
        self.assertEqual(shuffled["support_sha256"], base["support_sha256"])

    def test_direct_predictor_rejects_a_draw_noise_length_mismatch(self):
        teacher = factorial().teacher_for("direct_predictor", "nonlinear")
        with self.assertRaises(ControlError) as raised:
            DirectPredictorControl(teacher, (99991, 4), (0,))
        self.assertEqual(str(raised.exception), "invalid_draws")
        self.assertNotIn("99991", str(raised.exception))
        predictor = DirectPredictorControl(teacher, (0, 1), (0, 1))
        with self.assertRaises(ControlError) as diagnostic:
            predictor.empirical_diagnostic((0, 1, 2), (0,))
        self.assertEqual(str(diagnostic.exception), "invalid_draws")

    def test_control_errors_do_not_echo_draw_values(self):
        factor = factorial()
        with self.assertRaises(ControlError) as raised:
            factor.cell("nonlinear", "independent", "real_n", (99991,), (0, 0))
        self.assertEqual(str(raised.exception), "invalid_draws")
        self.assertNotIn("99991", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
