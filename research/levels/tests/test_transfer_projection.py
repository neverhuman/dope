"""B00/B03 token instrument. No checkpoint, training, or panel score."""
import math
import unittest

from research.levels.transfer_projection import (
    AFFINE_PARAMETER_COUNT,
    DIRECT_SLOTS,
    DOPE_WIDTH,
    TOKEN_COUNT,
    TOKEN_WIDTH,
    SvdSlot,
    TransferProjectionError,
    arm,
    b03_tokens,
    fold_slots,
    initialization_reproduces_control,
    project_control_slots,
    projector_normalizer_and_token_layout_sha256,
)


def _slot(edits, *, mean=None, std=None, active_at=()):
    mean_values = [0.0] * DOPE_WIDTH if mean is None else list(mean)
    std_values = [1.0] * DOPE_WIDTH if std is None else list(std)
    active = [0.0] * DOPE_WIDTH
    for index in active_at:
        active[index] = 1.0
    projection = [0.0] * (DOPE_WIDTH * TOKEN_WIDTH)
    for (row, column), value in edits.items():
        projection[row * TOKEN_WIDTH + column] = value
    return SvdSlot(tuple(mean_values), tuple(std_values), tuple(active), tuple(projection))


def _features(values=()):
    features = [0.0] * DOPE_WIDTH
    for index, value in dict(values).items():
        features[index] = value
    return features


def _linear_slots():
    mean = [0.0] * DOPE_WIDTH
    mean[0] = 0.25
    std = [1.0] * DOPE_WIDTH
    std[2] = 2.0
    return {
        2: _slot({(0, 0): 0.5}, mean=mean, active_at=(0,)),
        4: _slot({(1, 1): 0.25}, active_at=(1,)),
        5: _slot({(2, 0): 0.5}, std=std, active_at=(2,)),
    }


def _control_matrix(direct):
    tokens = [[0.0] * TOKEN_WIDTH for _ in range(TOKEN_COUNT)]
    tokens[0][0] = 0.5
    tokens[1][3] = -0.5
    tokens[7][4] = 0.25
    tokens[9][0] = 0.125
    for slot_id, row in direct.items():
        tokens[slot_id] = list(row)
    return tokens


class TransferProjectionTests(unittest.TestCase):
    def test_folded_init_reproduces_the_control_inside_the_linear_region(self):
        slots = _linear_slots()
        features = _features({0: 0.75, 1: -0.25, 2: 1.0})
        self.assertTrue(initialization_reproduces_control(features, slots))
        control = project_control_slots(features, slots)
        self.assertEqual(control[2][0], 0.25)
        self.assertEqual(control[4][1], -0.0625)
        self.assertEqual(control[5][0], 0.25)
        self.assertEqual(b03_tokens(features, slots, _control_matrix(control)), _control_matrix(control))
        self.assertEqual(fold_slots(slots).parameter_count, AFFINE_PARAMETER_COUNT)
        self.assertEqual(AFFINE_PARAMETER_COUNT, 60_480)

    def test_saturation_is_not_reported_as_a_reproduction(self):
        slots = {
            2: _slot({(0, 0): 0.01}, std=[0.01] + [1.0] * (DOPE_WIDTH - 1), active_at=(0,)),
            4: _slot({}, active_at=()),
            5: _slot({}, active_at=()),
        }
        features = _features({0: 1.0})
        self.assertFalse(initialization_reproduces_control(features, slots))
        control = project_control_slots(features, slots)
        folded = b03_tokens(features, slots, _control_matrix(control))
        self.assertEqual(control[2][0], 0.08)
        self.assertEqual(folded[2][0], 1.0)
        self.assertEqual(folded[0][0], 0.5)
        self.assertEqual(folded[1][3], -0.5)
        self.assertEqual(folded[7][4], 0.25)
        self.assertEqual(folded[9][0], 0.125)
        self.assertNotEqual(folded[9][0], folded[2][0] * folded[0][0])
        for slot_id in (3, 6, 8):
            self.assertEqual(folded[slot_id], [0.0] * TOKEN_WIDTH)

    def test_nonfinite_feature_uses_the_same_zero_substitution(self):
        slots = {
            2: _slot({(0, 0): 1.0}, active_at=(0,)),
            4: _slot({}, active_at=()),
            5: _slot({}, active_at=()),
        }
        features = _features({0: math.nan})
        self.assertTrue(initialization_reproduces_control(features, slots))
        self.assertEqual(project_control_slots(features, slots)[2][0], 0.0)

    def test_fixed_channels_and_one_field_commitment(self):
        slots = _linear_slots()
        original = projector_normalizer_and_token_layout_sha256(slots)
        self.assertEqual(original, projector_normalizer_and_token_layout_sha256(slots))
        self.assertRegex(original, r"^[0-9a-f]{64}$")
        edited = dict(slots)
        edited[4] = _slot({(1, 1): 0.5}, active_at=(1,))
        self.assertNotEqual(original, projector_normalizer_and_token_layout_sha256(edited))

    def test_arms_have_one_intervention_and_do_not_train_or_score(self):
        interventions = set()
        for arm_id in ("B00", "B01", "B02", "B03"):
            record = arm(arm_id)
            self.assertTrue(record.intervention)
            self.assertFalse(record.trains)
            self.assertFalse(record.reports_downstream_decisions)
            self.assertFalse(record.changes_encoder_weights)
            interventions.add(record.intervention)
        self.assertEqual(len(interventions), 4)
        self.assertTrue(arm("B00").constructs_tokens)
        self.assertTrue(arm("B03").constructs_tokens)
        self.assertFalse(arm("B01").constructs_tokens)
        self.assertFalse(arm("B02").constructs_tokens)
        with self.assertRaises(TransferProjectionError) as raised:
            arm("B04")
        self.assertEqual(str(raised.exception), "transfer_projection_arm")

    def test_refusals_do_not_echo_source_values(self):
        mean = [999_999.0] + [0.0] * (DOPE_WIDTH - 1)
        std = [-12_345.678] + [1.0] * (DOPE_WIDTH - 1)
        with self.assertRaises(TransferProjectionError) as raised:
            _slot({}, mean=mean, std=std, active_at=(0,)).checked()
        self.assertEqual(str(raised.exception), "transfer_projection_slot")
        self.assertNotIn("999999", str(raised.exception))
        self.assertNotIn("12345", str(raised.exception))
        slots = _linear_slots()
        control = _control_matrix(project_control_slots(_features(), slots))
        control[3][0] = 4.0
        with self.assertRaises(TransferProjectionError) as raised:
            b03_tokens(_features(), slots, control)
        self.assertEqual(str(raised.exception), "transfer_projection_mask")
        with self.assertRaises(TransferProjectionError) as raised:
            project_control_slots([0.0], slots)
        self.assertEqual(str(raised.exception), "transfer_projection_shape")


if __name__ == "__main__":
    unittest.main()
