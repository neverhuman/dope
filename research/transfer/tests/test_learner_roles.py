"""Version-2 learner roles. v1 behavior-contract bytes are not rewritten."""

import json
from pathlib import Path
import unittest

from research.transfer.learner_roles import assign_learner_roles, select_development_learner


ROOT = Path(__file__).resolve().parents[3]


def roles(**overrides):
    record = {
        "format": "dope-learner-roles",
        "version": 2,
        "selection_families": ["linear", "tree"],
        "evaluation_required_families": ["linear", "tree", "kumo"],
        "held_out_families": ["kumo"],
    }
    record.update(overrides)
    return record


class LearnerRoleTests(unittest.TestCase):
    def test_schema_identity_is_version_2(self):
        schema = json.loads((ROOT / "schemas/learner-roles-v2.schema.json").read_text())
        self.assertEqual(schema["$id"], "urn:dope:learner-roles:v2")
        self.assertEqual(schema["properties"]["version"]["const"], 2)
        self.assertEqual(schema["properties"]["format"]["const"], "dope-learner-roles")
        self.assertEqual(
            schema["required"],
            ["format", "version", "selection_families", "evaluation_required_families", "held_out_families"],
        )
        self.assertIs(schema["additionalProperties"], False)

    def test_held_out_family_cannot_win_selection(self):
        candidates = [
            {"family_id": "kumo", "learner_id": "kumo-best", "development_loss": 0.01},
            {"family_id": "tree", "learner_id": "tree-a", "development_loss": 0.4},
            {"family_id": "linear", "learner_id": "linear-a", "development_loss": 0.2},
            {"family_id": "kumo", "learner_id": "kumo-unscored"},
        ]
        chosen = select_development_learner(candidates, roles())
        self.assertEqual(chosen["selected_learner"], "linear-a")
        self.assertEqual(chosen["held_out_families"], ("kumo",))
        self.assertIs(chosen["held_out_excluded_from_selection"], True)
        self.assertIs(chosen["protocol_frozen"], False)
        self.assertIs(chosen["confirmation_allowed"], False)
        self.assertIs(chosen["native_certification"], False)
        self.assertIs(chosen["qualification"], False)
        self.assertEqual(chosen["campaign"], "unmeasured")
        self.assertEqual(chosen["selection_not_evaluated"], ())

    def test_role_rules_refuse_legacy_overlap_and_freeze(self):
        assigned = assign_learner_roles(roles(selection_families=["linear"], evaluation_required_families=["kumo"]))
        self.assertEqual(assigned["selection_not_evaluated"], ("linear",))
        with self.assertRaises(ValueError) as raised:
            assign_learner_roles({"required_families": ["linear"], "held_out_families": ["kumo"]})
        self.assertEqual(str(raised.exception), "legacy_role_identity")
        with self.assertRaises(ValueError) as raised:
            assign_learner_roles(roles(version=1))
        self.assertEqual(str(raised.exception), "legacy_role_identity")
        with self.assertRaises(ValueError) as raised:
            assign_learner_roles(roles(selection_families=["kumo"], evaluation_required_families=["kumo"]))
        self.assertEqual(str(raised.exception), "held_out_selected")
        with self.assertRaises(ValueError) as raised:
            assign_learner_roles(roles(evaluation_required_families=["linear", "tree"], held_out_families=["kumo"]))
        self.assertEqual(str(raised.exception), "held_out_not_evaluated")
        with self.assertRaises(ValueError) as raised:
            assign_learner_roles({**roles(), "protocol_frozen": True})
        self.assertEqual(str(raised.exception), "invalid_learner_roles")


if __name__ == "__main__":
    unittest.main()
