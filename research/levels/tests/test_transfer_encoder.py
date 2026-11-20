"""Fixtures for the versioned transfer-encoder candidates. No training run."""
import inspect
import unittest

import torch

from research.levels.contracts import ModelConfig
from research.levels.encoder import DOPE_FEATURES_629_DIM, DopeEncoder629
from research.levels.transfer_encoder import (
    GLOBAL_WIDTH,
    RESIDUAL_ALPHA,
    SCHEMA,
    ResidualControl,
    TransferEncoderError,
    candidate,
    encode,
    inventory,
    limits,
    realize,
    same_abi,
    task_slot,
)


class TransferEncoderTests(unittest.TestCase):
    def test_counts_and_limits_come_from_the_realized_graph(self):
        recorded = limits()
        config = ModelConfig().validate()
        self.assertEqual(recorded["schema"], SCHEMA)
        self.assertEqual(recorded["raw_table_width"], config.width)
        self.assertEqual(recorded["raw_table_heads"], config.heads)
        self.assertEqual(recorded["feature_blocks"], config.feature_blocks)
        self.assertEqual(recorded["inducing_blocks"], config.inducing_blocks)
        self.assertEqual(recorded["target_blocks"], config.target_blocks)
        self.assertEqual(recorded["max_features"], config.max_features)
        self.assertEqual(recorded["cell_cap"], config.cell_cap)
        self.assertEqual(recorded["full629_width"], DOPE_FEATURES_629_DIM)
        self.assertFalse(recorded["released_weights_loaded"])
        self.assertEqual(candidate("C").input_width, config.width)
        self.assertEqual(candidate("A").input_width, DOPE_FEATURES_629_DIM)
        self.assertEqual(candidate("B").input_width, DOPE_FEATURES_629_DIM)
        readout = realize("A")
        counted = inventory(readout)
        live = sum(parameter.numel() for parameter in readout.parameters())
        self.assertEqual(counted["parameter_count"], live)
        self.assertEqual(counted["transferred_from_released"], 0)
        self.assertEqual(counted["limits"]["released_architecture_parameters"], 353095)
        self.assertGreater(live, 0)

    def test_width_match_is_not_an_abi_alias(self):
        consumer = candidate("A")
        continued = candidate("B")
        self.assertFalse(same_abi(consumer.identity, continued.identity, 629, 629))
        self.assertFalse(same_abi("Dope512", consumer.identity, GLOBAL_WIDTH, GLOBAL_WIDTH))
        self.assertFalse(same_abi("DopeEncoder629", continued.identity, 629, 629))
        self.assertTrue(same_abi(consumer.identity, consumer.identity, 629, 629))
        self.assertNotEqual(consumer.graph, continued.graph)
        self.assertFalse(continued.loads_released_weights)
        self.assertTrue(continued.continues_released_graph)
        features = torch.zeros(1, DOPE_FEATURES_629_DIM)
        with self.assertRaises(TransferEncoderError) as caught:
            encode(
                realize("B"),
                "B",
                features,
                task_id="rank",
                claimed_identity="DopeEncoder629",
            )
        self.assertEqual(str(caught.exception), "transfer_encoder_alias")
        with self.assertRaises(TransferEncoderError) as denied:
            encode(
                realize("A"),
                "A",
                features,
                task_id="rank",
                claimed_identity="Dope512",
            )
        self.assertNotIn("Dope512", str(denied.exception))
        self.assertNotIn("512", str(denied.exception))

    def test_unsupported_shapes_refuse_without_echo(self):
        features = torch.zeros(1, DOPE_FEATURES_629_DIM)
        readout = realize("A")
        with self.assertRaises(TransferEncoderError) as caught:
            encode(readout, "A", features[:, :256], task_id="rank", claimed_identity=candidate("A").identity)
        self.assertEqual(str(caught.exception), "transfer_encoder_shape")
        self.assertNotIn("256", str(caught.exception))
        raw = torch.zeros(1, ModelConfig().width)
        with self.assertRaises(TransferEncoderError) as raw_caught:
            encode(
                realize("C"),
                "C",
                torch.zeros(1, DOPE_FEATURES_629_DIM),
                task_id="rank",
                claimed_identity=candidate("C").identity,
            )
        self.assertEqual(str(raw_caught.exception), "transfer_encoder_shape")
        active = features.clone()
        active[0, 0] = float("nan")
        with self.assertRaises(TransferEncoderError) as nonfinite:
            encode(readout, "A", active, task_id="rank", claimed_identity=candidate("A").identity)
        self.assertEqual(str(nonfinite.exception), "transfer_encoder_nonfinite")
        self.assertNotIn("nan", str(nonfinite.exception).lower())
        self.assertEqual(raw.shape[1], candidate("C").input_width)

    def test_feature_identity_survives_realization(self):
        readout = realize("C", column_tokens=True)
        identity = candidate("C").identity
        hidden = torch.zeros(1, candidate("C").input_width)
        tokens = torch.zeros(1, 2, 128)
        tokens[0, 0, 0] = 1.0
        tokens[0, 1, 0] = 4.0
        ids = ("age", "zip")
        observed = torch.tensor([[True, False]])
        encoded = encode(
            readout,
            "C",
            hidden,
            task_id="rank",
            claimed_identity=identity,
            column_ids=ids,
            column_tokens=tokens,
            observed=observed,
        )
        self.assertEqual(encoded["column_ids"], ("age", "zip"))
        self.assertEqual(encoded["task_id"], "rank")
        self.assertEqual(encoded["schema"], SCHEMA)
        self.assertTrue(torch.equal(encoded["column_tokens"], tokens))
        self.assertTrue(torch.equal(encoded["observed"], observed))
        swapped_tokens = tokens[:, [1, 0], :]
        swapped = encode(
            readout,
            "C",
            hidden,
            task_id="rank",
            claimed_identity=identity,
            column_ids=("zip", "age"),
            column_tokens=swapped_tokens,
            observed=torch.tensor([[False, True]]),
        )
        self.assertEqual(swapped["column_ids"], ("zip", "age"))
        self.assertTrue(torch.equal(swapped["column_tokens"], swapped_tokens))
        with self.assertRaises(TransferEncoderError) as caught:
            encode(
                readout,
                "C",
                hidden,
                task_id="rank",
                claimed_identity=identity,
                column_ids=("secret-column",),
                column_tokens=tokens,
                observed=observed,
            )
        self.assertEqual(str(caught.exception), "transfer_encoder_identity")
        self.assertNotIn("secret-column", str(caught.exception))

    def test_task_head_and_optional_column_tokens_change_the_readout(self):
        self.assertNotEqual(task_slot("rank"), task_slot("generate"))
        readout = realize("A")
        features = torch.zeros(1, DOPE_FEATURES_629_DIM)
        identity = candidate("A").identity
        rank = encode(readout, "A", features, task_id="rank", claimed_identity=identity)
        generate = encode(readout, "A", features, task_id="generate", claimed_identity=identity)
        self.assertEqual(tuple(rank["global"].shape), (1, GLOBAL_WIDTH))
        self.assertFalse(torch.equal(rank["global"], generate["global"]))
        self.assertIsNone(rank["column_tokens"])
        with_columns = realize("C", column_tokens=True)
        hidden = torch.zeros(1, candidate("C").input_width)
        zeros = torch.zeros(1, 1, 128)
        ones = torch.ones(1, 1, 128)
        mask = torch.tensor([[True]])
        with torch.no_grad():
            with_columns.column.weight.fill_(0.01)
            with_columns.column.bias.zero_()
        quiet = encode(
            with_columns,
            "C",
            hidden,
            task_id="rank",
            claimed_identity=candidate("C").identity,
            column_ids=("age",),
            column_tokens=zeros,
            observed=mask,
        )
        loud = encode(
            with_columns,
            "C",
            hidden,
            task_id="rank",
            claimed_identity=candidate("C").identity,
            column_ids=("age",),
            column_tokens=ones,
            observed=mask,
        )
        self.assertFalse(torch.equal(quiet["global"], loud["global"]))
        pooled = realize("C", column_tokens=False)
        self.assertLess(inventory(pooled)["parameter_count"], inventory(with_columns)["parameter_count"])

    def test_residual_starts_at_the_teacher_and_receives_a_gradient(self):
        control = ResidualControl()
        incumbent = torch.tensor([2.0, -3.0], dtype=torch.float32, requires_grad=True)
        hidden = torch.ones(2, GLOBAL_WIDTH, dtype=torch.float32)
        score = control(incumbent, hidden)
        self.assertTrue(torch.equal(score, incumbent.detach()))
        score.sum().backward()
        self.assertIsNone(incumbent.grad)
        self.assertIsNotNone(control.correction.weight.grad)
        self.assertGreater(float(control.correction.weight.grad.abs().sum()), 0.0)
        counted = inventory(control)
        self.assertEqual(counted["residual_alpha"], RESIDUAL_ALPHA)
        self.assertFalse(counted["residual_alpha_trainable"])
        self.assertEqual(counted["tensors"][0]["origin"], "residual-zero")
        self.assertEqual(counted["transferred_from_released"], 0)

    def test_released_encoder_output_stays_unchanged(self):
        torch.manual_seed(7)
        encoder = DopeEncoder629().eval()
        features = torch.zeros(1, 256)
        columns = torch.zeros(1, 64, 17)
        with torch.no_grad():
            before = encoder(features, columns).clone()
        weights = [parameter.detach().clone() for parameter in encoder.parameters()]
        source = inspect.getsource(DopeEncoder629.forward)
        teacher = before.detach().clone().requires_grad_(True)
        readout = realize("A")
        encoded = encode(
            readout,
            "A",
            teacher,
            task_id="rank",
            claimed_identity=candidate("A").identity,
        )
        encoded["global"].sum().backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(readout.hidden.weight.grad)
        with torch.no_grad():
            after = encoder(features, columns)
        self.assertTrue(torch.equal(before, after))
        for parameter, saved in zip(encoder.parameters(), weights):
            self.assertTrue(torch.equal(parameter, saved))
        self.assertNotIn("transfer_encoder", source)
        self.assertEqual(inspect.getsource(DopeEncoder629.forward), source)


if __name__ == "__main__":
    unittest.main()
