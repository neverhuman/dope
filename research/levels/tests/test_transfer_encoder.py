"""Fixtures for the versioned transfer-encoder candidates. No training run."""
import inspect
import unittest

import torch

from research.levels.contracts import ModelConfig
from research.levels.encoder import DOPE_FEATURES_629_DIM, DopeEncoder629
from research.levels.transfer_encoder import (
    ATTENTION_ELEMENT_CAP,
    COLUMN_WIDTH,
    GLOBAL_WIDTH,
    RESIDUAL_ALPHA,
    SCHEMA,
    FrozenBackboneConsumer,
    ResidualControl,
    TransferEncoderError,
    attention_budget,
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
        self.assertEqual(recorded["attention_element_cap"], ATTENTION_ELEMENT_CAP)
        self.assertTrue(candidate("A").frozen_backbone)
        self.assertFalse(candidate("B").frozen_backbone)
        self.assertFalse(candidate("C").frozen_backbone)
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
            table_rows=1,
        )
        self.assertEqual(encoded["column_ids"], ("age", "zip"))
        self.assertEqual(encoded["task_id"], "rank")
        self.assertEqual(encoded["schema"], SCHEMA)
        accepted = tokens.clone()
        accepted[0, 1].zero_()
        self.assertTrue(torch.equal(encoded["column_tokens"], accepted))
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
            table_rows=1,
        )
        self.assertEqual(swapped["column_ids"], ("zip", "age"))
        accepted_swap = swapped_tokens.clone()
        accepted_swap[0, 0].zero_()
        self.assertTrue(torch.equal(swapped["column_tokens"], accepted_swap))
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
            table_rows=1,
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
            table_rows=1,
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

    def test_continued_candidates_keep_observed_gradients(self):
        observed = torch.tensor([[True, False]])
        ids = ("age", "zip")
        for name in ("B", "C"):
            readout = realize(name, column_tokens=True)
            with torch.no_grad():
                readout.hidden.weight.fill_(0.01)
                readout.hidden.bias.zero_()
                readout.column.weight.fill_(0.01)
                readout.column.bias.zero_()
            features = torch.zeros(1, candidate(name).input_width, requires_grad=True)
            tokens = torch.zeros(1, 2, COLUMN_WIDTH)
            tokens[0, 0, 0] = 0.5
            tokens.requires_grad_(True)
            encoded = encode(
                readout,
                name,
                features,
                task_id="rank",
                claimed_identity=candidate(name).identity,
                column_ids=ids,
                column_tokens=tokens,
                observed=observed,
                table_rows=4,
            )
            encoded["global"].sum().backward()
            self.assertIsNotNone(features.grad)
            self.assertGreater(float(features.grad.abs().sum()), 0.0)
            self.assertIsNotNone(tokens.grad)
            self.assertGreater(float(tokens.grad[0, 0].abs().sum()), 0.0)
            self.assertTrue(torch.equal(tokens.grad[0, 1], torch.zeros(COLUMN_WIDTH)))
        frozen = realize("A")
        teacher = torch.zeros(1, DOPE_FEATURES_629_DIM, requires_grad=True)
        encode(
            frozen,
            "A",
            teacher,
            task_id="rank",
            claimed_identity=candidate("A").identity,
        )["global"].sum().backward()
        self.assertIsNone(teacher.grad)

    def test_unobserved_padding_leaves_the_readout_unchanged(self):
        readout = realize("C", column_tokens=True)
        with torch.no_grad():
            readout.hidden.weight.fill_(0.01)
            readout.hidden.bias.zero_()
            readout.column.weight.fill_(0.01)
            readout.column.bias.zero_()
        observed = torch.tensor([[True, False]])
        base = torch.zeros(1, 2, COLUMN_WIDTH)
        base[0, 0, 0] = 0.5
        hidden = torch.zeros(1, candidate("C").input_width)
        hidden[0, 0] = 0.25
        identity = candidate("C").identity

        def run(padding):
            tokens = base.clone()
            tokens[0, 1].fill_(padding)
            features = hidden.detach().clone().requires_grad_(True)
            leaf = tokens.detach().clone().requires_grad_(True)
            encoded = encode(
                readout,
                "C",
                features,
                task_id="generate",
                claimed_identity=identity,
                column_ids=("age", "zip"),
                column_tokens=leaf,
                observed=observed,
                table_rows=4,
            )
            encoded["global"].sum().backward()
            return encoded["global"].detach(), features.grad.detach(), leaf.grad.detach(), encoded["column_tokens"]

        clean_global, clean_features, clean_tokens, clean_stored = run(0.0)
        dirty_global, dirty_features, dirty_tokens, dirty_stored = run(999.0)
        nan_global, nan_features, nan_tokens, nan_stored = run(float("nan"))
        self.assertTrue(torch.equal(clean_global, dirty_global))
        self.assertTrue(torch.equal(clean_global, nan_global))
        self.assertTrue(torch.equal(clean_features, dirty_features))
        self.assertTrue(torch.equal(clean_features, nan_features))
        self.assertTrue(torch.equal(clean_tokens[0, 0], dirty_tokens[0, 0]))
        self.assertTrue(torch.equal(clean_tokens[0, 0], nan_tokens[0, 0]))
        self.assertTrue(torch.equal(dirty_tokens[0, 1], torch.zeros(COLUMN_WIDTH)))
        self.assertTrue(torch.equal(nan_tokens[0, 1], torch.zeros(COLUMN_WIDTH)))
        self.assertTrue(torch.equal(clean_stored, dirty_stored))
        self.assertTrue(torch.equal(clean_stored, nan_stored))
        self.assertTrue(torch.equal(clean_stored[0, 1], torch.zeros(COLUMN_WIDTH)))
        self.assertTrue(bool(torch.isfinite(nan_global).all()))

    def test_outer_train_leaves_the_released_encoder_frozen(self):
        torch.manual_seed(7)
        encoder = DopeEncoder629()
        with self.assertRaises(TransferEncoderError) as refused:
            FrozenBackboneConsumer(DopeEncoder629(), realize("B"))
        self.assertEqual(str(refused.exception), "transfer_encoder_candidate")
        consumer = FrozenBackboneConsumer(encoder, realize("A"))
        consumer_inv = inventory(consumer)
        self.assertEqual(consumer_inv["transferred_from_released"], 353095)
        self.assertEqual(consumer_inv["transferred_tensors"], 40)
        enc_rows = [r for r in consumer_inv["tensors"] if r["name"].startswith("encoder.")]
        self.assertTrue(enc_rows and all(r["origin"] == "released-frozen" and not r["trainable"] for r in enc_rows))
        saved_parameters = [parameter.detach().clone() for parameter in encoder.parameters()]
        saved_buffers = [buffer.detach().clone() for _, buffer in encoder.named_buffers()]
        encoder.train()
        consumer.train()
        self.assertFalse(encoder.training)
        self.assertTrue(consumer.readout.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        readout_before = consumer.readout.hidden.weight.detach().clone()
        trainable = [parameter for parameter in consumer.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=0.1)
        score = consumer(torch.zeros(1, 256), torch.zeros(1, 64, 17), task_id="rank").sum()
        score.backward()
        optimizer.step()
        self.assertFalse(encoder.training)
        self.assertFalse(torch.equal(consumer.readout.hidden.weight, readout_before))
        for parameter, saved in zip(encoder.parameters(), saved_parameters):
            self.assertTrue(torch.equal(parameter, saved))
        for buffer, saved in zip((buffer for _, buffer in encoder.named_buffers()), saved_buffers):
            self.assertTrue(torch.equal(buffer, saved))

    def test_task_vocabulary_and_table_budget_refuse_without_echo(self):
        self.assertNotEqual(task_slot("rank"), task_slot("generate"))
        for collided in ("ab", "ba"):
            with self.assertRaises(TransferEncoderError) as caught:
                task_slot(collided)
            self.assertEqual(str(caught.exception), "transfer_encoder_identity")
            self.assertNotIn(collided, str(caught.exception))
        config = ModelConfig().validate()
        wide_config = ModelConfig(max_features=256).validate()
        self.assertEqual(
            config.max_context * config.max_features * config.max_features * config.heads,
            ATTENTION_ELEMENT_CAP,
        )
        attention_budget(config.max_context, config.max_features)
        with self.assertRaises(TransferEncoderError) as wide:
            attention_budget(1, wide_config.max_features + 1)
        self.assertEqual(str(wide.exception), "transfer_encoder_limit")
        self.assertNotIn(str(wide_config.max_features + 1), str(wide.exception))
        with self.assertRaises(TransferEncoderError) as tall:
            attention_budget(config.max_context + 1, config.max_features)
        self.assertEqual(str(tall.exception), "transfer_encoder_limit")
        self.assertNotIn(str(config.max_context + 1), str(tall.exception))
        readout = realize("C", column_tokens=True)
        hidden = torch.zeros(1, candidate("C").input_width)
        wide72 = encode(
            readout,
            "C",
            hidden,
            task_id="rank",
            claimed_identity=candidate("C").identity,
            column_ids=tuple(f"c{index}" for index in range(72)),
            column_tokens=torch.zeros(1, 72, COLUMN_WIDTH),
            observed=torch.ones(1, 72, dtype=torch.bool),
            table_rows=4,
        )
        self.assertEqual(wide72["global"].shape, (1, GLOBAL_WIDTH))
        wide_count = wide_config.max_features + 1
        tokens = torch.zeros(1, wide_count, COLUMN_WIDTH)
        observed = torch.ones(1, wide_count, dtype=torch.bool)
        with self.assertRaises(TransferEncoderError) as refused:
            encode(
                readout,
                "C",
                hidden,
                task_id="rank",
                claimed_identity=candidate("C").identity,
                column_ids=tuple(f"c{index}" for index in range(wide_count)),
                column_tokens=tokens,
                observed=observed,
                table_rows=1,
            )
        self.assertEqual(str(refused.exception), "transfer_encoder_limit")
        self.assertNotIn(str(wide_count), str(refused.exception))
        with self.assertRaises(TransferEncoderError) as dt_err:
            encode(
                readout, "C", hidden, task_id="rank", claimed_identity=candidate("C").identity,
                column_ids=("age",), column_tokens=torch.zeros(1, 1, COLUMN_WIDTH, dtype=torch.float64),
                observed=torch.ones(1, 1, dtype=torch.bool), table_rows=1,
            )
        self.assertEqual(str(dt_err.exception), "transfer_encoder_dtype")
        with self.assertRaises(TransferEncoderError) as obs_dt:
            encode(
                readout, "C", hidden, task_id="rank", claimed_identity=candidate("C").identity,
                column_ids=("age",), column_tokens=torch.zeros(1, 1, COLUMN_WIDTH),
                observed=torch.ones(1, 1, dtype=torch.int64), table_rows=1,
            )
        self.assertEqual(str(obs_dt.exception), "transfer_encoder_dtype")
        ctrl = ResidualControl()
        with self.assertRaises(TransferEncoderError) as rc_dt:
            ctrl(torch.zeros(1, dtype=torch.float64), torch.zeros(1, GLOBAL_WIDTH))
        self.assertEqual(str(rc_dt.exception), "transfer_encoder_dtype")
        with self.assertRaises(TransferEncoderError) as rc_dev:
            ctrl(torch.zeros(1), torch.zeros(1, GLOBAL_WIDTH, device="meta"))
        self.assertEqual(str(rc_dev.exception), "transfer_encoder_device")
        with self.assertRaises(TransferEncoderError) as col_dev:
            encode(
                readout, "C", hidden, task_id="rank", claimed_identity=candidate("C").identity,
                column_ids=("age",), column_tokens=torch.zeros(1, 1, COLUMN_WIDTH, device="meta"),
                observed=torch.ones(1, 1, dtype=torch.bool), table_rows=1,
            )
        self.assertEqual(str(col_dev.exception), "transfer_encoder_device")
        blank = torch.zeros(1, 1, COLUMN_WIDTH)
        blank_mask = torch.zeros(1, 1, dtype=torch.bool)
        empty = encode(
            readout,
            "C",
            hidden,
            task_id="validate",
            claimed_identity=candidate("C").identity,
            column_ids=("age",),
            column_tokens=blank,
            observed=blank_mask,
            table_rows=1,
        )
        self.assertTrue(bool(torch.isfinite(empty["global"]).all()))
        empty["global"].sum().backward()


if __name__ == "__main__":
    unittest.main()
