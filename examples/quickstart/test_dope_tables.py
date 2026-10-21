"""Unit tests for the quickstart encoder and retention helper (no native binary needed).

    python3 -B -m unittest discover -s examples/quickstart -p 'test_*.py' -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dope_eval import AuditorResult, retention_report, worst_retention  # noqa: E402
from dope_tables import MISSING, OTHER, TableEncoder, read_dope_samples  # noqa: E402


def mixed_table(rows: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    city = rng.choice(["Paris", "Lyon", "Nice", "Lille"], size=rows, p=[0.4, 0.3, 0.2, 0.1])
    plan = rng.choice(["basic", "pro"], size=rows)
    income = np.exp(rng.normal(10, 1, rows))
    age = rng.integers(18, 80, rows)
    when = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, rows), unit="D")
    effect = {"Paris": 2.0, "Lyon": 1.0, "Nice": 0.5, "Lille": -1.0}
    y = np.array([effect[c] for c in city]) + 0.02 * age + 0.3 * np.log(income) + rng.normal(0, 0.5, rows)
    table = pd.DataFrame({"city": city, "plan": plan, "income": income, "age": age, "signup": when, "y": y})
    table.loc[rng.random(rows) < 0.05, "income"] = np.nan
    table.loc[rng.random(rows) < 0.03, "city"] = None
    return table


class EncoderTests(unittest.TestCase):
    def test_matrix_is_unit_interval_with_target_last(self):
        table = mixed_table()
        encoder = TableEncoder("y").fit(table)
        matrix = encoder.transform(table)
        finite = matrix[np.isfinite(matrix)]
        self.assertTrue(((finite >= 0) & (finite <= 1)).all())
        self.assertEqual(encoder.encoded_columns[-1], "y")
        self.assertEqual(matrix.shape[1], len(encoder.encoded_columns))
        self.assertEqual(encoder.task, "regression")
        # Missing numeric cells stay missing; DOPE models them.
        self.assertTrue(np.isnan(matrix[:, encoder.encoded_columns.index("income")]).any())

    def test_round_trip_restores_schema_and_categories(self):
        table = mixed_table()
        encoder = TableEncoder("y").fit(table)
        decoded = encoder.inverse_transform(encoder.transform(table))
        self.assertEqual(list(decoded.columns), list(table.columns))
        expected = table["city"].where(table["city"].notna(), None)
        self.assertTrue((decoded["city"].fillna(MISSING) == expected.fillna(MISSING)).all())
        self.assertTrue((decoded["plan"] == table["plan"]).all())
        self.assertTrue(pd.api.types.is_integer_dtype(decoded["age"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(decoded["signup"]))
        drift = (decoded["signup"] - table["signup"]).abs().max()
        self.assertLess(drift, pd.Timedelta(days=1))
        np.testing.assert_allclose(decoded["y"], table["y"], rtol=1e-9, atol=1e-9)
        inner = table["income"].between(*table["income"].quantile([0.01, 0.99]))
        np.testing.assert_allclose(decoded["income"][inner], table["income"][inner], rtol=1e-6)

    def test_target_ordered_codes_follow_the_target(self):
        table = mixed_table(2000)
        encoder = TableEncoder("y", categorical="target_ordered").fit(table)
        spec = next(c for c in encoder.columns if c.name == "city")
        self.assertEqual(spec.kind, "ordered")
        named = [level for level in spec.levels if level not in (MISSING, OTHER)]
        self.assertEqual(named, ["Lille", "Nice", "Lyon", "Paris"])
        self.assertEqual(spec.codes, sorted(spec.codes))
        self.assertEqual(len(set(spec.codes)), len(spec.codes))

    def test_auto_mode_uses_onehot_only_for_small_categoricals(self):
        table = mixed_table()
        table["code"] = [f"c{i % 9}" for i in range(len(table))]
        encoder = TableEncoder("y", categorical="auto").fit(table)
        kinds = {c.name: c.kind for c in encoder.columns}
        self.assertEqual(kinds["code"], "ordered")
        self.assertEqual(kinds["plan"], "binary")
        small = TableEncoder("y", categorical="auto", onehot_max_levels=8).fit(table)
        self.assertEqual({c.name: c.kind for c in small.columns}["city"], "onehot")

    def test_onehot_mode_matches_get_dummies_width(self):
        table = mixed_table()
        encoder = TableEncoder("y", categorical="onehot").fit(table)
        groups = encoder.feature_groups
        # drop_first=True semantics: the most frequent level is the reference.
        self.assertEqual(len(groups["city"]), table["city"].fillna(MISSING).nunique() - 1)
        decoded = encoder.inverse_transform(encoder.transform(table))
        self.assertTrue((decoded["city"].fillna(MISSING) == table["city"].fillna(MISSING)).all())

    def test_booleans_round_trip_as_booleans(self):
        table = mixed_table()
        flag = pd.Series(table["y"] > table["y"].median(), dtype=object)
        flag[:4] = None
        table["vip"] = flag
        encoder = TableEncoder("y").fit(table)
        decoded = encoder.inverse_transform(encoder.transform(table))
        self.assertEqual(set(decoded["vip"].dropna().unique()), {True, False})
        self.assertTrue(decoded["vip"].head(4).isna().all())

    def test_binary_target_and_multiclass_refusal(self):
        table = mixed_table()
        table["label"] = np.where(table["y"] > table["y"].median(), "yes", "no")
        encoder = TableEncoder("label", drop=["y"], positive="yes").fit(table)
        self.assertEqual(encoder.task, "binary")
        self.assertEqual(set(encoder.transform(table)[:, -1]), {0.0, 1.0})
        table["three"] = np.random.default_rng(1).choice(["a", "b", "c"], size=len(table))
        with self.assertRaises(ValueError):
            TableEncoder("three", drop=["y", "label"]).fit(table)

    def test_rare_levels_pool_and_unseen_levels_map(self):
        table = mixed_table()
        table["code"] = [f"c{i % 200}" for i in range(len(table))]
        encoder = TableEncoder("y", max_levels=16).fit(table)
        spec = next(c for c in encoder.columns if c.name == "code")
        self.assertLessEqual(len(spec.levels), 16)
        self.assertIn(OTHER, spec.levels)
        fresh = table.head(5).copy()
        fresh["code"] = "never-seen"
        self.assertTrue(np.isfinite(encoder.transform(fresh)).any())

    def test_binary_state_is_compact_and_deterministic(self):
        table = mixed_table()
        first = TableEncoder("y").fit(table).to_bytes()
        second = TableEncoder("y").fit(table).to_bytes()
        self.assertEqual(first, second)
        self.assertTrue(first.startswith(b"DOPEENC1"))
        self.assertLess(len(first), 400)
        breakdown = TableEncoder("y").fit(table).byte_breakdown()
        self.assertEqual(breakdown["total_bytes"], len(first))

    def test_robust_scaling_resists_outliers(self):
        rng = np.random.default_rng(3)
        table = pd.DataFrame({"x": np.r_[rng.normal(0, 1, 999), 1e6], "y": rng.normal(size=1000)})
        robust = TableEncoder("y", numeric="robust").fit(table).transform(table)[:-1, 0]
        minmax = TableEncoder("y", numeric="minmax").fit(table).transform(table)[:-1, 0]
        self.assertGreater(robust.std(), 0.08)
        self.assertLess(minmax.std(), 1e-5)

    def test_dataset_dir_layout_matches_the_compiler_contract(self):
        table = mixed_table()
        encoder = TableEncoder("y").fit(table)
        with tempfile.TemporaryDirectory() as tmp:
            out = encoder.write_dataset_dir(table, tmp)
            lines = (out / "train.csv").read_text().splitlines()
            self.assertEqual(len(lines), len(table))
            self.assertEqual(len(lines[0].split(",")), len(encoder.encoded_columns))
            self.assertTrue((out / "encoder.bin").read_bytes().startswith(b"DOPEENC1"))
            matrix = read_dope_samples(out / "train.csv")
            np.testing.assert_allclose(np.nan_to_num(matrix, nan=-1.0),
                                       np.nan_to_num(encoder.transform(table), nan=-1.0), atol=1e-8)


    def test_explicit_first_target_is_canonical_last_and_not_a_feature(self):
        table = mixed_table(40)
        table = table[["y"] + [c for c in table.columns if c != "y"]]
        encoder = TableEncoder("y").fit(table)
        self.assertNotIn("y", encoder.source_columns)
        matrix = encoder.transform(table)
        np.testing.assert_allclose(matrix[:, -1], encoder.encode_target(table["y"]))
        self.assertEqual(encoder.transform(table, with_target=False).shape[1], matrix.shape[1] - 1)
        self.assertEqual(encoder.inverse_transform(matrix).columns[-1], "y")

    def test_duplicate_headers_and_existing_outputs_refuse_without_mutation(self):
        duplicate = pd.DataFrame([[1, 2, 3]], columns=["x", "y", "y"])
        with self.assertRaisesRegex(ValueError, "duplicate_column_declaration"):
            TableEncoder("y").fit(duplicate)
        table = mixed_table(40)
        encoder = TableEncoder("y").fit(table)
        with tempfile.TemporaryDirectory() as tmp:
            out = encoder.write_dataset_dir(table, Path(tmp) / "private")
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            with self.assertRaisesRegex(ValueError, "output_directory_unavailable"):
                encoder.write_dataset_dir(table, out)
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})
            alias = Path(tmp) / "alias"
            alias.symlink_to(out, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "output_directory_unavailable"):
                encoder.write_dataset_dir(table, alias)


class RetentionTests(unittest.TestCase):
    def test_definition_and_informative_gate(self):
        self.assertAlmostEqual(AuditorResult("a", 1.0, 0.5, 0.75).retention, 0.5)
        self.assertIsNone(AuditorResult("a", 1.0, 0.999, 0.5).retention)
        self.assertLess(AuditorResult("a", 1.0, 0.5, 1.5).retention, 0)
        results = [AuditorResult("a", 1.0, 0.5, 0.6), AuditorResult("b", 1.0, 0.5, 0.9),
                   AuditorResult("c", 1.0, 1.0, 0.1)]
        self.assertAlmostEqual(worst_retention(results), 0.2)

    def test_informative_minimum_does_not_claim_complete_auditors(self):
        from dope_eval import local_summary
        complete = [AuditorResult("a", 1.0, 0.5, 0.6)]
        partial = complete + [AuditorResult("b", 1.0, 1.0, 0.1)]
        self.assertFalse(local_summary(partial)["required_auditors_complete"])
        self.assertAlmostEqual(local_summary(partial)["worst_retention"], 0.8)
        self.assertTrue(local_summary(complete)["required_auditors_complete"])
        self.assertFalse(local_summary(complete)["tdl_qualified"])
        self.assertIsNone(local_summary([])["worst_retention"])
        self.assertFalse(local_summary([])["required_auditors_complete"])

    def test_custom_target_name_stays_out_of_normal_console_output(self):
        import contextlib
        import io
        import json
        import os
        from unittest.mock import patch
        import quickstart
        target = "CANARY_PRIVATE_TARGET_385729"
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "private.csv"
            pd.DataFrame({target: np.linspace(0.1, 0.9, 40), "feature": np.linspace(0.2, 0.8, 40)}).to_csv(source, index=False)
            out = Path(tmp) / "output"
            def native(command):
                if command[1] == "compile":
                    (out / "model.dpk").write_bytes(b"fixture-artifact")
                    (out / "compile-report.json").write_text(json.dumps({"selected_candidate": "fixture"}))
                if command[1] == "sample":
                    (out / "samples.csv").write_text("0.2,0.4\n0.8,0.6\n")
                return {"marginals": [], "target": "fixture", "noise": "fixture"}
            console = io.StringIO()
            previous_umask = os.umask(0o077)
            try:
                with (
                    patch.object(quickstart, "find_kernel", return_value="fixture-kernel"),
                    patch.object(quickstart, "run", side_effect=native),
                    patch.object(quickstart, "retention_report", return_value=[AuditorResult("linear", 1.0, 0.5, 0.6)]),
                    contextlib.redirect_stdout(console),
                ):
                    self.assertEqual(quickstart.main(["--csv", str(source), "--target", target,
                        "--backend", "chow_liu", "--auditors", "linear", "--out", str(out)]), 0)
            finally:
                os.umask(previous_umask)
            self.assertNotIn(target, console.getvalue())
            summary = json.loads((out / "report.json").read_text())
            self.assertEqual(summary["retention_aggregation"], "minimum_over_informative_local_auditors")
            self.assertFalse(summary["tdl_qualified"])
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)

    def test_real_rows_as_synthetic_give_retention_one(self):
        table = mixed_table(800).drop(columns=["signup"])
        fit, holdout = table.iloc[:600], table.iloc[600:]
        results = retention_report(fit, holdout, fit, "y", "regression", auditors=["linear"])
        self.assertAlmostEqual(results[0].retention, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
