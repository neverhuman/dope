import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.benchmark.mfs_v3_panel import (
    DENSITY_SPECS,
    MODEL_FILES,
    REFUSED_ENCODERS,
    VECTOR_PIN,
    apply_map,
    build_nulls,
    cleartext_absent,
    cohort_document,
    density_lineages,
    distinct_nearest_quantile_squared,
    exact_row_matches,
    fit_map,
    gpu_may_start,
    gpu_stop_reason,
    load_numeric_table,
    near_copy_ok,
    null_seed,
    refuse_test_path,
    retention,
    verify_cell_hash,
)
from research.benchmark.representation import TABULAR_AUDITORS


ROOT = Path(__file__).resolve().parents[3]
DENSITY = ROOT / "research/benchmark/results/density-matched-population-validation.json"


def _cell(dataset, method, configuration, seed, informative=True):
    return {
        "charged_artifact_bytes": 100,
        "configuration": configuration,
        "dataset": dataset,
        "fit_seed": 11,
        "method": method,
        "metric_receipt": {
            "metric_path": f"/tmp/{dataset}-{method}-{seed}.metric.json",
            "sample_sha256": "abc",
        },
        "sample_seed": seed,
        "size_multiplier": 4,
        "status": "ok",
        "utility": {
            "catboost": {
                "informative": informative,
                "retention": 0.5 if informative else None,
            }
        },
    }


class PanelProtocolTest(unittest.TestCase):
    def test_official_test_path_is_refused(self):
        with self.assertRaises(ValueError):
            refuse_test_path(Path("/tmp/worker/test.csv"))
        with self.assertRaises(ValueError):
            load_numeric_table(Path("/tmp/test.csv"))

    def test_median_null_is_fixed_before_an_encoder_distance(self):
        fit = np.column_stack([np.linspace(0.1, 0.9, 12), np.linspace(0.2, 0.8, 12)])
        holdout = np.column_stack([np.linspace(0.15, 0.85, 8), np.linspace(0.25, 0.75, 8)])
        seeds = tuple(null_seed("aa", "DOPE", "features12_steps2048", 101, index) for index in range(3))
        tables, losses, null_index, match_index = build_nulls(fit, holdout, 6, seeds)
        self.assertEqual(len(tables), 3)
        self.assertEqual(len(losses), 3)
        self.assertEqual(null_index, 0)
        self.assertIn(match_index, (0, 1, 2))
        self.assertEqual(match_index, sorted(range(3), key=lambda i: (losses[i], i))[1])
        self.assertTrue(all(np.isfinite(loss) for loss in losses))
        again = build_nulls(fit, holdout, 6, seeds)
        self.assertEqual(losses, again[1])
        with self.assertRaises(ValueError):
            build_nulls(fit, holdout, 6, seeds[:2])
        with self.assertRaises(ValueError):
            build_nulls(fit, holdout, 6, (seeds[0],) * 3)

    def test_near_copy_floor_is_declared_quantile_not_minimum(self):
        from research.benchmark.mfs_v3_score import _v3_contract

        fit = np.arange(101, dtype=float).reshape(-1, 1) * 10
        fit[1] = 0.01
        distances = [0.0001, 0.0001] + [100.0] * 99
        q = _v3_contract()["release_gates"]["near_copy_quantile"]
        expected = float(np.quantile(distances, q, method="linear"))
        self.assertAlmostEqual(distinct_nearest_quantile_squared(fit), expected)
        # With more distinct real rows, the declared quantile exceeds the minimum.
        fit = np.arange(201, dtype=float).reshape(-1, 1) * 10
        fit[1] = 0.01
        self.assertGreater(distinct_nearest_quantile_squared(fit), 0.0001)
        self.assertFalse(near_copy_ok(fit, np.array([[0.1]])))
        self.assertFalse(near_copy_ok(np.zeros((2, 1)), np.ones((1, 1))))

    def test_pins_and_a_tiny_cohort_do_not_claim_a_win(self):
        self.assertEqual(list(TABULAR_AUDITORS), ["kumo_tabular_l", "mitra_v2", "tabicl2"])
        self.assertEqual(VECTOR_PIN["kumo_tabular_l"], "row_project_output_mean_labeled_rows")
        self.assertIn("tabpfn35", REFUSED_ENCODERS)
        self.assertIn("foundation", REFUSED_ENCODERS)
        self.assertTrue(MODEL_FILES["kumo_tabular_l"].endswith("large/regressor.pt"))
        cells = []
        for dataset in ("aa", "bb"):
            for method, configuration in DENSITY_SPECS:
                for seed in (101, 211, 307):
                    cells.append(_cell(dataset, method, configuration, seed))
        document = cohort_document({"cells": cells})
        self.assertIs(document["counts_as_dope_win"], False)
        self.assertIsNone(document["superiority"])
        self.assertFalse(document["official_tests_opened"])
        self.assertEqual(document["lineages"], ["aa", "bb"])
        self.assertEqual(document["protocol"], "train_on_synthetic_score_on_real")
        self.assertTrue(all(cell["counts_as_dope_win"] is False for cell in document["cells"]))

    def test_noninformative_lineage_is_omitted(self):
        cells = []
        for method, configuration in DENSITY_SPECS:
            for seed in (101, 211, 307):
                cells.append(_cell("aa", method, configuration, seed, informative=method != "DOPE"))
                cells.append(_cell("bb", method, configuration, seed))
        self.assertEqual(density_lineages({"cells": cells}), ["bb"])
        self.assertIsNone(retention(1.0, 0.999, 0.5))
        self.assertAlmostEqual(retention(1.0, 0.5, 0.75), 0.5)

    def test_exact_and_near_copy_on_the_shared_map(self):
        fit = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
        copied = np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float64)
        self.assertEqual(exact_row_matches(fit, copied), 1)
        self.assertFalse(near_copy_ok(fit, copied))
        shifted = fit + 0.5
        self.assertEqual(exact_row_matches(fit, shifted), 0)
        mapped = apply_map(fit_map(fit), np.array([[5.0, -1.0]]))
        self.assertLess(mapped[0, 0], 1.0)
        self.assertGreater(mapped[0, 0], 0.0)

    def test_internal_projection_codes_are_not_source_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.dpk"
            artifact.write_bytes(b"weights c10 age")
            projection = root / "projection.json"
            projection.write_text(json.dumps({"input_columns": ["c0", "c1"], "target": "c10"}))
            self.assertIsNone(cleartext_absent(artifact, projection))
            projection.write_text(json.dumps({"input_columns": ["age"], "target": "c10"}))
            self.assertFalse(cleartext_absent(artifact, projection))
            artifact.write_bytes(b"weights only")
            self.assertTrue(cleartext_absent(artifact, projection))

    def test_gpu_yield_thresholds(self):
        self.assertTrue(gpu_may_start(19000, 0.4))
        self.assertFalse(gpu_may_start(9000, 0.4))
        self.assertEqual(gpu_stop_reason(7000, 0.4, 100, 100), "free_vram")
        self.assertEqual(gpu_stop_reason(12000, 0.10, 100, 100), "mem_available")
        self.assertEqual(gpu_stop_reason(12000, 0.4, 3200, 1000), "other_grew")
        self.assertIsNone(gpu_stop_reason(12000, 0.4, 2000, 1000))


class PublishedDensityCohortTest(unittest.TestCase):
    def test_paired_lineages_and_one_sample_hash(self):
        record = json.loads(DENSITY.read_text())
        lineages = density_lineages(record)
        self.assertEqual(len(lineages), 97)
        document = cohort_document(record)
        self.assertEqual(len(document["cells"]), 97 * 4 * 3)
        sample = Path(document["cells"][0]["sample_csv"])
        self.assertNotIn("test.csv", sample.name)
        # The csv lives on the benchmark scratch disk. CI checks the receipt
        # counts; the byte hash is checked wherever that disk is mounted.
        if sample.is_file():
            self.assertTrue(verify_cell_hash(document["cells"][0]))


class HoldoutPackTest(unittest.TestCase):
    def test_a_value_outside_the_unit_interval_is_refused_without_echoing_it(self):
        from research.benchmark.mfs_v3_holdout import _matrix

        with self.assertRaises(ValueError) as caught:
            _matrix(np.array([[0.2, 1.5], [0.4, 0.6]]))
        self.assertNotIn("1.5", str(caught.exception))

    def test_pack_record_matches_the_scorer_layout(self):
        import io
        import struct

        from research.benchmark.mfs_v3_holdout import MAGIC, append_cell

        handle = io.BytesIO()
        handle.write(MAGIC)
        cell = {
            "configuration": "features12_steps2048",
            "dataset": "abc",
            "method": "DOPE",
            "sample_seed": 101,
        }
        arrays = {
            "fit": np.array([[0.25, 0.75], [0.5, 0.5]]),
            "holdout": np.array([[0.1, 0.2]]),
            "synthetic": np.array([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
        }
        append_cell(handle, cell, arrays)
        raw = handle.getvalue()
        self.assertTrue(raw.startswith(MAGIC))
        fit_rows, hold_rows, syn_rows, width = struct.unpack_from("<IIII", raw, len(MAGIC))
        self.assertEqual((fit_rows, hold_rows, syn_rows, width), (2, 1, 3, 2))


def _measured_seed(cleartext=True):
    def auditor(distance, d_null, d_match):
        return {
            "d_match": d_match,
            "d_null": d_null,
            "distance": distance,
            "null_loss": 1.0,
            "real_loss": 0.2,
            "synthetic_loss": 0.36,
        }

    return {
        "artifact_bytes": 1000,
        "auditors": {
            "kumo_tabular_l": auditor(0.2, 0.5, 0.4),
            "mitra_v2": auditor(0.4, 0.5, 0.4),
            "tabicl2": auditor(0.3, 0.5, 0.4),
        },
        "cleartext_absent": cleartext,
        "exact_row_matches": 0,
        "holdout": {
            "attribute_inference_advantage": 0.0,
            "coverage_realism": 1.0,
            "dependence_fidelity": 1.0,
            "driver_fidelity": 1.0,
            "marginal_fidelity": 1.0,
            "membership_auc": 0.5,
            "mmd_fidelity": 1.0,
            "sliced_wasserstein_fidelity": 1.0,
        },
        "near_copy_ok": True,
    }


class AssembleScoreTest(unittest.TestCase):
    def test_distribution_uses_the_production_mix(self):
        from research.benchmark.mfs_v3_assemble import distribution_fidelity

        self.assertEqual(distribution_fidelity(1.0, 1.0, 1.0), 1.0)
        self.assertEqual(distribution_fidelity(0.0, 0.0, 0.0), 0.0)
        self.assertIsNone(distribution_fidelity(-0.1, 1.0, 1.0))
        self.assertIsNone(distribution_fidelity(1.0, 1.1, 1.0))

    def test_legacy_lineage_without_prerequisite_evidence_stays_null(self):
        from research.benchmark.mfs_v3_assemble import panel_receipt, score_lineage

        seeds = [_measured_seed() for _ in range(3)]
        scored = score_lineage(seeds, lambda values: sum(values) / len(values))
        self.assertFalse(scored["counts_as_dope_win"])
        self.assertIsNone(scored["superiority"])
        from research.benchmark.mfs_v3_score import PREREQUISITES
        self.assertTrue(set(PREREQUISITES).issubset(scored["failed_gates"]))
        self.assertAlmostEqual(scored["retention"]["kumo_tabular_l"], 0.8)
        self.assertIsNone(scored["score"])
        withheld = score_lineage([_measured_seed(cleartext=None) for _ in range(3)], lambda values: 0.8)
        self.assertIsNone(withheld["score"])
        self.assertIn("cleartext_absent", withheld["failed_gates"])
        self.assertAlmostEqual(withheld["retention"]["kumo_tabular_l"], 0.8)
        receipt = panel_receipt([scored])
        self.assertFalse(receipt["counts_as_dope_win"])
        self.assertFalse(receipt["official_tests_opened"])

    def test_measured_prerequisites_propagate_and_fractional_bytes_do_not(self):
        from research.benchmark.mfs_v3_assemble import score_lineage
        from research.benchmark.mfs_v3_score import PREREQUISITES

        seeds = [dict(_measured_seed(), **{key: True for key in PREREQUISITES}) for _ in range(3)]
        self.assertIsNotNone(score_lineage(seeds, lambda values: 0.8)["score"])
        seeds[0]["artifact_bytes"] = 1000.5
        self.assertIn("tier_bytes", score_lineage(seeds, lambda values: 0.8)["failed_gates"])

    def test_prepared_prerequisites_survive_loading_and_receipt_compaction(self):
        import tempfile

        from research.benchmark.mfs_v3_assemble import SAMPLE_SEEDS, compact_seed, load_seeds, score_lineage
        from research.benchmark.mfs_v3_score import PREREQUISITES

        measured = dict(_measured_seed(), **{key: True for key in PREREQUISITES})
        root = Path(__file__).resolve().parents[3] / "target"
        with tempfile.TemporaryDirectory(dir=root) as directory:
            folder = Path(directory)
            for sample_seed in SAMPLE_SEEDS:
                base = folder / f"seed{sample_seed}"
                base.with_suffix(".json").write_text(json.dumps(measured))
                for auditor, values in measured["auditors"].items():
                    base.with_name(base.name + f".{auditor}.json").write_text(json.dumps(values))
                base.with_name(base.name + ".holdout.json").write_text(json.dumps(measured["holdout"]))
            loaded = load_seeds(folder)
            self.assertIsNotNone(score_lineage(loaded, lambda values: 0.8)["score"])
            for seed in loaded:
                self.assertTrue(all(compact_seed(seed)[key] is True for key in PREREQUISITES))
            prepared_path = folder / f"seed{SAMPLE_SEEDS[0]}.json"
            measured.pop(PREREQUISITES[0])
            measured[PREREQUISITES[1]] = False
            prepared_path.write_text(json.dumps(measured))
            loaded = load_seeds(folder)
            self.assertIsNone(compact_seed(loaded[0])[PREREQUISITES[0]])
            self.assertIs(compact_seed(loaded[0])[PREREQUISITES[1]], False)
            report = score_lineage(loaded, lambda values: 0.8)
            self.assertIsNone(report["score"])
            self.assertTrue(set(PREREQUISITES[:2]).issubset(report["failed_gates"]))

    def test_one_invalid_seed_cannot_be_hidden_by_lineage_aggregation(self):
        from research.benchmark.mfs_v3_assemble import score_lineage
        from research.benchmark.mfs_v3_score import PREREQUISITES

        for field, gate, values in (
            ("artifact_bytes", "tier_bytes", (-1, 0, 10241, 1000.5, True, None)),
            ("exact_row_matches", "exact_row_match", (False, 0.0, 1, None)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    seeds = [dict(_measured_seed(), **{key: True for key in PREREQUISITES}) for _ in range(3)]
                    seeds[0][field] = value
                    report = score_lineage(seeds, lambda values: 0.8)
                    self.assertIsNone(report["score"])
                    self.assertIn(gate, report["failed_gates"])
        for field, gate in (("membership_auc", "membership_auc"),
                            ("attribute_inference_advantage", "attribute_inference")):
            seeds = [dict(_measured_seed(), **{key: True for key in PREREQUISITES}) for _ in range(3)]
            seeds[0]["holdout"][field] = -1.0
            report = score_lineage(seeds, lambda values: 0.8)
            self.assertIsNone(report["score"])
            self.assertIn(gate, report["failed_gates"])

    def test_one_invalid_component_cannot_be_hidden_by_minimum_or_mixture(self):
        from research.benchmark.mfs_v3_assemble import score_lineage
        from research.benchmark.mfs_v3_score import PREREQUISITES

        fields = ("marginal_fidelity", "sliced_wasserstein_fidelity", "mmd_fidelity",
                  "dependence_fidelity", "coverage_realism", "driver_fidelity")
        for field in fields:
            for value in (-0.1, 1.1):
                with self.subTest(field=field, value=value):
                    seeds = [dict(_measured_seed(), **{key: True for key in PREREQUISITES}) for _ in range(3)]
                    seeds[0]["holdout"][field] = value
                    report = score_lineage(seeds, lambda values: 0.8)
                    self.assertIsNone(report["score"])
                    self.assertIn("mfs_components_complete", report["failed_gates"])

    def test_invalid_auditor_distances_and_losses_stay_null(self):
        from research.benchmark.mfs_v3_assemble import score_lineage
        from research.benchmark.mfs_v3_score import PREREQUISITES

        for auditor in ("kumo_tabular_l", "mitra_v2", "tabicl2"):
            for field, value, gate in (
                ("distance", -0.1, "representation_closeness"),
                ("distance", 1.1, "representation_closeness"),
                ("d_null", -0.1, "representation_closeness"),
                ("d_null", 1.1, "representation_closeness"),
                ("null_loss", -0.1, "tabular_transfer"),
                ("real_loss", -0.1, "tabular_transfer"),
                ("synthetic_loss", -0.1, "tabular_transfer"),
            ):
                with self.subTest(auditor=auditor, field=field, value=value):
                    seeds = [dict(_measured_seed(), **{key: True for key in PREREQUISITES}) for _ in range(3)]
                    seeds[0]["auditors"][auditor][field] = value
                    report = score_lineage(seeds, lambda values: 0.8)
                    self.assertIsNone(report["score"])
                    self.assertIn(gate, report["failed_gates"])

    def test_retention_maps_keep_only_finite_lineage_medians(self):
        from research.benchmark.mfs_v3_assemble import retention_maps

        maps = retention_maps(
            [
                {"dataset": "aa", "method": "DOPE", "retention": {"kumo_tabular_l": 0.5, "mitra_v2": None}},
                {"dataset": "bb", "method": "DOPE", "retention": {"kumo_tabular_l": float("nan")}},
            ]
        )
        self.assertEqual(maps["kumo_tabular_l"]["DOPE"], {"aa": 0.5})
        self.assertEqual(maps["mitra_v2"]["DOPE"], {})

    def test_a_cohort_that_claims_a_win_is_refused(self):
        import tempfile

        from research.benchmark.mfs_v3_assemble import collect_panel

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            path.write_text(json.dumps({"counts_as_dope_win": True, "official_tests_opened": False, "cells": []}))
            with self.assertRaises(ValueError):
                collect_panel(path, lambda values: 0.0)
