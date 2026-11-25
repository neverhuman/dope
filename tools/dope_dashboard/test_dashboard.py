"""Unit tests for the read-only DOPE Fleet & Fidelity KPI Dashboard."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp.test_utils import AioHTTPTestCase

from tools.dope_dashboard.metrics import fleet_summary, host_view
from tools.dope_dashboard.probe import collect_snapshot
from tools.dope_dashboard.server import DopeFleetState, build_app


class TestDopeDashboardUnits(unittest.TestCase):
    def test_probe_and_metrics_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "dope_v5_s3_ddp4"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(json.dumps({"world_size": 4}), encoding="utf-8")
            (run_dir / "train.jsonl").write_text(
                json.dumps(
                    {
                        "step": 20,
                        "time": 1000.0,
                        "loss": 0.42,
                        "tstr_loss": 0.25,
                        "driver_loss": 0.12,
                        "realism_loss": 0.08,
                        "train_unclipped_retention": 0.89,
                        "tables_per_s": 14.5,
                        "unique_splits_seen": 480,
                        "avg_n_fit": 220.0,
                        "avg_n_cols": 24.0,
                        "avg_noise_cols": 3.2,
                        "avg_derived_cols": 1.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "eval.jsonl").write_text(
                json.dumps(
                    {
                        "step": 20,
                        "time": 1005.0,
                        "main_kpi_retention_median": 0.91,
                        "regression_retention": 0.93,
                        "binary_retention": 0.90,
                        "multiclass_retention": 0.88,
                        "multilabel_retention": 0.87,
                        "parity_coverage": 0.82,
                        "realism_coverage": 0.90,
                        "privacy_pass_coverage": 0.96,
                        "driver_spearman_median": 0.85,
                        "c2st_auc_median": 0.58,
                        "dcr_share_median": 0.50,
                        "tables_evaluated": 48,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            snap = collect_snapshot((td,))
            self.assertEqual(len(snap["runs"]), 1)
            hv = host_view("worker-1", snap, 1006.0, "", 1007.0)
            self.assertEqual(hv["status"], "live")
            hist: deque = deque(maxlen=32)
            summ = fleet_summary([hv], hist, 1007.0)
            self.assertEqual(summ["live_nodes"], 1)
            self.assertEqual(summ["formal_dp_state"], "not_claimed")
            self.assertAlmostEqual(summ["best_eval"]["main_kpi_retention_median"], 0.91, places=4)
            self.assertEqual(summ["unique_splits_seen"], 480)


class TestDopeDashboardServer(AioHTTPTestCase):
    async def get_application(self):
        state = DopeFleetState(("ctrl-0", "worker-1"))
        state.feed("worker-1", {"host": "worker-1", "at": 100.0, "resources": {"cpus": 60}, "gpus": [], "procs": [], "runs": []})
        state.pulse()
        return build_app(state)

    async def test_http_index_and_snapshot(self) -> None:
        resp = await self.client.get("/")
        self.assertEqual(resp.status, 200)
        body = await resp.text()
        self.assertIn("DOPE v2.0.0", body)
        self.assertIn("Checkpoints &amp; Lineage", body)
        self.assertIn("CanvasChart", body)
        api = await self.client.get("/api/snapshot")
        self.assertEqual(api.status, 200)
        payload = await api.json()
        self.assertEqual(payload["type"], "snapshot")
        self.assertEqual(payload["fleet"]["total_nodes"], 2)


class TestDopeDashboardMacro(AioHTTPTestCase):
    async def get_application(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.macro = Path(self._tmp.name) / "fleet_macro.json"
        self.macro.write_text(json.dumps({"schema": "dope.fleet-macro/v1", "headline": {"tables_per_s_now": 0.0}}), encoding="utf-8")
        return build_app(DopeFleetState(("ctrl-0",)), self.macro)

    async def tearDownAsync(self) -> None:
        self._tmp.cleanup()

    async def test_macro_route_serves_and_reports_missing(self) -> None:
        resp = await self.client.get("/api/macro")
        self.assertEqual(resp.status, 200)
        self.assertEqual((await resp.json())["schema"], "dope.fleet-macro/v1")
        self.macro.write_text("{not json", encoding="utf-8")
        self.assertEqual((await self.client.get("/api/macro")).status, 503)
        self.macro.unlink()
        self.assertEqual((await self.client.get("/api/macro")).status, 404)
        body = await (await self.client.get("/")).text()
        self.assertIn("Whole campaign", body)
        self.assertIn("/api/macro", body)


if __name__ == "__main__":
    unittest.main()
