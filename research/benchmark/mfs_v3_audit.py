"""Score one frozen auditor on a prepared MFS-v3 cohort.

Local weight files only. One model resident at a time. The process exits when
the yield rule trips; it does not stop any other process.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from research.benchmark.mfs_v3_panel import (
    KUMO_ESTIMATORS,
    MODEL_FILES,
    VECTOR_PIN,
    WEIGHT_ROOT,
    gpu_may_start,
    gpu_stop_reason,
    refuse_test_path,
)
from research.benchmark.representation import REFUSED_ENCODERS, unit_half_distance

OFM_CODE = Path("/home/ubuntu/jopedime_probe_exp/ofm-20261008/code")
OFM_SRC = Path("/home/ubuntu/jopedime_probe_exp/ofm-20261008/src/structured-data-models")


def gpu_snapshot() -> dict:
    import subprocess

    used, total, util = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(", ")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    other = 0
    own = os.getpid()
    for line in apps.splitlines():
        if not line.strip():
            continue
        pid, memory = line.split(", ")
        if int(pid) != own:
            other += int(memory)
    available = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0])
    total_kb = int(Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0])
    return {
        "free_mib": float(total) - float(used),
        "used_mib": float(used),
        "mem_available_fraction": available / total_kb,
        "other_used_mib": float(other),
        "utilization": float(util),
    }


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and bool(np.isfinite(value))


def _already_scored(path: Path, auditor_name: str) -> bool:
    if not path.is_file():
        return False
    try:
        prior = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    if prior.get("auditor") != auditor_name or prior.get("counts_as_dope_win") is not False:
        return False
    if not (_finite(prior.get("synthetic_loss")) and _finite(prior.get("real_loss"))):
        return False
    if prior.get("vector_ok") is False:
        return True
    return all(_finite(prior.get(key)) for key in ("distance", "d_null", "d_match"))


def _mean_loss(context_y: np.ndarray, query_y: np.ndarray) -> float:
    return float(np.mean((query_y - float(context_y.mean())) ** 2))


class KumoLarge:
    name = "kumo_tabular_l"
    estimators = KUMO_ESTIMATORS

    def __init__(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        sys.path.insert(0, str(OFM_CODE))
        sys.path.insert(0, str(OFM_SRC))
        weight = WEIGHT_ROOT / MODEL_FILES[self.name]
        if not weight.is_file():
            raise FileNotFoundError(weight)
        import sdm
        import sdm.models.kumo.tabular.model as kumo_model
        from ofm_models import Kumo

        def local_checkpoint(repo_id, filename, **_kwargs):
            root = WEIGHT_ROOT / "models/nvidia__Kumo-Tabular"
            if filename.endswith("regressor.pt"):
                return str(root / "large/regressor.pt")
            if filename.endswith("classifier.pt"):
                return str(root / "large/classifier.pt")
            if filename.endswith("config.json"):
                return str(root / "config.json")
            raise FileNotFoundError(filename)

        kumo_model.download_checkpoint = local_checkpoint
        self.recipe = kumo_model.KumoTabular(task="regression", size="large", device="cuda")
        self.vector_model = Kumo("large", "cuda")
        self.sdm = sdm
        self.captured: dict = {}

    def close(self) -> None:
        del self.recipe
        del self.vector_model

    def _predict(self, context: np.ndarray, query: np.ndarray) -> np.ndarray:
        import torch

        columns = [f"f{index}" for index in range(context.shape[1] - 1)]
        features = self._frame(context[:, :-1], columns)
        target = self._frame(context[:, -1:], ["y"])
        query_features = self._frame(query[:, :-1], columns)
        stypes = self.sdm.infer_stypes(features)
        device = "cuda"
        context_x = self.sdm.TableTensor.from_pandas(df=features, stypes=stypes, device=device)
        context_y = self.sdm.TableTensor.from_pandas(
            df=target, stypes={"y": "numerical"}, device=device
        )
        query_x = self.sdm.TableTensor.from_pandas(df=query_features, stypes=stypes, device=device)
        generator = torch.Generator(device=device).manual_seed(0)
        with torch.amp.autocast(device, torch.float16):
            self.recipe.fit(
                x=context_x,
                y=context_y,
                num_estimators=self.estimators,
                generator=generator,
            )
            raw = self.recipe.predict(x=query_x).numerical.detach().float().cpu().numpy()
        predicted = np.sort(raw.reshape(query.shape[0], -1), axis=1).mean(axis=1)
        return predicted

    @staticmethod
    def _frame(values: np.ndarray, columns: list[str]):
        import pandas as pd

        return pd.DataFrame(np.ascontiguousarray(values, dtype=np.float64), columns=columns)

    def loss(self, context: np.ndarray, query: np.ndarray) -> float:
        prediction = self._predict(context, query)
        if prediction.shape != query[:, -1].shape or not np.isfinite(prediction).all():
            raise RuntimeError("auditor prediction is not a finite holdout vector")
        return float(np.mean((prediction - query[:, -1]) ** 2))

    def vector(self, table: np.ndarray) -> np.ndarray:
        import torch

        self.captured.clear()
        handle = self.vector_model.m.row_project.register_forward_hook(
            lambda _module, _inputs, output: self.captured.__setitem__("h", output.detach())
        )
        try:
            self.vector_model.run(
                np.ascontiguousarray(table[:, :-1], dtype=np.float64),
                np.ascontiguousarray(table[:, -1], dtype=np.float64),
                np.ascontiguousarray(table[:1, :-1], dtype=np.float64),
            )
        finally:
            handle.remove()
        hidden = self.captured["h"].float()
        if hidden.ndim == 3:
            hidden = hidden[0]
        labeled = hidden[: table.shape[0]]
        mean = labeled.mean(dim=0)
        norm = torch.linalg.vector_norm(mean).clamp(min=1e-12)
        return (mean / norm).detach().cpu().numpy().astype(np.float64)


class MissingRowEmbedding(RuntimeError):
    """The auditor has no separable row embedding for the frozen pin."""


def _unit(vector: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise MissingRowEmbedding("row embedding is not a finite vector")
    norm = float(np.linalg.norm(flat))
    if norm <= 1e-12:
        raise MissingRowEmbedding("row embedding has no direction")
    return flat / norm


def _context_scale(target: np.ndarray) -> tuple[float, float]:
    center = float(target.mean())
    scale = float(target.std())
    if not np.isfinite(center) or not np.isfinite(scale) or scale < 1e-8:
        return center, 1.0
    return center, scale


class _OfmAuditor:
    """Zero-shot Mitra or TabICL. One instance lives in the process."""

    def __init__(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(OFM_CODE))
        weight = WEIGHT_ROOT / MODEL_FILES[self.name]
        if not weight.is_file():
            raise FileNotFoundError(weight)
        from ofm_models import Mitra, TabICL

        self.model = {"mitra_v2": Mitra, "tabicl2": TabICL}[self.name]("cuda")

    def close(self) -> None:
        del self.model

    def _run(self, context: np.ndarray, query: np.ndarray, target: np.ndarray):
        return self.model.run(
            np.ascontiguousarray(context, dtype=np.float64),
            np.ascontiguousarray(target, dtype=np.float64),
            np.ascontiguousarray(query, dtype=np.float64),
        )

    def loss(self, context: np.ndarray, query: np.ndarray) -> float:
        target = context[:, -1]
        if self.name == "tabicl2":
            center, scale = _context_scale(target)
            target = (target - center) / scale
        else:
            center, scale = 0.0, 1.0
        _hidden, prediction, _spread, _extra = self._run(context[:, :-1], query[:, :-1], target)
        predicted = np.asarray(prediction, dtype=np.float64).reshape(-1) * scale + center
        if predicted.shape != query[:, -1].shape or not np.isfinite(predicted).all():
            raise RuntimeError("auditor prediction is not a finite holdout vector")
        return float(np.mean((predicted - query[:, -1]) ** 2))

    def vector(self, table: np.ndarray) -> np.ndarray:
        target = table[:, -1]
        if self.name == "tabicl2":
            center, scale = _context_scale(target)
            target = (target - center) / scale
        hidden, _prediction, _spread, _extra = self._run(table[:, :-1], table[:, :-1], target)
        rows = np.asarray(hidden, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[0] != table.shape[0]:
            raise MissingRowEmbedding(
                f"the frozen pin returned shape {tuple(int(axis) for axis in rows.shape)} for {table.shape[0]} rows"
            )
        return _unit(rows.mean(axis=0))


class MitraV2(_OfmAuditor):
    name = "mitra_v2"


class TabICL2(_OfmAuditor):
    name = "tabicl2"


AUDITORS = {
    "kumo_tabular_l": KumoLarge,
    "mitra_v2": MitraV2,
    "tabicl2": TabICL2,
}


def score_arrays(auditor, fit, holdout, synthetic, nulls, match_index: int) -> dict:
    null_loss = _mean_loss(fit[:, -1], holdout[:, -1])
    real_loss = auditor.loss(fit, holdout)
    synthetic_loss = auditor.loss(synthetic, holdout)
    distances = {"d_match": None, "d_null": None, "distance": None, "vector_ok": False}
    try:
        real_vector = auditor.vector(holdout)
        synthetic_vector = auditor.vector(synthetic)
        null_vector = auditor.vector(nulls[0])
        match_vector = auditor.vector(nulls[match_index])
        distances = {
            "d_match": unit_half_distance(real_vector.tolist(), match_vector.tolist()),
            "d_null": unit_half_distance(real_vector.tolist(), null_vector.tolist()),
            "distance": unit_half_distance(real_vector.tolist(), synthetic_vector.tolist()),
            "vector_ok": True,
        }
    except MissingRowEmbedding as error:
        print(f"vector missing: {error}", flush=True)
    return {
        "auditor": auditor.name,
        "null_loss": null_loss,
        "real_loss": real_loss,
        "estimators": getattr(auditor, "estimators", None),
        "synthetic_loss": synthetic_loss,
        "vector_pin": VECTOR_PIN[auditor.name],
        **distances,
    }


def score_manifest(manifest: Path, auditor_name: str, limit: int) -> None:
    if auditor_name in REFUSED_ENCODERS or auditor_name not in AUDITORS:
        raise ValueError("refused encoder")
    document = json.loads(manifest.read_text())
    if document.get("counts_as_dope_win") is not False or document.get("official_tests_opened") is not False:
        raise ValueError("cohort claims are not allowed")
    cells = document["cells"][: limit or None]
    pending = []
    skipped = 0
    for cell in cells:
        mapped = Path(cell["mapped"])
        refuse_test_path(mapped)
        if _already_scored(mapped.with_name(mapped.stem + f".{auditor_name}.json"), auditor_name):
            skipped += 1
            continue
        pending.append(cell)
    run_path = manifest.parent / f"{auditor_name}.run.json"
    if not pending:
        run_path.write_text(
            json.dumps(
                {"auditor": auditor_name, "pending": 0, "skipped": skipped, "counts_as_dope_win": False},
                sort_keys=True,
            )
            + "\n"
        )
        print(f"already scored {skipped} cells", flush=True)
        return
    snapshot = gpu_snapshot()
    if not gpu_may_start(snapshot["free_mib"], snapshot["mem_available_fraction"]):
        raise SystemExit(f"yield: start refused {snapshot}")
    baseline = snapshot["other_used_mib"]
    peak_used = snapshot["used_mib"]
    auditor = AUDITORS[auditor_name]()
    done = 0
    try:
        peak_used = max(peak_used, gpu_snapshot()["used_mib"])
        for index, cell in enumerate(pending, start=1):
            mapped = Path(cell["mapped"])
            now = gpu_snapshot()
            peak_used = max(peak_used, now["used_mib"])
            reason = gpu_stop_reason(
                now["free_mib"], now["mem_available_fraction"], now["other_used_mib"], baseline
            )
            if reason is not None:
                print(f"yield: {reason} after {done} new cells {now}", flush=True)
                raise SystemExit(75)
            started = time.perf_counter()
            arrays = np.load(mapped)
            nulls = [arrays["null0"], arrays["null1"], arrays["null2"]]
            scored = score_arrays(
                auditor,
                arrays["fit"],
                arrays["holdout"],
                arrays["synthetic"],
                nulls,
                int(cell["match_index"]),
            )
            now = gpu_snapshot()
            peak_used = max(peak_used, now["used_mib"])
            scored.update(
                {
                    "configuration": cell["configuration"],
                    "counts_as_dope_win": False,
                    "dataset": cell["dataset"],
                    "method": cell["method"],
                    "official_tests_opened": False,
                    "sample_seed": cell["sample_seed"],
                    "seconds": time.perf_counter() - started,
                    "superiority": None,
                    "used_mib": now["used_mib"],
                }
            )
            out = mapped.with_name(mapped.stem + f".{auditor_name}.json")
            out.write_text(json.dumps(scored, sort_keys=True) + "\n")
            done += 1
            run_path.write_text(
                json.dumps(
                    {
                        "auditor": auditor_name,
                        "counts_as_dope_win": False,
                        "done": done,
                        "official_tests_opened": False,
                        "peak_used_mib": peak_used,
                        "pending": len(pending),
                        "skipped": skipped,
                        "superiority": None,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            print(
                f"{done + skipped}/{len(cells)} {cell['dataset']} {cell['method']} seed {cell['sample_seed']} "
                f"loss {scored['synthetic_loss']:.6g} distance {scored['distance']} "
                f"seconds {scored['seconds']:.1f} used_mib {now['used_mib']:.0f}",
                flush=True,
            )
    finally:
        auditor.close()
        print(f"peak_used_mib {peak_used:.0f} done {done} skipped {skipped}", flush=True)


def main() -> None:
    manifest = Path(os.environ["MFS_V3_MANIFEST"])
    limit = int(os.environ.get("MFS_V3_LIMIT", "0"))
    score_manifest(manifest, os.environ.get("MFS_V3_AUDITOR", "kumo_tabular_l"), limit)


if __name__ == "__main__":
    main()
