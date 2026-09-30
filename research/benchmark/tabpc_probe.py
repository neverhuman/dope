"""Train one pinned TabPC author configuration on train-only pilot projection.

This is dependency and cost evidence, not a locked comparison adapter. The
upstream checkpoint includes a training tensor and cannot pass release checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/fast-scratch/dope-benchmark")
SOURCE = ROOT / "source-snapshots" / "tabpc-a470848"
CONFIG = {"Adult": (4096, 512, 0.25), "News": (1024, 512, 0.1)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIG)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    worker = ROOT / "public-prepared" / "worker" / args.dataset
    manifest = json.loads((worker / "worker-manifest.json").read_text())
    if sha256(worker / "train.csv") != manifest["projected_hashes"]["train"]:
        raise ValueError("train hash mismatch")
    if sha256(worker / "projection.json") != manifest["projection_sha256"]:
        raise ValueError("projection hash mismatch")
    probe = ROOT / "pilot-4h" / "tabpc-probe" / f"{args.dataset}-{args.seed}"
    if probe.exists():
        raise FileExistsError("probe artifact already exists")
    probe.mkdir(parents=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import pandas as pd
    import torch
    from src.experiment_config import ExperimentConfig
    from src.models.probabilistic_circuit import ProbabilisticCircuit
    from src.nn.optimization import Optimization
    from src.preprocessors.df_type_conversion import TypeConversion
    from src.preprocessors.nan_handler import NanHandler
    from src.preprocessors.df_dequantization import Dequantization
    from src.preprocessors.df_quantile_normalizer import QuantileNormalizer
    from src.preprocessors.string_to_int import StringToInt
    from src.preprocessors.to_tensor import ToTensor

    if not torch.cuda.is_available():
        raise RuntimeError("idle GPU required for TabPC probe")
    numeric = pd.read_csv(worker / "train.csv", header=None)
    numeric.columns = [f"c{i}" for i in range(numeric.shape[1])]
    data_path = probe / "train-only.csv"
    numeric.to_csv(data_path, index=False)
    units, batch, rate = CONFIG[args.dataset]
    model = ProbabilisticCircuit(
        fit_preprocessor_only_on_train=False,
        optimization=Optimization(
            epochs=150, batch_size=batch, compile_nn=False,
            lr_scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
            lr_scheduler_params={"patience": 0, "factor": 0.85, "min_lr": 2e-4},
            optimizer_class=torch.optim.RAdam,
            optimizer_hyperparameters={"lr": rate}, patience=10),
        validation_size=0.10,
        preprocessors=[TypeConversion(cat_max_states=50, int_max_states=None),
                       NanHandler(handle_inflated_values=True, handle_missing_values=False),
                       Dequantization(dequantize=True, dequantize_all_floats=False),
                       QuantileNormalizer(), StringToInt(), ToTensor()],
        num_sum_units=units, num_input_units=units,
        region_graph="chow-liu-tree", sum_product_layer="cp", bin_for_mi=15)
    experiment = ExperimentConfig(dataset_path=str(data_path), model=model,
                                  name=f"tabpc_{args.dataset}_{args.seed}",
                                  device="cuda:0", seed=args.seed)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    report = model.train(numeric, device="cuda:0", wandb_run=None, use_codecarbon=False)
    fit_seconds = time.perf_counter() - start
    training_peak_vram = torch.cuda.max_memory_allocated()
    experiment.store(probe / "artifact")
    sample_start = time.perf_counter()
    loaded = __import__("src.experiment_config", fromlist=["load_experiment"]).load_experiment(probe / "artifact")
    sample = loaded.model.generate(device="cuda:0", batch_size=100, num_to_generate=100)
    sample.to_csv(probe / "sample-100.csv", index=False)
    receipt = {"format": "dope-tabpc-author-config-probe", "dataset": args.dataset,
               "fit_seed": args.seed, "fit_seconds": fit_seconds,
               "sample_100_seconds": time.perf_counter() - sample_start,
               "training_peak_vram_bytes": training_peak_vram,
               "sampling_peak_vram_bytes": torch.cuda.max_memory_allocated(),
               "torch_version": torch.__version__,
               "artifact_bytes": sum(p.stat().st_size for p in (probe / "artifact").rglob("*") if p.is_file()),
               "retains_source_rows": True, "train_sha256": manifest["projected_hashes"]["train"],
               "training_report": {key: value for key, value in report.items()
                                   if key in ("best_validation_loss", "num_parameters", "mean_epoch_time")}}
    (probe / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    sys.path[:0] = [str(SOURCE), str(SOURCE / "cirkit")]
    main()
