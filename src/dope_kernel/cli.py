from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .eval import evaluate_kernel
from .gp import gp_search
from .kernel import fit_kernel_from_dir, load_kernel, sample_kernel, save_kernel
from .utils import canonical_dumps, normalize_task, write_headerless_csv


SMOKE_TUNER_DEFAULTS = {
    "max_datasets": 8,
    "epochs": 3,
    "batch_size": 4,
    "row_cap": 64,
    "max_seconds": None,
    "prep_max_seconds": None,
    "target_prepared_datasets": None,
    "checkpoint_every": 300.0,
    "full_kpi_every": 0,
    "full_kpi_top_k": 3,
    "full_kpi_audit_count": 0,
    "early_stop_patience": 0,
    "early_stop_min_delta": 0.0,
    "synthetic_mode": "real-first",
    "synthetic_label_mode": "anchor-consistency",
    "synthetic_fast_relabel_limit": 1,
    "binary_label_mode": "shortlist",
    "validation_binary_label_mode": "all-fast",
    "example_cache_dir": None,
    "adaptive_relabel": False,
}
LONG_TUNER_DEFAULTS = {
    "epochs": 100000,
    "batch_size": 16,
    "row_cap": 256,
    "max_seconds": 10800.0,
    "prep_max_seconds": None,
    "target_prepared_datasets": None,
    "checkpoint_every": 300.0,
    "full_kpi_every": 0,
    "full_kpi_top_k": 3,
    "full_kpi_audit_count": 0,
    "early_stop_patience": 512,
    "early_stop_min_delta": 1e-4,
    "synthetic_mode": "synthetic-heavy",
    "synthetic_label_mode": "anchor-consistency",
    "synthetic_fast_relabel_limit": 1,
    "binary_label_mode": "shortlist",
    "validation_binary_label_mode": "all-fast",
    "example_cache_dir": None,
    "adaptive_relabel": False,
}
TRUST_TUNER_DEFAULTS = {
    "epochs": 100000,
    "batch_size": 32,
    "row_cap": 256,
    "max_seconds": 43200.0,
    "prep_max_seconds": 7200.0,
    "target_prepared_datasets": 2000,
    "checkpoint_every": 300.0,
    "full_kpi_every": 0,
    "full_kpi_top_k": 3,
    "full_kpi_audit_count": 200,
    "early_stop_patience": 512,
    "early_stop_min_delta": 1e-4,
    "synthetic_mode": "real-first",
    "synthetic_label_mode": "anchor-consistency",
    "synthetic_fast_relabel_limit": 1,
    "binary_label_mode": "all-fast",
    "validation_binary_label_mode": "all-fast",
    "example_cache_dir": None,
    "adaptive_relabel": True,
}


class _StoreExplicit(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: str | None = None) -> None:
        explicit = getattr(namespace, "_explicit_train_tuner_args", set())
        if not isinstance(explicit, set):
            explicit = set(explicit)
        explicit.add(self.dest)
        setattr(namespace, "_explicit_train_tuner_args", explicit)
        setattr(namespace, self.dest, values)


class _StoreTrueExplicit(argparse.Action):
    def __init__(self, option_strings: list[str], dest: str, default: Any = None, required: bool = False, help: str | None = None) -> None:
        super().__init__(option_strings=option_strings, dest=dest, nargs=0, default=default, required=required, help=help)

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: str | None = None) -> None:
        explicit = getattr(namespace, "_explicit_train_tuner_args", set())
        if not isinstance(explicit, set):
            explicit = set(explicit)
        explicit.add(self.dest)
        setattr(namespace, "_explicit_train_tuner_args", explicit)
        setattr(namespace, self.dest, True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dope-kernel", description="Fit, sample, evaluate, and search Dope Data Kernels.")
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit a symbolic data kernel from train.csv")
    fit.add_argument("--dataset-dir", required=True)
    fit.add_argument("--task", required=True, choices=["regression", "binary", "reg", "bin"])
    fit.add_argument("--out", required=True)
    fit.add_argument("--seed", type=int, default=None)

    sample = sub.add_parser("sample", help="sample a headerless synthetic CSV from a kernel")
    sample.add_argument("--kernel", required=True)
    sample.add_argument("--rows", required=True, type=int)
    sample.add_argument("--out", required=True)
    sample.add_argument("--seed", type=int, default=None)

    ev = sub.add_parser("eval", help="evaluate a kernel against real train/test data")
    ev.add_argument("--real-dir", required=True)
    ev.add_argument("--kernel", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--seed", type=int, default=1729)

    gp = sub.add_parser("gp-search", help="search typed kernel variants under a time budget")
    gp.add_argument("--dataset-dir", required=True)
    gp.add_argument("--budget-secs", required=True, type=float)
    gp.add_argument("--out", required=True)
    gp.add_argument("--task", choices=["regression", "binary", "reg", "bin"], default=None)
    gp.add_argument("--seed", type=int, default=1729)

    tuner = sub.add_parser("train-tuner", help="train a frozen-backbone kernel proposal tuner")
    tuner.add_argument("--preset", choices=["smoke", "long", "trust"], default="smoke")
    tuner.add_argument("--corpus", required=True)
    tuner.add_argument("--tasks", default="regression,binary")
    tuner.add_argument("--max-datasets", type=int, default=SMOKE_TUNER_DEFAULTS["max_datasets"], action=_StoreExplicit)
    tuner.add_argument("--all-datasets", action="store_true", help="use every supported regression/binary dataset in the corpus")
    tuner.add_argument("--epochs", type=int, default=SMOKE_TUNER_DEFAULTS["epochs"], action=_StoreExplicit)
    tuner.add_argument("--batch-size", type=int, default=SMOKE_TUNER_DEFAULTS["batch_size"], action=_StoreExplicit)
    tuner.add_argument("--device", default="cuda")
    tuner.add_argument("--out", default="runs/kernel_tuner_smoke")
    tuner.add_argument("--seed", type=int, default=1729)
    tuner.add_argument("--row-cap", type=int, default=SMOKE_TUNER_DEFAULTS["row_cap"], action=_StoreExplicit)
    tuner.add_argument("--max-seconds", type=float, default=SMOKE_TUNER_DEFAULTS["max_seconds"], action=_StoreExplicit)
    tuner.add_argument("--prep-max-seconds", type=float, default=SMOKE_TUNER_DEFAULTS["prep_max_seconds"], action=_StoreExplicit)
    tuner.add_argument("--target-prepared-datasets", type=int, default=SMOKE_TUNER_DEFAULTS["target_prepared_datasets"], action=_StoreExplicit)
    tuner.add_argument("--checkpoint-every", type=float, default=SMOKE_TUNER_DEFAULTS["checkpoint_every"], action=_StoreExplicit)
    tuner.add_argument("--full-kpi-every", type=int, default=SMOKE_TUNER_DEFAULTS["full_kpi_every"], action=_StoreExplicit)
    tuner.add_argument("--full-kpi-top-k", type=int, default=SMOKE_TUNER_DEFAULTS["full_kpi_top_k"], action=_StoreExplicit)
    tuner.add_argument("--full-kpi-audit-count", type=int, default=SMOKE_TUNER_DEFAULTS["full_kpi_audit_count"], action=_StoreExplicit)
    tuner.add_argument("--early-stop-patience", type=int, default=SMOKE_TUNER_DEFAULTS["early_stop_patience"], action=_StoreExplicit)
    tuner.add_argument("--early-stop-min-delta", type=float, default=SMOKE_TUNER_DEFAULTS["early_stop_min_delta"], action=_StoreExplicit)
    tuner.add_argument(
        "--synthetic-mode",
        choices=["real-first", "synthetic-heavy", "real-only"],
        default=SMOKE_TUNER_DEFAULTS["synthetic_mode"],
        action=_StoreExplicit,
        help="optimizer batch mix: real-first=80/20, synthetic-heavy=20/80 experimental, real-only disables augmentation",
    )
    tuner.add_argument(
        "--synthetic-label-mode",
        choices=["anchor-consistency", "fast-relabel"],
        default=SMOKE_TUNER_DEFAULTS["synthetic_label_mode"],
        action=_StoreExplicit,
        help="synthetic label source: low-weight anchor consistency or bounded fast relabeling",
    )
    tuner.add_argument(
        "--synthetic-fast-relabel-limit",
        type=int,
        default=SMOKE_TUNER_DEFAULTS["synthetic_fast_relabel_limit"],
        action=_StoreExplicit,
        help="maximum fast-relabel synthetic examples per optimizer batch",
    )
    tuner.add_argument(
        "--binary-label-mode",
        choices=["shortlist", "all-fast"],
        default=SMOKE_TUNER_DEFAULTS["binary_label_mode"],
        action=_StoreExplicit,
        help="binary training label mode; validation remains controlled separately",
    )
    tuner.add_argument(
        "--validation-binary-label-mode",
        choices=["shortlist", "all-fast"],
        default=SMOKE_TUNER_DEFAULTS["validation_binary_label_mode"],
        action=_StoreExplicit,
        help="binary label mode for held-out validation and audit examples",
    )
    tuner.add_argument("--example-cache-dir", default=SMOKE_TUNER_DEFAULTS["example_cache_dir"], action=_StoreExplicit)
    tuner.add_argument("--adaptive-relabel", action=_StoreTrueExplicit, default=SMOKE_TUNER_DEFAULTS["adaptive_relabel"])
    tuner.add_argument("--report-html", dest="report_html", action="store_true", default=True)
    tuner.add_argument("--no-report-html", dest="report_html", action="store_false")
    tuner.add_argument("--resume", action="store_true")
    tuner.add_argument("--tui", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "fit": handle_fit,
        "sample": handle_sample,
        "eval": handle_eval,
        "gp-search": handle_gp_search,
        "train-tuner": handle_train_tuner,
    }
    handlers[args.command](args)
    return 0


def handle_fit(args: argparse.Namespace) -> None:
    kernel = fit_kernel_from_dir(args.dataset_dir, normalize_task(args.task), seed=args.seed)
    save_kernel(kernel, args.out)
    print_summary({"out": args.out, "kernel_bytes": kernel["kernel_bytes"], "program": kernel["program"]})


def handle_sample(args: argparse.Namespace) -> None:
    kernel = load_kernel(args.kernel)
    data = sample_kernel(kernel, args.rows, seed=args.seed)
    write_headerless_csv(data, args.out)
    print_summary({"out": args.out, "rows": int(args.rows), "cols": int(data.shape[1])})


def handle_eval(args: argparse.Namespace) -> None:
    kernel = load_kernel(args.kernel)
    report = evaluate_kernel(args.real_dir, kernel, seed=args.seed)
    write_json(report, args.out)
    print_summary({"out": args.out, "rbcs": report["kernel"]["rbcs"], "fidelity_real": report["kernel"]["fidelity_real"]})


def handle_gp_search(args: argparse.Namespace) -> None:
    kernel = gp_search(args.dataset_dir, args.budget_secs, task=args.task, seed=args.seed)
    save_kernel(kernel, args.out)
    print_summary({"out": args.out, "fitness": kernel["gp_search"]["fitness"], "program": kernel["program"]})


def handle_train_tuner(args: argparse.Namespace) -> None:
    from .tuner import train_tuner

    _resolve_train_tuner_args(args)
    summary = train_tuner(
        corpus=args.corpus,
        tasks=args.tasks,
        max_datasets=args.max_datasets,
        all_datasets=args.all_datasets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        out=args.out,
        tui=args.tui,
        seed=args.seed,
        row_cap=args.row_cap,
        max_seconds=args.max_seconds,
        prep_max_seconds=args.prep_max_seconds,
        target_prepared_datasets=args.target_prepared_datasets,
        checkpoint_every=args.checkpoint_every,
        full_kpi_every=args.full_kpi_every,
        full_kpi_top_k=args.full_kpi_top_k,
        full_kpi_audit_count=args.full_kpi_audit_count,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        report_html=args.report_html,
        resume=args.resume,
        preset=args.preset,
        synthetic_mode=args.synthetic_mode,
        synthetic_label_mode=args.synthetic_label_mode,
        synthetic_fast_relabel_limit=args.synthetic_fast_relabel_limit,
        binary_label_mode=args.binary_label_mode,
        validation_binary_label_mode=args.validation_binary_label_mode,
        example_cache_dir=args.example_cache_dir,
        adaptive_relabel=args.adaptive_relabel,
    )
    print_summary(
        {
            "out": summary["out"],
            "checkpoint": summary["checkpoint"],
            "summary": str(Path(summary["out"]) / "summary.json"),
            "report_html": summary.get("report_html"),
            "trust_report": summary.get("trust_report_json"),
            "trust_status": summary.get("trust", {}).get("trust_status"),
            "rbcs": summary.get("final", {}).get("rbcs"),
            "device": summary["device"],
            "gpu": summary["gpu"],
        }
    )


def _resolve_train_tuner_args(args: argparse.Namespace) -> None:
    if args.preset not in {"long", "trust"}:
        return
    defaults = TRUST_TUNER_DEFAULTS if args.preset == "trust" else LONG_TUNER_DEFAULTS
    explicit = getattr(args, "_explicit_train_tuner_args", set())
    for key, value in defaults.items():
        if key not in explicit:
            setattr(args, key, value)
    if "max_datasets" not in explicit:
        args.all_datasets = True
        args.max_datasets = None


def write_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(obj) + "\n", encoding="utf-8")


def print_summary(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, sort_keys=True))
