from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dope_kernel.cli import build_parser
from dope_kernel.cli import _resolve_train_tuner_args
from dope_kernel.tuner import (
    DatasetSpec,
    KPI_FIELDS,
    KernelTuner,
    MetricsSink,
    TunerDashboard,
    TunerExample,
    _cap_xy,
    _evaluate_model,
    _prepare_deadline,
    _synthetic_example_from_anchor,
    _training_mix_counts,
    candidate_configs,
    prepare_tuner_examples,
    prepare_tuner_example,
    select_all_supported_datasets,
    select_smoke_datasets,
    train_tuner,
    tuner_batch_loss,
)


def write_dataset(path: Path, task: str, rows: int = 32) -> None:
    rng = np.random.default_rng(zlib.crc32(f"{path}:{task}".encode("utf-8")))
    x0 = rng.random(rows)
    x1 = np.clip(0.3 * x0 + 0.7 * rng.random(rows), 0.0, 1.0)
    if task == "binary":
        y = (x0 + x1 > 1.0).astype(float)
    else:
        y = np.clip(0.2 + 0.5 * x0 + 0.2 * x1, 0.0, 1.0)
    data = np.column_stack([x0, x1, y])
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data[:24]).to_csv(path / "train.csv", header=False, index=False)
    pd.DataFrame(data[24:]).to_csv(path / "test.csv", header=False, index=False)
    (path / "meta.json").write_text(json.dumps({"dataset_hash": path.name, "task_type": task}))


def test_smoke_selector_is_deterministic_and_mixed(tmp_path: Path) -> None:
    for task in ("regression", "binary"):
        for idx in range(3):
            write_dataset(tmp_path / task / f"{task[:3]}-{idx}", task)

    first = select_smoke_datasets(tmp_path, "regression,binary", max_datasets=4)
    second = select_smoke_datasets(tmp_path, "regression,binary", max_datasets=4)

    assert first == second
    assert [spec.task for spec in first] == ["regression", "binary", "regression", "binary"]
    assert all((spec.path / "train.csv").exists() and (spec.path / "test.csv").exists() for spec in first)


def test_all_supported_selector_uses_task_dirs_and_excludes_multi_label(tmp_path: Path) -> None:
    rows = []
    for task in ("regression", "binary", "multi_label"):
        for idx in range(2):
            dataset_id = f"{task[:3]}-{idx}"
            if task != "multi_label":
                write_dataset(tmp_path / task / dataset_id, task)
            rows.append({"dataset_hash": dataset_id, "task_type": task, "output_dir": str(tmp_path / task / dataset_id)})
    write_dataset(tmp_path / "regression" / "reg-extra-not-in-manifest", "regression")
    write_dataset(tmp_path / "binary" / "bin-extra-not-in-manifest", "binary")
    write_dataset(tmp_path / "binary" / ".incoming-hidden", "binary")
    (tmp_path / "manifest.json").write_text(json.dumps({"datasets": rows}), encoding="utf-8")

    selected = select_all_supported_datasets(tmp_path)

    assert len(selected) == 6
    assert {spec.task for spec in selected} == {"regression", "binary"}
    assert all(spec.task != "multi_label" for spec in selected)
    assert "reg-extra-not-in-manifest" in {spec.dataset_id for spec in selected}
    assert "bin-extra-not-in-manifest" in {spec.dataset_id for spec in selected}
    assert ".incoming-hidden" not in {spec.dataset_id for spec in selected}


def test_frozen_encoder_stays_fixed_and_heads_update() -> None:
    torch.manual_seed(1729)
    model = KernelTuner(input_dim=9, hidden_dim=12)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=0.01)
    features = torch.rand(2, 3, 9)
    kpis = torch.rand(2, 3, len(KPI_FIELDS))
    best_index = torch.tensor([1, 2])

    encoder_before = [param.detach().clone() for param in model.encoder.parameters()]
    head_before = [param.detach().clone() for param in model.ranking_head.parameters()]

    loss = tuner_batch_loss(model, features, kpis, best_index)["loss"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert all(not param.requires_grad for param in model.encoder.parameters())
    assert all(torch.equal(before, after) for before, after in zip(encoder_before, model.encoder.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(head_before, model.ranking_head.parameters()))


def test_metrics_sink_writes_jsonl_and_summary(tmp_path: Path) -> None:
    sink = MetricsSink(tmp_path)
    sink.write({"epoch": 1, "rbcs": 0.5})
    sink.write({"epoch": 2, "rbcs": 0.6})
    sink.write_event({"type": "candidate_score", "rbcs": 0.6})
    sink.write_summary({"final": {"rbcs": 0.6}})
    sink.write_report({"final": {"rbcs": 0.6}, "selected_datasets": [], "kernel_summaries": []})

    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert rows == [{"epoch": 1, "rbcs": 0.5}, {"epoch": 2, "rbcs": 0.6}]
    assert events == [{"type": "candidate_score", "rbcs": 0.6}]
    assert summary["final"]["rbcs"] == 0.6
    assert "Dope Kernel Tuner Report" in (tmp_path / "report.html").read_text()


def test_cli_parses_train_tuner() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-tuner",
            "--corpus",
            "/home/ubuntu/remote_super/quant",
            "--tasks",
            "regression,binary",
            "--max-datasets",
            "8",
            "--epochs",
            "3",
            "--batch-size",
            "4",
            "--device",
            "cuda",
            "--out",
            "runs/kernel_tuner_smoke",
            "--tui",
        ]
    )

    assert args.command == "train-tuner"
    assert args.corpus == "/home/ubuntu/remote_super/quant"
    assert args.tasks == "regression,binary"
    assert args.max_datasets == 8
    assert args.tui is True


def test_cli_long_preset_resolves_long_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["train-tuner", "--preset", "long", "--corpus", "/tmp/corpus", "--device", "cpu"])

    assert args.max_datasets == 8
    assert args.all_datasets is False
    _resolve_train_tuner_args(args)

    assert args.max_datasets is None
    assert args.all_datasets is True
    assert args.epochs == 100000
    assert args.batch_size == 16
    assert args.row_cap == 256
    assert args.max_seconds == 10800.0
    assert args.full_kpi_every == 0
    assert args.full_kpi_top_k == 3
    assert args.early_stop_patience == 512
    assert args.early_stop_min_delta == 1e-4
    assert args.synthetic_mode == "synthetic-heavy"


def test_cli_long_preset_keeps_explicit_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(["train-tuner", "--preset", "long", "--corpus", "/tmp/corpus", "--max-datasets", "4", "--max-seconds", "20"])
    _resolve_train_tuner_args(args)

    assert args.max_datasets == 4
    assert args.all_datasets is False
    assert args.max_seconds == 20.0


def test_cli_trust_preset_resolves_trust_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["train-tuner", "--preset", "trust", "--corpus", "/tmp/corpus", "--device", "cpu"])

    _resolve_train_tuner_args(args)

    assert args.all_datasets is True
    assert args.max_datasets is None
    assert args.max_seconds == 43200.0
    assert args.prep_max_seconds == 7200.0
    assert args.target_prepared_datasets == 2000
    assert args.batch_size == 32
    assert args.row_cap == 256
    assert args.synthetic_mode == "real-first"
    assert args.synthetic_label_mode == "anchor-consistency"
    assert args.binary_label_mode == "all-fast"
    assert args.validation_binary_label_mode == "all-fast"
    assert args.full_kpi_audit_count == 200
    assert args.adaptive_relabel is True


def test_candidate_space_expands_empq_k() -> None:
    for task in ("regression", "binary"):
        configs = candidate_configs(task)
        assert len(configs) == 24
        assert {config["empq_k"] for config in configs} == {None, 9, 17, 33}


def test_all_supported_selector_interleaves_tasks_for_time_budgeted_prep(tmp_path: Path) -> None:
    for idx in range(8):
        write_dataset(tmp_path / "regression" / f"reg-{idx}", "regression")
    for idx in range(8):
        write_dataset(tmp_path / "binary" / f"bin-{idx}", "binary")

    selected = select_all_supported_datasets(tmp_path, seed=1729)
    repeated = select_all_supported_datasets(tmp_path, seed=1729)
    reseeded = select_all_supported_datasets(tmp_path, seed=1730)

    assert [spec.task for spec in selected[:8]] == ["regression", "binary", "regression", "binary", "regression", "binary", "regression", "binary"]
    assert [spec.dataset_id for spec in selected] == [spec.dataset_id for spec in repeated]
    assert [spec.dataset_id for spec in selected[:8]] != [spec.dataset_id for spec in reseeded[:8]]


def test_all_datasets_prepare_deadline_leaves_training_time() -> None:
    start = 1000.0

    assert _prepare_deadline(start, 10800.0, all_datasets=True) == start + 900.0
    assert _prepare_deadline(start, 10800.0, all_datasets=False) == start + 10800.0


def test_cli_all_datasets_flag_is_available_for_smoke() -> None:
    parser = build_parser()
    args = parser.parse_args(["train-tuner", "--corpus", "/tmp/corpus", "--all-datasets"])

    assert args.all_datasets is True
    assert args.max_datasets == 8


def test_tui_renderer_accepts_synthetic_events() -> None:
    dashboard = TunerDashboard({"host": "unit", "device": "cpu", "gpu": None, "disk": {"free_mb": 100}, "max_datasets": 2, "config": {"max_seconds": 20}})
    dashboard.apply({"type": "dataset_start", "dataset_index": 0, "dataset": {"dataset_id": "ds0", "task": "regression", "path": "/tmp/ds0"}})
    dashboard.apply(
        {
            "type": "candidate_score",
            "dataset": {"dataset_id": "ds0", "task": "regression", "path": "/tmp/ds0"},
            "candidate_index": 1,
            "score_mode": "fast",
            "rbcs": 0.7,
            "fidelity_real": 0.8,
            "kernel_bytes": 123,
            "config": {"dependence_kind": "ind", "target_kind": "linear_sparse", "empq_k": None},
            "kernel_summary": {
                "canonical_program": "(dk (target kind=linear_sparse))",
                "config": {"dependence_kind": "ind", "target_kind": "linear_sparse", "empq_k": None},
                "kernel_bytes": 123,
                "dependence": {"kind": "ind"},
                "target": {"kind": "linear_sparse"},
                "marginals": {"count": 2, "kinds": {"empq": 2}},
                "top_kpi_components": [{"name": "rbcs", "value": 0.7}],
            },
        }
    )
    dashboard.apply({"type": "epoch_metrics", "train_loss": 0.4, "val_loss": 0.5, "rank_accuracy": 1.0})
    dashboard.apply({"type": "checkpoint", "checkpoint_kind": "latest", "path": "/tmp/latest_tuner.pt", "epoch": 1, "step": 1})

    assert dashboard.render() is not None


def test_tui_renderer_does_not_use_ascii_sparkline_strings() -> None:
    from rich.console import Console

    dashboard = TunerDashboard({"host": "unit", "device": "cpu", "gpu": None, "disk": {"free_mb": 100}, "max_datasets": 1, "config": {}})
    for value in (0.9, 0.6, 0.3):
        dashboard.apply({"type": "train_step", "loss": value})
    dashboard.apply({"type": "epoch_metrics", "train_loss": 0.3, "val_loss": 0.4, "rank_accuracy": 1.0, "rbcs": 0.7})

    console = Console(record=True, width=140)
    console.print(dashboard.render())
    rendered = console.export_text()

    assert "sparkline" not in rendered.lower()
    assert "[=:=#" not in rendered
    assert "Learning Health" in rendered
    assert "Generalization / Best Checkpoint" in rendered
    assert "train eval regret" in rendered
    assert "optimizer loss" in rendered


def test_eval_loss_is_bounded_selection_regret_not_raw_nll() -> None:
    class ConfidentWrongModel(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
            batch, candidates = features.shape[:2]
            logits = torch.tensor([[-20.0, 20.0, 0.0]], dtype=torch.float32, device=features.device).expand(batch, candidates)
            return {
                "ranking_logits": logits,
                "kpi": torch.zeros(batch, candidates, len(KPI_FIELDS), dtype=torch.float32, device=features.device),
                "candidate_loss": torch.zeros(batch, candidates, dtype=torch.float32, device=features.device),
            }

    kpis = np.zeros((3, len(KPI_FIELDS)), dtype=np.float32)
    kpis[:, 0] = np.asarray([1.0, 0.75, 0.0], dtype=np.float32)
    example = TunerExample(
        spec=DatasetSpec(path=Path("/tmp/ds"), task="regression", dataset_id="ds"),
        features=np.zeros((3, 9), dtype=np.float32),
        kpis=kpis,
        best_index=0,
        records=(),
    )

    report = _evaluate_model(ConfidentWrongModel(), [example], torch.device("cpu"))

    assert report["ranking_nll"] > 10.0
    assert report["loss"] == report["selection_regret"]
    assert report["loss"] == 0.25


def test_training_loss_uses_bounded_regret_not_hard_cross_entropy() -> None:
    class ConfidentWrongModel(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
            batch, candidates = features.shape[:2]
            logits = torch.tensor([[-20.0, 20.0, 0.0]], dtype=torch.float32, device=features.device).expand(batch, candidates)
            return {
                "ranking_logits": logits,
                "kpi": torch.zeros(batch, candidates, len(KPI_FIELDS), dtype=torch.float32, device=features.device),
                "candidate_loss": torch.zeros(batch, candidates, dtype=torch.float32, device=features.device),
            }

    features = torch.zeros(1, 3, 9)
    kpis = torch.zeros(1, 3, len(KPI_FIELDS))
    kpis[0, :, 0] = torch.tensor([1.0, 0.75, 0.0])
    best_index = torch.tensor([0])

    loss_parts = tuner_batch_loss(ConfidentWrongModel(), features, kpis, best_index)

    assert float(loss_parts["hard_ranking_nll"]) > 10.0
    assert abs(float(loss_parts["ranking_loss"]) - 0.25) < 1e-6
    assert float(loss_parts["loss"]) < 1.0


def test_binary_sampler_preserves_both_classes_when_possible() -> None:
    X = np.arange(200, dtype=float).reshape(100, 2) / 200.0
    y = np.array([0.0] * 95 + [1.0] * 5)

    _, sampled_y = _cap_xy(X, y, max_rows=16, task="binary", seed=1729)

    assert set(np.unique(sampled_y)) == {0.0, 1.0}


def test_binary_shortlist_uses_proxy_and_still_full_scores_selected_candidates(tmp_path: Path) -> None:
    write_dataset(tmp_path / "binary" / "bin-0", "binary", rows=48)
    spec = select_smoke_datasets(tmp_path, "binary", max_datasets=1)[0]
    events: list[dict[str, object]] = []

    example = prepare_tuner_example(spec, row_cap=24, event_callback=events.append, full_kpi=True, full_kpi_top_k=1)
    score_modes = [record["score_mode"] for record in example.records]

    assert "proxy" in score_modes
    assert any(event["type"] == "binary_shortlist" for event in events)
    assert any(event["type"] == "binary_diagnostics" for event in events)
    assert any(event["type"] == "full_kpi_score" for event in events)
    assert any(event["type"] == "label_quality" for event in events)
    assert any(
        event.get("proxy_vs_fast_delta") is not None
        for event in events
        if event.get("type") == "candidate_score"
    )
    assert example.label_quality["sampled_rows"]["train"] == 24
    assert "class_balance" in example.label_quality


def test_binary_all_fast_label_mode_scores_every_candidate(tmp_path: Path) -> None:
    write_dataset(tmp_path / "binary" / "bin-0", "binary", rows=40)
    spec = select_smoke_datasets(tmp_path, "binary", max_datasets=1)[0]
    events: list[dict[str, object]] = []

    example = prepare_tuner_example(spec, row_cap=16, event_callback=events.append, binary_label_mode="all-fast")
    score_modes = {record["score_mode"] for record in example.records}

    assert len(example.records) == len(candidate_configs("binary"))
    assert score_modes == {"fast"}
    assert example.label_quality["label_mode"] == "all-fast"
    assert example.label_quality["proxy_label_count"] == 0
    assert any(event.get("label_mode") == "all-fast" for event in events if event.get("type") == "binary_shortlist")


def test_example_cache_reuses_prepared_labels(tmp_path: Path) -> None:
    write_dataset(tmp_path / "regression" / "reg-0", "regression", rows=40)
    spec = select_smoke_datasets(tmp_path, "regression", max_datasets=1)[0]
    cache_dir = tmp_path / "cache"
    first_events: list[dict[str, object]] = []
    second_events: list[dict[str, object]] = []

    first = prepare_tuner_examples([spec], row_cap=16, seed=1729, event_callback=first_events.append, example_cache_dir=cache_dir)
    second = prepare_tuner_examples([spec], row_cap=16, seed=1729, event_callback=second_events.append, example_cache_dir=cache_dir)

    assert any(event["type"] == "example_cache_miss" for event in first_events)
    assert any(event["type"] == "example_cache_write" for event in first_events)
    assert any(event["type"] == "example_cache_hit" for event in second_events)
    assert np.array_equal(first[0].features, second[0].features)
    assert np.array_equal(first[0].kpis, second[0].kpis)
    assert first[0].label_quality["scorer_version"] == second[0].label_quality["scorer_version"]


def test_training_mix_counts_honor_real_first_and_synthetic_heavy() -> None:
    assert _training_mix_counts(10, "real-first") == (8, 2)
    assert _training_mix_counts(10, "synthetic-heavy") == (2, 8)
    assert _training_mix_counts(10, "real-only") == (10, 0)
    assert _training_mix_counts(5, "real-first") == (4, 1)
    assert _training_mix_counts(5, "synthetic-heavy") == (1, 4)


def test_anchor_consistency_synthetic_labels_are_low_weight(tmp_path: Path) -> None:
    write_dataset(tmp_path / "regression" / "reg-0", "regression", rows=40)
    spec = select_smoke_datasets(tmp_path, "regression", max_datasets=1)[0]
    anchor = prepare_tuner_example(spec, row_cap=16)

    synthetic = _synthetic_example_from_anchor(anchor, seed=1729, synthetic_label_mode="anchor-consistency")

    assert synthetic is not None
    assert synthetic.source == "synthetic"
    assert synthetic.label_weight < 1.0
    assert synthetic.label_quality["label_source"] == "anchor_consistency"
    assert synthetic.label_quality["authoritative"] is False
    assert {record["label_source"] for record in synthetic.records} == {"anchor_consistency"}


def test_synthetic_training_batches_do_not_drive_validation(tmp_path: Path) -> None:
    for idx in range(4):
        write_dataset(tmp_path / "corpus" / "regression" / f"reg-{idx}", "regression")

    out = tmp_path / "run"
    summary = train_tuner(
        corpus=tmp_path / "corpus",
        tasks="regression",
        max_datasets=4,
        epochs=1,
        batch_size=5,
        device="cpu",
        out=out,
        row_cap=16,
        report_html=False,
        synthetic_mode="synthetic-heavy",
    )
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    train_starts = [event for event in events if event.get("type") == "train_step_start"]

    assert summary["synthetic_mode"] == "synthetic-heavy"
    assert summary["final"]["val_eval_source"] == "held_out_real"
    assert summary["final"]["val_eval_synthetic_examples"] == 0
    assert summary["final"]["optimization_synthetic_examples"] > summary["final"]["optimization_real_examples"]
    assert train_starts
    assert train_starts[0]["real_batch_size"] == 1
    assert train_starts[0]["synthetic_batch_size"] == 4
    checkpoint = torch.load(summary["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["metrics"]["last"]["val_eval_source"] == "held_out_real"
    assert checkpoint["metrics"]["last"]["val_eval_synthetic_examples"] == 0


def test_trust_report_emits_gates_and_audit_counts(tmp_path: Path) -> None:
    for task in ("regression", "binary"):
        for idx in range(2):
            write_dataset(tmp_path / "corpus" / task / f"{task[:3]}-{idx}", task, rows=40)
    out = tmp_path / "run"

    summary = train_tuner(
        corpus=tmp_path / "corpus",
        tasks="regression,binary",
        max_datasets=4,
        epochs=1,
        batch_size=2,
        device="cpu",
        out=out,
        row_cap=16,
        full_kpi_audit_count=1,
        report_html=False,
        synthetic_mode="real-only",
        binary_label_mode="shortlist",
        validation_binary_label_mode="all-fast",
    )
    trust = json.loads((out / "trust_report.json").read_text())

    assert summary["trust_report_json"] == str(out / "trust_report.json")
    assert trust["prepared"]["real_datasets"] == 4
    assert trust["split"]["val_real_datasets"] > 0
    assert trust["labels"]["full_kpi_audited_datasets"] == 1
    assert trust["labels"]["proxy_label_share"] >= 0.0
    assert "beats_random_candidate" in trust["gates"]
    assert trust["trust_status"] in {"experimental", "audit trusted"}


def test_kernel_summary_excludes_embeddings_and_includes_program(tmp_path: Path) -> None:
    write_dataset(tmp_path / "regression" / "reg-0", "regression")
    spec = select_smoke_datasets(tmp_path, "regression", max_datasets=1)[0]
    events: list[dict[str, object]] = []

    example = prepare_tuner_example(spec, row_cap=16, event_callback=events.append, full_kpi=True, full_kpi_top_k=1)

    summary = example.records[example.best_index]["kernel_summary"]
    serialized = json.dumps(summary, sort_keys=True).lower()
    assert summary["canonical_program"]
    assert summary["config"]
    assert summary["kernel_bytes"] > 0
    assert "embedding" not in serialized
    assert "hidden" not in serialized
    assert any(event["type"] == "full_kpi_score" for event in events)


def test_resume_appends_events_and_keeps_outputs_valid(tmp_path: Path) -> None:
    for task in ("regression", "binary"):
        write_dataset(tmp_path / "corpus" / task / f"{task[:3]}-0", task)
    out = tmp_path / "run"

    first = train_tuner(
        corpus=tmp_path / "corpus",
        max_datasets=2,
        epochs=1,
        batch_size=2,
        device="cpu",
        out=out,
        row_cap=16,
        report_html=True,
    )
    initial_event_count = len((out / "events.jsonl").read_text().splitlines())

    second = train_tuner(
        corpus=tmp_path / "corpus",
        max_datasets=2,
        epochs=2,
        batch_size=2,
        device="cpu",
        out=out,
        row_cap=16,
        report_html=True,
        resume=True,
    )
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]

    assert Path(first["latest_checkpoint"]).exists()
    assert Path(second["latest_checkpoint"]).exists()
    assert len(events) > initial_event_count
    assert any(event["type"] == "resume_loaded" for event in events)
    assert (out / "metrics.jsonl").read_text().strip()
    assert (out / "summary.json").exists()
    assert "Dope Kernel Tuner Report" in (out / "report.html").read_text()
