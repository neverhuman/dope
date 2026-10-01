"""Rebuild the descriptive single-fit TabPC/DOPE figure from committed JSON."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from statistics import median


def render(report, destination):
    from .publish_tabpc_native import render as check_tables

    check_tables(report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.hashsalt": "dope-tabpc-native-pilot-v1",
            "font.family": "DejaVu Sans",
            "font.size": 9,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    for axis, dataset in zip(axes, ("California", "News")):
        groups = [
            (
                "DOPE",
                "#2459a6",
                [r for r in report["dope_references"] if r["dataset"] == dataset],
            ),
            (
                "TabPC default",
                "#2c8067",
                [
                    r
                    for r in report["common_cells"]
                    if r["dataset"] == dataset and r["configuration"] == "default"
                ],
            ),
            (
                "TabPC native-selected",
                "#bb5a2b",
                [
                    r
                    for r in report["common_cells"]
                    if r["dataset"] == dataset and r["configuration"] == "tuned"
                ],
            ),
        ]
        for name, color, rows in groups:
            centers = []
            for x, size in enumerate((1, 4)):
                cells = [r for r in rows if r["size_multiplier"] == size]
                if len(cells) != 3 or (
                    name != "DOPE"
                    and any(
                        r["status"] != "ok"
                        or r["metrics"]["utility"]["catboost"]["informative"]
                        is not True
                        for r in cells
                    )
                ):
                    raise ValueError("figure requires all informative sample cells")
                values = [
                    (
                        r["retention"]
                        if name == "DOPE"
                        else r["metrics"]["utility"]["catboost"]["retention"]
                    )
                    for r in cells
                ]
                centers.append(median(values))
                axis.scatter(
                    [x - 0.05, x, x + 0.05], values, color=color, s=20, alpha=0.45
                )
                axis.vlines(x, min(values), max(values), color=color, linewidth=1)
            axis.plot([0, 1], centers, color=color, marker="D", linewidth=2, label=name)
        axis.set_title(dataset)
        axis.set_xticks([0, 1], ["n", "4n"])
        axis.set_xlim(-0.2, 1.2)
        axis.grid(axis="y", color="#d9dfe5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Null-normalized CatBoost TSTR/TRTR retention")
    figure.suptitle("TabPC author-NLL selection vs DOPE: fixed pilot fit seed 23")
    figure.text(
        0.5,
        0.015,
        "Three sample seeds; diamonds = median; bars = sample range. Training-derived validation only.\nAdult: all eight TabPC native trials failed; no DOPE win. TabPC bytes exceed L3. Official tests sealed; gated scores null.",
        ha="center",
        fontsize=8,
        color="#4a5661",
    )
    figure.tight_layout(rect=[0, 0.1, 1, 0.94])
    svg, pdf = BytesIO(), BytesIO()
    figure.savefig(
        svg, format="svg", metadata={"Date": None, "Creator": "DOPE benchmark"}
    )
    figure.savefig(
        pdf,
        format="pdf",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "DOPE benchmark"},
    )
    plt.close(figure)
    destination.with_suffix(".svg").write_text(
        "".join(line.rstrip() + "\n" for line in svg.getvalue().decode().splitlines())
    )
    destination.with_suffix(".pdf").write_bytes(pdf.getvalue())


def main():
    from jsonschema import Draft202012Validator

    path = Path(__file__).with_name("results") / "pilot24-tabpc-native.json"
    report = json.loads(path.read_text())
    schema = json.loads(path.with_suffix(".schema.json").read_text())
    Draft202012Validator(schema).validate(report)
    render(report, path)
    print("TabPC native pilot figures regenerated")


if __name__ == "__main__":
    main()
