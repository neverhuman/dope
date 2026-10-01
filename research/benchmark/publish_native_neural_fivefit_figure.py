"""Regenerate shared-auditor five-fit pilot comparison figures."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path


def render(report, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (
        report["format"] != "dope-native-neural-fivefit-matched-validation"
        or report["official_tests_opened"] is not False
        or report["production_certified"] is not False
        or report["mfs_v2"] is not None
        or report["ptf_v1"] is not None
    ):
        raise ValueError("matched figure input claim changed")
    summary = report["summaries"]["per_group"]
    perfit = report["summaries"]["per_fit"]
    matplotlib.rcParams["svg.hashsalt"] = "dope-native-neural-fivefit-v1"
    figure, axes = plt.subplots(2, 3, figsize=(14.3, 8.1), sharex=True)
    colors = {
        "DOPE q8": "#31688e",
        "CTGAN default": "#777777",
        "CTGAN tuned": "#b35a25",
        "CTGAN default + tuned": "#b35a25",
        "TVAE default": "#777777",
        "TVAE tuned": "#2b8560",
        "TVAE default + tuned": "#2b8560",
    }
    for i, dataset in enumerate(("Adult", "California")):
        for j, auditor in enumerate(("catboost", "linear", "mlp")):
            axis = axes[i, j]
            groups = sorted(
                {
                    (r["method"], r["configuration"])
                    for r in summary
                    if r["dataset"] == dataset and r["auditor"] == auditor
                }
            )
            for k, (method, configuration) in enumerate(groups):
                label = (
                    "DOPE q8"
                    if method == "DOPE"
                    else f'{method} {configuration.replace("_and_"," + ")}'
                )
                color = colors[label]
                offset = (k - (len(groups) - 1) / 2) * 0.08
                rows = [
                    next(
                        r
                        for r in summary
                        if (
                            r["dataset"],
                            r["auditor"],
                            r["method"],
                            r["configuration"],
                            r["size_multiplier"],
                        )
                        == (dataset, auditor, method, configuration, z)
                    )
                    for z in (1, 4)
                ]
                centers = [r["complete_median_of_fit_medians"] for r in rows]
                x = [offset, 1 + offset]
                if all(v is None for v in centers):
                    axis.plot(
                        [],
                        [],
                        color=color,
                        marker="x",
                        linestyle="none",
                        label=f"{label}: {rows[0]['complete_fit_seeds']}/{rows[0]['planned_fit_seeds']} complete fits",
                    )
                    continue
                for z, location, row in zip((1, 4), x, rows):
                    fits = [
                        r["complete_median_retention"]
                        for r in perfit
                        if (
                            r["dataset"],
                            r["auditor"],
                            r["method"],
                            r["configuration"],
                            r["size_multiplier"],
                        )
                        == (dataset, auditor, method, configuration, z)
                        and r["complete_median_retention"] is not None
                    ]
                    axis.scatter(
                        [location] * len(fits),
                        fits,
                        s=15,
                        color=color,
                        alpha=0.38,
                        zorder=2,
                    )
                    if row["complete_median_of_fit_medians"] is not None:
                        axis.vlines(
                            location,
                            row["complete_fit_min_retention"],
                            row["complete_fit_max_retention"],
                            color=color,
                            linewidth=1,
                            zorder=1,
                        )
                if all(v is not None for v in centers):
                    axis.plot(
                        x,
                        centers,
                        color=color,
                        linewidth=2,
                        marker="D",
                        markersize=4,
                        label=label,
                        zorder=3,
                    )
                else:
                    axis.plot([], [], color=color, label=f"{label}: incomplete")
            axis.set_title(
                f'{dataset} · {auditor.title() if auditor!="linear" else "Linear/logistic"}'
            )
            axis.set_xticks([0, 1], ["n", "4n"])
            axis.set_xlim(-0.28, 1.28)
            axis.grid(axis="y", color="#d7dce1", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(fontsize=7, loc="best")
            if j == 0:
                axis.set_ylabel("Null-normalized TSTR/TRTR retention")
    figure.suptitle(
        "DOPE vs native-selected CTGAN/TVAE: five fixed fit seeds", fontsize=14
    )
    figure.text(
        0.5,
        0.012,
        "Training-derived pilot validation; three sample seeds per fit; diamonds = complete five-fit median; bars = fit range.\nOfficial tests sealed; L3 bytes and release/privacy eligibility reported separately; gated scores null.",
        ha="center",
        fontsize=8,
        color="#4a5661",
    )
    figure.tight_layout(rect=[0, 0.06, 1, 0.96])
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
    from research.benchmark.publish_native_neural_fivefit import render as check_tables

    path = Path(__file__).with_name("results") / "pilot24-native-neural-fivefit.json"
    report = json.loads(path.read_text())
    check_tables(report)
    render(report, path)
    print("matched five-fit validation figures regenerated")


if __name__ == "__main__":
    main()
