"""Regenerate Adult/California fixed q8 fit-seed stability figures."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path


RESULTS = Path(__file__).with_name("results")
REPORT = RESULTS / "pilot24-dope-q8-fivefit.json"
DATASETS = ("Adult", "California")
FITS = (11, 23, 37, 53, 71)
SIZES = (1, 4)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(REPORT.read_text())
    if (report["format"] != "dope-pilot24-q8-fivefit-validation"
            or report["status_counts"]["fit_ok"] != 10
            or report["status_counts"]["sample_ok"] != 120
            or report["status_counts"]["metric_replay_exact"] != 60
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None
            or report["ptf_v1"] is not None
            or report["production_certified"] is not False):
        raise ValueError("q8 five-fit figure input changed")
    by_fit = {(row["dataset"], row["fit_seed"], row["size_multiplier"]): row
              for row in report["fit_summary"]}
    by_dataset = {(row["dataset"], row["size_multiplier"]): row
                  for row in report["dataset_summary"]}
    if (set(by_fit) != {(dataset, fit, size) for dataset in DATASETS
                        for fit in FITS for size in SIZES}
            or set(by_dataset) != {(dataset, size) for dataset in DATASETS
                                   for size in SIZES}):
        raise ValueError("q8 figure fit matrix incomplete")
    matplotlib.rcParams["svg.hashsalt"] = "dope-q8-fivefit-validation-v1"
    figure, axes = plt.subplots(1, 2, figsize=(11.3, 4.4), sharey=False)
    colors = ("#31688e", "#e07b39", "#35a389", "#9c5caa", "#b49b31")
    for axis, dataset in zip(axes, DATASETS):
        values = []
        for fit, color in zip(FITS, colors):
            pair = [by_fit[(dataset, fit, size)]["median_catboost_retention"]
                    for size in SIZES]
            values.extend(pair)
            axis.plot([0, 1], pair, marker="o", markersize=5,
                      linewidth=1.4, color=color, label=f"Fit {fit}")
        center = [by_dataset[(dataset, size)][
            "median_of_fit_medians_catboost_retention"] for size in SIZES]
        values.extend(center)
        axis.plot([0, 1], center, marker="D", markersize=5,
                  linewidth=2.5, linestyle="--", color="#202b38",
                  label="Median across fits")
        low, high = min(values), max(values)
        pad = max(0.04, (high - low) * 0.18)
        axis.set_ylim(low - pad, high + pad)
        axis.set_xlim(-0.12, 1.12)
        axis.set_xticks([0, 1], ["n", "4n"])
        axis.set_title(dataset)
        axis.grid(axis="y", color="#d7dce1", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("CatBoost TSTR/TRTR retention")
    axes[1].legend(loc="best", fontsize=8)
    figure.suptitle("DOPE q8: five fixed fit seeds on pilot validation")
    figure.text(0.5, 0.008,
                "Three sample seeds per fit and size; official tests sealed; "
                "release gates unverified",
                ha="center", fontsize=8, color="#4a5661")
    figure.tight_layout(rect=[0, 0.06, 1, 0.95])
    svg, pdf = BytesIO(), BytesIO()
    figure.savefig(svg, format="svg",
                   metadata={"Date": None, "Creator": "DOPE benchmark"})
    figure.savefig(pdf, format="pdf",
                   metadata={"CreationDate": None, "ModDate": None,
                             "Creator": "DOPE benchmark"})
    plt.close(figure)
    svg_text = svg.getvalue().decode("utf-8")
    (RESULTS / "pilot24-dope-q8-fivefit.svg").write_text(
        "".join(line.rstrip() + "\n" for line in svg_text.splitlines()))
    (RESULTS / "pilot24-dope-q8-fivefit.pdf").write_bytes(pdf.getvalue())
    print("q8 five-fit validation figure regenerated")


if __name__ == "__main__":
    main()
