"""Regenerate the five-GPU-fit News q10 validation stability figure."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path


RESULTS = Path(__file__).with_name("results")
REPORT = RESULTS / "pilot24-news-q10-fivefit.json"
FITS = (11, 23, 37, 53, 71)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(REPORT.read_text())
    if (report["format"] != "dope-pilot24-news-q10-fivefit-validation"
            or report["packed_sample_checks_exact"] != 60
            or report["common_metric_replays_exact"] != 30
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None
            or report["ptf_v1"] is not None
            or report["production_certified"] is not False):
        raise ValueError("News five-fit validation figure input changed")
    by_fit = {(row["fit_seed"], row["size_multiplier"]): row
              for row in report["fit_summary"]}
    if set(by_fit) != {(fit, size) for fit in FITS for size in (1, 4)}:
        raise ValueError("News five-fit figure matrix incomplete")
    summary = {row["size_multiplier"]: row for row in report["summary"]}
    if set(summary) != {1, 4}:
        raise ValueError("News five-fit median missing")
    matplotlib.rcParams["svg.hashsalt"] = "dope-news-q10-fivefit-validation-v1"
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    colors = ("#31688e", "#e07b39", "#35a389", "#9c5caa", "#b49b31")
    for fit, color in zip(FITS, colors):
        axis.plot([0, 1],
                  [by_fit[(fit, size)]["median_catboost_retention"]
                   for size in (1, 4)], marker="o", markersize=6,
                  linewidth=1.6, color=color, label=f"Fit seed {fit}")
    axis.plot([0, 1],
              [summary[size]["median_of_fit_medians_catboost_retention"]
               for size in (1, 4)], marker="D", markersize=6,
              linewidth=2.6, linestyle="--", color="#202b38",
              label="Median across fits")
    axis.set_xticks([0, 1], ["n", "4n"])
    axis.set_xlim(-0.14, 1.14)
    axis.set_ylim(0.0, 1.1)
    axis.set_ylabel("CatBoost TSTR/TRTR retention")
    axis.set_title("News q10: five GPU fits on training-derived validation")
    axis.grid(axis="y", color="#d7dce1", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower left", ncol=2, fontsize=8)
    figure.text(0.5, 0.008,
                "Fit seed 37 coincides with the median at both sizes; "
                "three sample seeds per fit and size.\n"
                "Official tests sealed; release gates unverified",
                ha="center", fontsize=8, color="#4a5661")
    figure.tight_layout(rect=[0, 0.08, 1, 1])
    svg, pdf = BytesIO(), BytesIO()
    figure.savefig(svg, format="svg",
                   metadata={"Date": None, "Creator": "DOPE benchmark"})
    figure.savefig(pdf, format="pdf",
                   metadata={"CreationDate": None, "ModDate": None,
                             "Creator": "DOPE benchmark"})
    plt.close(figure)
    svg_text = svg.getvalue().decode("utf-8")
    (RESULTS / "pilot24-news-q10-fivefit.svg").write_text(
        "".join(line.rstrip() + "\n" for line in svg_text.splitlines()))
    (RESULTS / "pilot24-news-q10-fivefit.pdf").write_bytes(pdf.getvalue())
    print("News five-fit validation figure regenerated")


if __name__ == "__main__":
    main()
