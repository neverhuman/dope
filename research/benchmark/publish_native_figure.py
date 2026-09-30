"""Regenerate the validation-only artifact-size comparison figure."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path


RESULTS = Path(__file__).with_name("results")


def counts() -> list[tuple[str, int, int]]:
    compact = json.loads((RESULTS / "compact-native-all-validation.json").read_text())
    copula = json.loads((RESULTS / "copula-native-all-validation.json").read_text())
    if (compact["production_certified"] is not False
            or copula["production_certified"] is not False
            or compact["ptf_v1"] is not None or copula["ptf_v1"] is not None):
        raise ValueError("figure input is not validation-only")
    result = []
    for method, cells in (
        ("Independent marginals", [cell for cell in compact["cells"]
                                   if cell["method"] == "independent_marginals"]),
        ("Chow–Liu", [cell for cell in compact["cells"]
                       if cell["method"] == "Chow-Liu"]),
        ("GaussianCopula", copula["cells"]),
    ):
        if len(cells) != 100 or len({cell["dataset"] for cell in cells}) != 100:
            raise ValueError("artifact figure requires complete S3 lineage set")
        default = {"Independent marginals": {"bins": 16},
                   "Chow–Liu": {"bins": 8, "laplace_alpha": 1.0},
                   "GaussianCopula": {"distribution": "Univariate"}}[method]
        defaults = []
        for cell in cells:
            trial = next((trial for trial in cell["trials"]
                          if all(trial["config"].get(key) == value
                                 for key, value in default.items())), None)
            if trial is None or trial["artifact_bytes"] is None:
                raise ValueError("author default receipt missing")
            defaults.append(trial["artifact_bytes"] <= 10_240)
        result.append((method, sum(defaults), sum(cell["selected_within_l3"]
                                                for cell in cells)))
    return result


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = counts()
    matplotlib.rcParams["svg.hashsalt"] = "dope-native-l3-validation-v1"
    figure, axis = plt.subplots(figsize=(8, 3.8))
    positions = list(range(len(data)))
    axis.barh([position - 0.18 for position in positions],
              [row[1] for row in data], height=0.33, label="Author default",
              color="#597b91")
    axis.barh([position + 0.18 for position in positions],
              [row[2] for row in data], height=0.33, label="Native tuned",
              color="#c06a35")
    axis.set_yticks(positions, [row[0] for row in data])
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Artifacts at or below 10,240 bytes, out of 100 S3 lineages")
    axis.set_title("Validation artifact sizes; no production gates assessed")
    axis.legend(loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    svg, pdf = BytesIO(), BytesIO()
    figure.savefig(svg, format="svg", metadata={"Date": None, "Creator": "DOPE benchmark"})
    figure.savefig(pdf, format="pdf", metadata={"CreationDate": None, "ModDate": None,
                                                "Creator": "DOPE benchmark"})
    plt.close(figure)
    (RESULTS / "native-validation-l3.svg").write_bytes(svg.getvalue())
    (RESULTS / "native-validation-l3.pdf").write_bytes(pdf.getvalue())
    print(data)


if __name__ == "__main__":
    main()
