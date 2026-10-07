"""Plot the authenticated public summary; never read scientific input payloads."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

PANEL_BYTES = 244428
PANEL_SHA256 = "5914fb8aea11e8090cd7cac70a748bccb75cf7a9f1092c25616b78cd1c5bb048"
PANELS = (("TabDDPM", "author_default", "TabDDPM author default", 10),
          ("TabDDPM", "native_selected", "TabDDPM native selected", 12),
          ("Forest-Flow", "author_default", "Forest author default", 11))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_summary(path):
    require(not path.is_symlink(), "input_alias_forbidden")
    raw = path.read_bytes()
    require(len(raw) == PANEL_BYTES and hashlib.sha256(raw).hexdigest() == PANEL_SHA256,
            "public_input_pin_mismatch")
    panel = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_JSON")))
    require(panel["format"] == panel["status"] == "measured_descriptive_validation", "input_scope")
    require(type(panel["fit_seed"]) is int and panel["fit_seed"] == 11 and
            panel["sample_seeds"] == [101, 211, 307], "protocol_scope")
    require(panel["gated_scores"] == dict(mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None), "gated_claim")
    require(panel["artifact_scope"]["comparison_unconstrained_by_stored_byte_cap"] is True, "artifact_scope")
    require(panel["official_tests_opened"] is False and panel["production_certified"] is False,
            "scope_upgrade")
    summary = panel["summary"]
    require(type(summary) is list and len(summary) == 6, "summary_count")
    keys = [(r["peer_method"], r["peer_configuration"], r["size_multiplier"]) for r in summary]
    require(len(set(keys)) == 6 and set(keys) == {(m, c, n) for m, c, _, _ in PANELS for n in (1, 4)},
            "summary_identity")
    for row in summary:
        require(type(row["size_multiplier"]) is int and type(row["paired_complete_informative_lineages"]) is int,
                "summary_integer_identity")
        require(all(finite(row[k]) for k in ("DOPE_median_of_same_cohort_three_sample_means",
                "peer_median_of_same_cohort_three_sample_means", "median_paired_difference")), "summary_nonfinite")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_environment")
    resource.setrlimit(resource.RLIMIT_AS, (805306368, 805306368))
    started = time.monotonic()
    initial_RSS_highwater_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    summary = read_summary(args.panel)
    require(args.output_dir.is_dir() and not args.output_dir.is_symlink(), "output_directory")
    names = ("matched-linear-fit11.svg", "matched-linear-fit11.pdf", "preview.private.png", "FIGURE_RECEIPT.json")
    require(all(not (args.output_dir / n).exists() for n in names), "output_already_exists")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy
    require(matplotlib.__version__ == "3.6.3" and numpy.__version__ == "1.26.4",
            "plotting_environment_version")

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "svg.hashsalt": PANEL_SHA256, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0), sharey=True)
    colors = {"DOPE": "#B45309", "Comparator": "#1D4ED8"}
    for ax, (method, role, title, expected_n) in zip(axes, PANELS):
        rows = sorted((r for r in summary if (r["peer_method"], r["peer_configuration"]) == (method, role)),
                      key=lambda r: r["size_multiplier"])
        require(len(rows) == 2 and all(r["paired_complete_informative_lineages"] == expected_n for r in rows),
                "cohort_count")
        for x, row in enumerate(rows):
            dope = row["DOPE_median_of_same_cohort_three_sample_means"]
            peer = row["peer_median_of_same_cohort_three_sample_means"]
            ax.scatter(x - .13, dope, color=colors["DOPE"], marker="o", s=50,
                       label="DOPE" if x == 0 else None, zorder=3)
            ax.scatter(x + .13, peer, color=colors["Comparator"], marker="s", s=44,
                       label="Comparator" if x == 0 else None, zorder=3)
            ax.annotate(f"{dope:.6f}", (x - .13, dope), xytext=(-3, 9), textcoords="offset points",
                        ha="right", fontsize=8, color=colors["DOPE"])
            ax.annotate(f"{peer:.6f}", (x + .13, peer), xytext=(3, -15), textcoords="offset points",
                        ha="left", fontsize=8, color=colors["Comparator"])
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlim(-.48, 1.48)
        ax.set_ylim(.77, 1.05)
        ax.set_xticks([0, 1], [f"n\npaired N={expected_n}", f"4n\npaired N={expected_n}"])
        ax.grid(axis="y", color="#D1D5DB", linewidth=.6)
        ax.set_axisbelow(True)
        ax.text(.5, -.25, "Median paired difference (DOPE − comparator)\n"
                f"n: {rows[0]['median_paired_difference']:+.6f}    4n: {rows[1]['median_paired_difference']:+.6f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    axes[0].set_ylabel("Median of three-sample mean retention")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .985), ncol=2, frameon=False)
    fig.subplots_adjust(left=.075, right=.99, top=.86, bottom=.35, wspace=.18)
    caption = ("One fit seed (11); three sample seeds; TRAIN-derived VALID; identical paired cohort within each row.\n"
               "Median paired difference is separate from the difference of displayed medians. Stored-byte cap unconstrained;\n"
               "two low-signal Forest lineages excluded. Descriptive only: no superiority, MFS/PTF or production eligibility claim.")
    fig.text(.5, .045, caption, ha="center", va="bottom", fontsize=8.5, linespacing=1.4)
    fixed_date = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
    fig.savefig(args.output_dir / names[0], metadata={"Date": fixed_date.isoformat(), "Creator": "Matplotlib public-scalar renderer"})
    svg_path = args.output_dir / names[0]
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(args.output_dir / names[1], metadata={"CreationDate": fixed_date, "ModDate": fixed_date,
                "Creator": "Matplotlib public-scalar renderer", "Title": "Matched fit11 linear-regression retention"})
    fig.savefig(args.output_dir / names[2], dpi=150)
    plt.close(fig)
    require(read_summary(args.panel) == summary, "public_input_changed")
    source = Path(__file__).read_bytes()
    artifacts = []
    for name in names[:3]:
        raw = (args.output_dir / name).read_bytes()
        artifacts.append(dict(filename=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    reported_RSS_highwater_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    peak_RSS_bytes = (reported_RSS_highwater_bytes
                      if reported_RSS_highwater_bytes > initial_RSS_highwater_bytes else None)
    receipt = dict(format="matched-linear-fit11-public-summary-figure-v1", version=1,
        input_ref=dict(filename="panel.json", bytes=PANEL_BYTES, sha256=PANEL_SHA256),
        summary_scalars=summary, source_ref=dict(filename=Path(__file__).name, bytes=len(source), sha256=hashlib.sha256(source).hexdigest()),
        environment=dict(python=sys.version, executable=sys.executable, matplotlib=matplotlib.__version__,
                         numpy=numpy.__version__, backend=matplotlib.get_backend(), CUDA_VISIBLE_DEVICES="",
                         CPU_affinity=sorted(os.sched_getaffinity(0)), address_space_limit_bytes=805306368),
        artifacts=artifacts, elapsed_seconds=time.monotonic() - started,
        peak_RSS_bytes=peak_RSS_bytes,
        initial_RSS_highwater_bytes=initial_RSS_highwater_bytes,
        reported_RSS_highwater_bytes=reported_RSS_highwater_bytes,
        RSS_highwater_inherited_or_unresolved=peak_RSS_bytes is None,
        rows_models_samples_TEST_or_host_queries=False, scientific_evaluation=False)
    with (args.output_dir / names[3]).open("x") as target:
        json.dump(receipt, target, sort_keys=True, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps(dict(status="ok", panels=3, summary_rows=6, artifact_count=3,
                         elapsed_seconds=receipt["elapsed_seconds"], peak_RSS_bytes=receipt["peak_RSS_bytes"])))


if __name__ == "__main__":
    main()
