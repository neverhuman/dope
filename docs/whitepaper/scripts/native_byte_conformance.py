#!/usr/bin/env python3
"""Byte accounting of one native artifact, pinned once and checked in README and supplement.

The pin is a source-free projection of the installed public demo's research receipt:
the same DOPERAW1 file measured by compile, inspect, the receipt and its file size.
``--project RECEIPT --evidence HANDOFF`` derives it from those measured records and
``--write`` renders the supplement paragraph from it into generated/; the default
mode checks the pin, the generated paragraph and the README sentences against each
other. It is an engineering conformance example, not a benchmark result, and
carries no row, header, extremum, fitted value or private path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
PIN = PAPER / "receipts" / "native-byte-conformance-v1.json"
GENERATED = PAPER / "generated" / "native-byte-conformance.tex"
SUBSECTION = "\\subsection{One native artifact, accounted end to end}"
FORMAT = "dope-native-byte-conformance"
PARTS = ("framing_bytes", "transform_schema_bytes", "native_payload_bytes")
SAME_OBJECT = ("complete_sampling_bytes", "transport_archive_bytes", *PARTS)


class ConformanceError(ValueError):
    """The measured records disagree, or a published sentence drifted from the pin."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConformanceError(message)


def project(receipt: dict, evidence: dict) -> dict:
    """Select the public fields of one measured receipt; refuse any disagreement among its records."""
    inspected = receipt["inspection"]["accounting"]
    replays = [arm["source_free_replay"]["inspection"]["accounting"] for arm in receipt["arms"]]
    complete = inspected["complete_sampling_bytes"]
    _require(receipt["artifact_sha256"] == evidence["artifact_sha256"], "artifact hash differs between receipt and evidence")
    _require(inspected["byte_profile_id"] == evidence["byte_profile_id"], "byte profile differs between receipt and evidence")
    _require(receipt["inspection"]["profile"] == evidence["profile"], "artifact profile differs between receipt and evidence")
    _require(evidence["artifact_file_bytes"] == complete, "file size differs from the complete sampling total")
    _require(complete == sum(inspected[part] for part in PARTS), "framing, schema and payload do not sum to the total")
    _require(inspected["transport_archive_bytes"] == complete, "the inline file is not its own transport archive")
    _require(all(arm["sampling"]["complete_sampling_bytes"] == complete for arm in receipt["arms"]),
             "a sampling arm reports another total")
    for replay in replays:
        _require(all(replay[key] == inspected[key] for key in (*SAME_OBJECT, "byte_profile_id")),
                 "a source-free replay inspected a different object")
        _require(replay["generic_runtime_bytes"] == replays[0]["generic_runtime_bytes"]
                 and replay["cold_sampling_closure_bytes"] == replay["generic_runtime_bytes"] + complete,
                 "runtime and cold closure are not one measured runtime plus the artifact")
    runtime = replays[0]["generic_runtime_bytes"]
    return {
        "format": FORMAT,
        "version": 1,
        "kind": "engineering_conformance_example",
        "scope": "one public grouped development split and one diagnostic learner; not a benchmark result, "
                 "release, or privacy, driver or geometry measurement",
        "artifact": {
            "sha256": receipt["artifact_sha256"],
            "container": "DOPERAW1",
            "format": receipt["inspection"]["format"],
            "format_version": receipt["inspection"]["version"],
            "profile": receipt["inspection"]["profile"],
            "byte_profile_id": inspected["byte_profile_id"],
            "file_bytes": evidence["artifact_file_bytes"],
        },
        "accounting": {
            **{key: inspected[key] for key in SAME_OBJECT},
            "generic_runtime_bytes": runtime,
            "cold_sampling_closure_bytes": replays[0]["cold_sampling_closure_bytes"],
        },
        "agreement": {
            "inspect_equals_compiled_arms": True,
            "file_size_equals_complete": True,
            "source_free_replays": [arm["multiplier"] for arm in evidence["source_free_replays"]],
        },
        "runtime": {
            "scope": evidence["runtime_scope"],
            "binary_sha256": evidence["measured_binary_sha256"],
            "binary_file_bytes": evidence["measured_binary_file_bytes"],
        },
        "unavailable": {
            "portable_bundle_bytes": "not measured; the inline file is the only transport object",
            "release_build_runtime_bytes": "not measured; the runtime above is a debug build",
        },
        "measured_run": {
            "source_commit": evidence["measured_source"],
            "source_tree": evidence["measured_tree"],
            "runner": "examples/public-regression/run.py with the installed dope-kernel",
            "dependency_versions": evidence["actual_dependency_versions"],
        },
        "evidence_sha256": {
            "research_receipt": evidence["saved_research_receipt_sha256"],
            "report_html": evidence["saved_report_html_sha256"],
            "inspect_record": evidence["compile_inspect_cli_diagnostics"]["native-inspection.json"]["sha256"],
            "inspect_stdout": evidence["compile_inspect_cli_diagnostics"]["native-inspection.json"]["stdout_sha256"],
            "fit_record": evidence["compile_inspect_cli_diagnostics"]["native-fit.json"]["sha256"],
        },
        "source_data": {
            "name": "Airfoil Self-Noise",
            "attribution": evidence["source"]["attribution"],
            "doi": evidence["source"]["doi"],
            "license": evidence["source"]["license"],
            "sha256": evidence["source"]["sha256"],
            "bytes": evidence["source"]["bytes"],
            "rows": receipt["source_rows"],
            "fit_rows": receipt["fit_rows"],
            "query_rows": receipt["query_rows"],
            "fit_groups": receipt["fit_groups"],
            "query_groups": receipt["query_groups"],
            "split_seed": receipt["seed"],
            "split_sha256": receipt["split_sha256"],
        },
        "distinct_objects": "The paper's benchmark charge (encoded .dpk plus projection.json) and the 196-byte README toy "
                            "are different objects and do not share these totals.",
    }


def check_pin(pin: dict) -> None:
    accounting = pin["accounting"]
    _require(pin["format"] == FORMAT and pin["version"] == 1, "pin format is not registered")
    complete = accounting["complete_sampling_bytes"]
    _require(pin["artifact"]["file_bytes"] == complete == accounting["transport_archive_bytes"]
             == sum(accounting[part] for part in PARTS), "pinned totals disagree")
    _require(accounting["cold_sampling_closure_bytes"] == accounting["generic_runtime_bytes"] + complete,
             "pinned cold closure is not runtime plus artifact")
    text = json.dumps(pin)
    _require("/home/" not in text and "/mnt/" not in text, "pin carries a private path")


def readme_number(value: int) -> str:
    return f"{value:,}"


def tex_number(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def _section(text: str, start: str, stop: str) -> str:
    _require(start in text, f"section {start!r} is missing")
    return text.split(start, 1)[1].split(stop, 1)[0]


def readme_fragments(pin: dict) -> list[str]:
    """The exact fragments the README paragraph must carry for this artifact."""
    accounting, artifact = pin["accounting"], pin["artifact"]
    totals = ("complete_sampling_bytes", "framing_bytes", "transform_schema_bytes", "native_payload_bytes",
              "generic_runtime_bytes", "cold_sampling_closure_bytes")
    return [*(f"{readme_number(accounting[key])} bytes" for key in totals),
            f"`{artifact['profile']}`", f"`{artifact['byte_profile_id']}`", artifact["sha256"][:12],
            pin["measured_run"]["source_commit"][:7]]


def supplement_paragraph(pin: dict) -> str:
    """The supplement paragraph, rendered from the pin so no number is typed by hand."""
    accounting, artifact, source = pin["accounting"], pin["artifact"], pin["source_data"]
    number = tex_number
    return (
        "% Generated by docs/whitepaper/scripts/native_byte_conformance.py --write from "
        "docs/whitepaper/receipts/native-byte-conformance-v1.json. Do not edit.\n"
        "This subsection is an engineering conformance example, not a benchmark result. The installed public demo "
        "(\\path{examples/public-regression}) fits the UCI Airfoil Self-Noise table (Brooks, Pope, and Marcolini; "
        f"CC~BY~{source['license'].removeprefix('CC-BY-')}; DOI {source['doi']}) on a grouped development split "
        f"(${number(source['fit_rows'])}$ fit rows and ${number(source['query_rows'])}$ query rows, seed "
        f"${source['split_seed']}$) and writes one native raw-unit artifact, a \\texttt{{{artifact['container']}}} "
        f"file (SHA-256 \\texttt{{{artifact['sha256'][:12]}\\ldots}}) with profile "
        f"\\texttt{{{artifact['profile']}}} under the byte profile \\texttt{{{artifact['byte_profile_id']}}}. "
        f"Compile, \\texttt{{inspect}}, the research receipt, and the file size all report "
        f"{number(accounting['complete_sampling_bytes'])} bytes: {number(accounting['framing_bytes'])} bytes of "
        f"native framing, {number(accounting['transform_schema_bytes'])} bytes of fitted schema and transforms, and "
        f"{number(accounting['native_payload_bytes'])} bytes of sampling payload. The file is its own transport "
        "archive, so every dataset-specific dependency of sampling is charged once inside it. The generic runtime "
        "is measured separately: the debug binary and its mapped libraries came to "
        f"{number(accounting['generic_runtime_bytes'])} bytes, so the cold sampling closure was "
        f"{number(accounting['cold_sampling_closure_bytes'])} bytes. No portable bundle or release-build runtime was "
        f"measured. The run used commit \\texttt{{{pin['measured_run']['source_commit'][:7]}}}, and privacy, driver, "
        "and geometry checks are unmeasured for this artifact. It is a different object from the benchmark charge "
        "(the encoded \\texttt{.dpk} file plus \\texttt{projection.json}) and from the README toy, and no number "
        "in the panel comes from it. \\path{docs/whitepaper/receipts/native-byte-conformance-v1.json} pins these "
        "values.\n"
    )


def text_failures(pin: dict, readme: str, supplement: str, generated: str) -> list[str]:
    section = _section(readme, "### One measured native artifact", "\n## ")
    failures = [f"README.md: missing {fragment!r}" for fragment in readme_fragments(pin) if fragment not in section]
    if generated != supplement_paragraph(pin):
        failures.append("generated/native-byte-conformance.tex differs from the pin; run --write")
    inputs = f"{SUBSECTION}\n\\label{{app:native-bytes}}\n\\input{{generated/native-byte-conformance.tex}}\n"
    if inputs not in supplement:
        failures.append("supplement.tex does not input the generated paragraph under its subsection")
    return failures


def check(readme_path: Path = REPO / "README.md", supplement_path: Path = PAPER / "supplement.tex") -> dict:
    pin = json.loads(PIN.read_text())
    check_pin(pin)
    failures = text_failures(pin, readme_path.read_text(), supplement_path.read_text(), GENERATED.read_text())
    if failures:
        raise ConformanceError("; ".join(failures))
    return pin


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", type=Path, metavar="RECEIPT", help="derive the pin from this research receipt")
    parser.add_argument("--evidence", type=Path, metavar="HANDOFF", help="C1's exact byte-accounting evidence JSON")
    parser.add_argument("--write", action="store_true", help="render the supplement paragraph from the pin")
    args = parser.parse_args(argv)
    if args.write:
        pin = json.loads(PIN.read_text())
        check_pin(pin)
        GENERATED.write_text(supplement_paragraph(pin))
        print("wrote", GENERATED.relative_to(REPO))
        return 0
    if args.project is not None:
        if args.evidence is None:
            parser.error("--project needs --evidence")
        pin = project(json.loads(args.project.read_text()), json.loads(args.evidence.read_text()))
        check_pin(pin)
        sys.stdout.write(json.dumps(pin, indent=2, sort_keys=True) + "\n")
        return 0
    pin = check()
    print("native byte conformance matches:", pin["artifact"]["byte_profile_id"],
          pin["accounting"]["complete_sampling_bytes"], "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
