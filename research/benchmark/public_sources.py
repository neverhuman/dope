"""Reproduce the two rights-cleared TabSyn pilot source tables from UCI ZIPs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

from .manifest import digest, row_digest
from .score import sha256


SOURCES = {
    "Adult": {
        "url": "https://archive.ics.uci.edu/static/public/2/adult.zip",
        "sha256": "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb",
        "license": {"status": "recorded", "spdx": "CC-BY-4.0",
                    "evidence_url": "https://archive.ics.uci.edu/dataset/2/adult"},
        "source_identity": "uci:2:adult",
    },
    "News": {
        "url": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
        "sha256": "dba2ae526f62ccef6f2f8efb53b0268319f4d0d2719bb98d88958cfe31d58a22",
        "license": {"status": "recorded", "spdx": "CC-BY-4.0",
                    "evidence_url": "https://archive.ics.uci.edu/dataset/332/online+news+popularity"},
        "source_identity": "uci:332:online-news-popularity",
    },
}


def transform(name: str, archive: Path, output: Path,
              test_output: Path | None = None) -> dict:
    source = SOURCES[name]
    if sha256(archive) != source["sha256"]:
        raise ValueError("public source ZIP hash mismatch")
    if output.exists() or (test_output is not None and test_output.exists()):
        raise FileExistsError("source transformation is immutable")
    if name == "Adult" and (test_output is None or output.resolve() == test_output.resolve()):
        raise ValueError("Adult official test requires a distinct evaluator output")
    if name != "Adult" and test_output is not None:
        raise ValueError("this public source has no official test file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source_zip:
        if name == "Adult":
            columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                       "marital-status", "occupation", "relationship", "race", "sex",
                       "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
            def rows(part: str) -> list[list[str]]:
                parsed = []
                reader = csv.reader(io.TextIOWrapper(source_zip.open(part), encoding="utf-8"))
                for row in reader:
                    if not row or row[0].startswith("|"):
                        continue
                    row = [cell.strip() for cell in row]
                    if len(row) != len(columns):
                        raise ValueError("unexpected Adult source width")
                    row[-1] = row[-1].rstrip(".")
                    parsed.append(row)
                return parsed
            official_test = rows("adult.test")
            test_groups = {row_digest(row) for row in official_test}
            official_train = rows("adult.data")
            excluded = [row_digest(row) for row in official_train
                        if row_digest(row) in test_groups]
            official_train = [row for row in official_train
                              if row_digest(row) not in test_groups]
            assert test_output is not None
            test_output.parent.mkdir(parents=True, exist_ok=True)
            for path, part_rows in ((output, official_train), (test_output, official_test)):
                with path.open("x", newline="") as target_stream:
                    writer = csv.writer(target_stream, lineterminator="\n")
                    writer.writerow(columns)
                    writer.writerows(part_rows)
            task, target = "binary", "income"
            transformed = {"train": sha256(output), "test": sha256(test_output)}
            counts = {"train": len(official_train), "test": len(official_test)}
        elif name == "News":
            with output.open("x", newline="") as target_stream:
                writer = csv.writer(target_stream, lineterminator="\n")
                reader = csv.reader(io.TextIOWrapper(
                    source_zip.open("OnlineNewsPopularity/OnlineNewsPopularity.csv"), encoding="utf-8"))
                original = [cell.strip() for cell in next(reader)]
                if "url" not in original or "shares" not in original:
                    raise ValueError("unexpected News source columns")
                keep = [index for index, column in enumerate(original) if column != "url"]
                writer.writerow([original[index] for index in keep])
                count = 0
                for row in reader:
                    if len(row) != len(original):
                        raise ValueError("unexpected News source width")
                    writer.writerow([row[index].strip() for index in keep])
                    count += 1
            task, target = "regression", "shares"
        else:
            raise ValueError("unknown public source")
    entry = {"id": name, "panel": "public_core", "source": source["url"],
            "source_identity": source["source_identity"], "license": source["license"],
            "task": task, "target": target, "header": True,
            "source_archive_sha256": source["sha256"],
            "transformation_sha256": sha256(Path(__file__))}
    if name == "Adult":
        entry.update({"raw": {"train": str(output), "test": str(test_output)},
                      "official_split_id": "uci:2:adult:data-test",
                      "official_split_source": source["url"],
                      "transformed_files_sha256": transformed,
                      "transformed_rows": counts,
                      "excluded_official_overlap_rows": len(excluded),
                      "excluded_official_overlap_sha256": digest(sorted(excluded))})
    else:
        entry.update({"raw": str(output), "transformed_sha256": sha256(output),
                      "transformed_rows": count})
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(SOURCES))
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--test-output", type=Path)
    args = parser.parse_args()
    for path in (args.output, args.test_output):
        if path is not None and (path.resolve().is_relative_to(Path(__file__).resolve().parents[2])
                                 or path.resolve().is_relative_to(Path("/tmp"))):
            parser.error("bulk transformed data must live outside worktrees and /tmp")
    print(json.dumps(transform(args.name, args.archive, args.output, args.test_output), sort_keys=True))


if __name__ == "__main__":
    main()
