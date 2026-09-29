"""Reproduce the two rights-cleared TabSyn pilot source tables from UCI ZIPs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

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


def transform(name: str, archive: Path, output: Path) -> dict:
    source = SOURCES[name]
    if sha256(archive) != source["sha256"]:
        raise ValueError("public source ZIP hash mismatch")
    if output.exists():
        raise FileExistsError("source transformation is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source_zip, output.open("w", newline="") as target_stream:
        writer = csv.writer(target_stream, lineterminator="\n")
        if name == "Adult":
            columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                       "marital-status", "occupation", "relationship", "race", "sex",
                       "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
            writer.writerow(columns)
            count = 0
            for part in ("adult.data", "adult.test"):
                reader = csv.reader(io.TextIOWrapper(source_zip.open(part), encoding="utf-8"))
                for row in reader:
                    if not row or row[0].startswith("|"):
                        continue
                    row = [cell.strip() for cell in row]
                    if len(row) != len(columns):
                        raise ValueError("unexpected Adult source width")
                    row[-1] = row[-1].rstrip(".")
                    writer.writerow(row)
                    count += 1
            task, target = "binary", "income"
        elif name == "News":
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
    return {"id": name, "panel": "public_core", "source": source["url"],
            "source_identity": source["source_identity"], "license": source["license"],
            "task": task, "target": target, "header": True, "raw": str(output),
            "source_archive_sha256": source["sha256"], "transformation_sha256": sha256(Path(__file__)),
            "transformed_sha256": sha256(output), "transformed_rows": count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(SOURCES))
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]) or args.output.resolve().is_relative_to(Path("/tmp")):
        parser.error("bulk transformed data must live outside worktrees and /tmp")
    print(json.dumps(transform(args.name, args.archive, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
