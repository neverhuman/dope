"""Download UCI Adult (CC BY 4.0) and write headered train.csv and test.csv.

The official adult.test file is the untouched holdout. Values are kept as they
appear in the source, including the "?" missing-value marker.

    python3 examples/adult/prepare.py --out target/adult
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
SHA256 = "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
           "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
           "hours_per_week", "native_country", "income"]


def rows(text: str) -> list[list[str]]:
    out = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(COLUMNS):
            continue  # blank lines and the test file's banner
        cells[-1] = cells[-1].rstrip(".")  # adult.test writes "<=50K."
        out.append(cells)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zip", type=Path, help="use a local copy of adult.zip")
    args = parser.parse_args()
    payload = args.zip.read_bytes() if args.zip else urllib.request.urlopen(URL, timeout=120).read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise SystemExit("adult.zip digest mismatch")
    archive = zipfile.ZipFile(io.BytesIO(payload))
    args.out.mkdir(parents=True, exist_ok=True)
    for source, name in (("adult.data", "train.csv"), ("adult.test", "test.csv")):
        table = rows(archive.read(source).decode("utf-8"))
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(COLUMNS)
            writer.writerows(table)
        print(f"{name}: {len(table)} rows")


if __name__ == "__main__":
    main()
