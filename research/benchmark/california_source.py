"""Pin the CC BY 4.0 Figshare California Housing source and sklearn mapping."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

from .score import sha256


ARCHIVE_SHA256 = "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"
SOURCE_URL = "https://ndownloader.figshare.com/files/5976036"
LICENSE_URL = "https://figshare.com/articles/dataset/cal_housing_tgz/3829992"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]


def transform(archive: Path, output: Path) -> dict:
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("California archive hash mismatch")
    if output.exists():
        raise FileExistsError("source transformation is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source, output.open("w", newline="") as destination:
        raw = source.extractfile("CaliforniaHousing/cal_housing.data")
        if raw is None:
            raise ValueError("California source table absent")
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow([*FEATURES, "MedHouseVal"])
        count = 0
        for row in reader:
            if len(row) != 9:
                raise ValueError("unexpected California source width")
            values = [float(value) for value in row]
            households = values[6]
            if households <= 0:
                raise ValueError("nonpositive households")
            # Match sklearn.datasets.fetch_california_housing's feature order and ratios.
            transformed = [values[7], values[2], values[3] / households,
                           values[4] / households, values[5], values[5] / households,
                           values[1], values[0], values[8] / 100000.0]
            writer.writerow([format(value, ".17g") for value in transformed])
            count += 1
    if count != 20640:
        raise ValueError("California source row count changed")
    return {"id": "California", "panel": "public_core", "source": SOURCE_URL,
            "source_identity": "figshare:3829992:california-housing",
            "license": {"status": "recorded", "spdx": "CC-BY-4.0", "evidence_url": LICENSE_URL},
            "task": "regression", "target": "MedHouseVal", "header": True, "raw": str(output),
            "source_archive_sha256": ARCHIVE_SHA256, "transformation_sha256": sha256(Path(__file__)),
            "transformed_sha256": sha256(output), "transformed_rows": count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]) or args.output.resolve().is_relative_to(Path("/tmp")):
        parser.error("bulk transformed data must live outside worktrees and /tmp")
    print(json.dumps(transform(args.archive, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
