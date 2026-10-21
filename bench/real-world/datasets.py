"""Real-world benchmark panels for DOPE 1.0.

Every table is a public UCI or PMLB release (CC BY 4.0 unless noted). A
dataset is downloaded once into the cache, its archive digest is checked
against the pin below, and it is written as two canonical CSVs (header row,
target column last): ``train.csv`` (80%) and ``test.csv`` (20%), split by a
fixed seed, stratified for classification. Training rows are capped at
20,000 and test rows at 10,000 so the whole panel runs on a laptop CPU.

    python3 bench/real-world/datasets.py --panel confirm --out target/bench-data
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

UCI = "https://archive.ics.uci.edu/static/public"
PMLB = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68/datasets"
MAX_TRAIN, MAX_TEST, SEED = 20_000, 10_000, 20261021


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    sha256: str | None
    task: str  # regression | classification
    target: str
    load: Callable[[bytes], pd.DataFrame]
    license: str = "CC-BY-4.0"
    citation: str = ""


def _zip(raw: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(raw))


def _member(raw: bytes, suffix: str) -> bytes:
    z = _zip(raw)
    for name in z.namelist():
        if name.endswith(suffix):
            return z.read(name)
        if name.endswith(".zip"):
            try:
                return _member(z.read(name), suffix)
            except KeyError:
                continue
    raise KeyError(suffix)


def adult(raw: bytes) -> pd.DataFrame:
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    df = pd.read_csv(io.BytesIO(_member(raw, "adult.data")), header=None, names=cols, skipinitialspace=True)
    return df.drop(columns=["fnlwgt"])


def bank(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(_member(raw, "bank-full.csv")), sep=";")
    return df.drop(columns=["duration"])  # known post-outcome leak (UCI note)


def abalone(raw: bytes) -> pd.DataFrame:
    cols = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"]
    return pd.read_csv(io.BytesIO(_member(raw, "abalone.data")), header=None, names=cols)


def wine(raw: bytes) -> pd.DataFrame:
    red = pd.read_csv(io.BytesIO(_member(raw, "winequality-red.csv")), sep=";").assign(color="red")
    white = pd.read_csv(io.BytesIO(_member(raw, "winequality-white.csv")), sep=";").assign(color="white")
    df = pd.concat([red, white], ignore_index=True)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df[[c for c in df.columns if c != "quality"] + ["quality"]]


def bike(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(_member(raw, "hour.csv")))
    return df.drop(columns=["instant", "dteday", "casual", "registered"])


def airfoil(raw: bytes) -> pd.DataFrame:
    cols = ["frequency", "angle", "chord", "velocity", "thickness", "sound_pressure"]
    return pd.read_csv(io.BytesIO(_member(raw, "airfoil_self_noise.dat")), sep="\t", header=None, names=cols)


def student(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_member(raw, "student-por.csv")), sep=";")


def car(raw: bytes) -> pd.DataFrame:
    cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    return pd.read_csv(io.BytesIO(_member(raw, "car.data")), header=None, names=cols)


def german(raw: bytes) -> pd.DataFrame:
    cols = ["checking", "duration", "history", "purpose", "amount", "savings", "employment", "rate", "status_sex",
            "debtors", "residence", "property", "age", "plans", "housing", "credits", "job", "liable", "phone",
            "foreign", "risk"]
    return pd.read_csv(io.BytesIO(_member(raw, "german.data")), sep=" ", header=None, names=cols)


def concrete(raw: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(_member(raw, "Concrete_Data.xls")))
    df.columns = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse", "fine", "age", "strength"]
    return df


def energy(raw: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(_member(raw, "ENB2012_data.xlsx")))
    df = df.dropna(how="all").drop(columns=["Y2"])
    df.columns = ["compactness", "surface", "wall", "roof", "height", "orientation", "glazing", "glazing_dist", "heating_load"]
    return df


def credit(raw: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(_member(raw, ".xls")), header=1)
    df = df.drop(columns=["ID"]).rename(columns={"default payment next month": "default"})
    for c in ("SEX", "EDUCATION", "MARRIAGE"):
        df[c] = df[c].astype(str)
    return df


def news(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(_member(raw, "OnlineNewsPopularity.csv")), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df.drop(columns=["url", "timedelta"])


def covertype(raw: bytes) -> pd.DataFrame:
    import gzip
    data = gzip.decompress(_member(raw, "covtype.data.gz"))
    df = pd.read_csv(io.BytesIO(data), header=None)
    names = ["elevation", "aspect", "slope", "h_hydro", "v_hydro", "h_road", "shade_9am", "shade_noon", "shade_3pm", "h_fire"]
    wild = df.iloc[:, 10:14].values.argmax(1)
    soil = df.iloc[:, 14:54].values.argmax(1)
    out = df.iloc[:, :10].copy()
    out.columns = names
    out["wilderness"] = [f"W{v}" for v in wild]
    out["soil"] = [f"S{v}" for v in soil]
    out["cover"] = df.iloc[:, 54].astype(str)
    return out


def mushroom(raw: bytes) -> pd.DataFrame:
    cols = ["edible", "cap_shape", "cap_surface", "cap_color", "bruises", "odor", "gill_attachment", "gill_spacing",
            "gill_size", "gill_color", "stalk_shape", "stalk_root", "stalk_surface_above", "stalk_surface_below",
            "stalk_color_above", "stalk_color_below", "veil_type", "veil_color", "ring_number", "ring_type",
            "spore_print_color", "population", "habitat"]
    df = pd.read_csv(io.BytesIO(_member(raw, "agaricus-lepiota.data")), header=None, names=cols)
    return df[[c for c in cols if c != "edible"] + ["edible"]]


def pmlb(name: str, target: str = "target") -> Callable[[bytes], pd.DataFrame]:
    def load(raw: bytes) -> pd.DataFrame:
        import gzip
        df = pd.read_csv(io.BytesIO(gzip.decompress(raw)), sep="\t")
        return df.rename(columns={"target": target})
    load.__name__ = name
    return load


CONFIRM = [
    Dataset("adult", f"{UCI}/2/adult.zip", None, "classification", "income", adult, citation="Becker & Kohavi 1996"),
    Dataset("bank", f"{UCI}/222/bank+marketing.zip", None, "classification", "y", bank, citation="Moro et al. 2014"),
    Dataset("credit", f"{UCI}/350/default+of+credit+card+clients.zip", None, "classification", "default", credit, citation="Yeh 2009"),
    Dataset("german", f"{UCI}/144/statlog+german+credit+data.zip", None, "classification", "risk", german, citation="Hofmann 1994"),
    Dataset("car", f"{UCI}/19/car+evaluation.zip", None, "classification", "class", car, citation="Bohanec 1988"),
    Dataset("mushroom", f"{UCI}/73/mushroom.zip", None, "classification", "edible", mushroom, citation="Schlimmer 1981"),
    Dataset("covertype", f"{UCI}/31/covertype.zip", None, "classification", "cover", covertype, citation="Blackard 1998"),
    Dataset("california", f"{PMLB}/537_houses/537_houses.tsv.gz", None, "regression", "median_house_value",
            pmlb("california", "median_house_value"), license="public domain (StatLib)", citation="Pace & Barry 1997"),
    Dataset("abalone", f"{UCI}/1/abalone.zip", None, "regression", "rings", abalone, citation="Nash et al. 1994"),
    Dataset("wine", f"{UCI}/186/wine+quality.zip", None, "regression", "quality", wine, citation="Cortez et al. 2009"),
    Dataset("bike", f"{UCI}/275/bike+sharing+dataset.zip", None, "regression", "cnt", bike, citation="Fanaee-T 2013"),
    Dataset("concrete", f"{UCI}/165/concrete+compressive+strength.zip", None, "regression", "strength", concrete, citation="Yeh 1998"),
    Dataset("airfoil", f"{UCI}/291/airfoil+self+noise.zip", None, "regression", "sound_pressure", airfoil, citation="Brooks et al. 1989"),
    Dataset("energy", f"{UCI}/242/energy+efficiency.zip", None, "regression", "heating_load", energy, citation="Tsanas & Xifara 2012"),
    Dataset("student", f"{UCI}/320/student+performance.zip", None, "regression", "G3", student, citation="Cortez & Silva 2008"),
    Dataset("news", f"{UCI}/332/online+news+popularity.zip", None, "regression", "shares", news, citation="Fernandes et al. 2015"),
]

DEV = [Dataset(n, f"{PMLB}/{n}/{n}.tsv.gz", None, "regression", "target", pmlb(n), license="see PMLB")
       for n in ["197_cpu_act", "201_pol", "218_house_8L", "225_puma8NH", "227_cpu_small", "344_mv", "503_wind",
                 "522_pm10", "529_pollen", "547_no2", "560_bodyfat", "574_house_16H", "666_rmftsa_ladata",
                 "678_visualizing_environmental", "690_visualizing_galaxy", "4544_GeographicalOriginalofMusic"]]

PANELS = {"confirm": CONFIRM, "dev": DEV}
PINS = Path(__file__).with_name("pins.json")


def fetch(ds: Dataset, cache: Path) -> bytes:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (ds.name + ".raw")
    if not path.exists():
        req = urllib.request.Request(ds.url, headers={"User-Agent": "dope-bench"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    raw = path.read_bytes()
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    digest = hashlib.sha256(raw).hexdigest()
    if ds.name in pins and pins[ds.name] != digest:
        raise ValueError(f"{ds.name}: archive digest differs from the pin")
    return raw


def split(df: pd.DataFrame, ds: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna(subset=[ds.target]).reset_index(drop=True)
    df = df[[c for c in df.columns if c != ds.target] + [ds.target]]
    rng = np.random.default_rng(SEED)
    if ds.task == "classification":
        test_idx = []
        for _, g in df.groupby(ds.target):
            k = int(round(0.2 * len(g)))
            test_idx.extend(rng.choice(g.index.values, size=k, replace=False))
        test_mask = df.index.isin(test_idx)
    else:
        test_mask = np.zeros(len(df), bool)
        test_mask[rng.permutation(len(df))[: int(round(0.2 * len(df)))]] = True
    train, test = df[~test_mask], df[test_mask]
    train = train.sample(n=min(MAX_TRAIN, len(train)), random_state=SEED)
    test = test.sample(n=min(MAX_TEST, len(test)), random_state=SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def prepare(panel: str, out: Path, only: str = "") -> list[dict]:
    rows = []
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    for ds in PANELS[panel]:
        if only and ds.name not in only.split(","):
            continue
        raw = fetch(ds, out / "cache")
        pins.setdefault(ds.name, hashlib.sha256(raw).hexdigest())
        train, test = split(ds.load(raw), ds)
        d = out / panel / ds.name
        d.mkdir(parents=True, exist_ok=True)
        train.to_csv(d / "train.csv", index=False)
        test.to_csv(d / "test.csv", index=False)
        meta = {"name": ds.name, "task": ds.task, "target": ds.target, "train_rows": len(train), "test_rows": len(test),
                "features": train.shape[1] - 1, "categorical": int((train.drop(columns=[ds.target]).dtypes == object).sum()),
                "license": ds.license, "source": ds.url, "citation": ds.citation}
        if ds.task == "classification":
            meta["classes"] = int(pd.concat([train[ds.target], test[ds.target]]).astype(str).nunique())
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        rows.append(meta)
        print(f"{ds.name:<12} {ds.task:<15} train {len(train):>6} test {len(test):>6} features {meta['features']:>3}")
    PINS.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", choices=sorted(PANELS), default="confirm")
    ap.add_argument("--out", type=Path, default=Path("target/bench-data"))
    ap.add_argument("--only", default="", help="comma list of dataset names")
    a = ap.parse_args()
    prepare(a.panel, a.out, a.only)
