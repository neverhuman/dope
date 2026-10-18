#!/usr/bin/env python3
"""Cross-system public bike-demand observation; no privacy or release claim."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from controls import (compressed_row_artifact, compressed_row_sample,
                      gaussian_copula_artifact, gaussian_copula_sample)
import zipfile

SEED = 1729
CAPS = (16384, 32768)
FEATURES = ("hour", "weekday", "month", "holiday", "workingday", "temperature_c",
            "humidity_pct", "windspeed_m_s")
ARCHIVES = {
    "capital": ("capital-bike.zip", "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
                "hour.csv", "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"),
    "seoul": ("seoul-bike.zip", "139e9908f0a3544bb222386855c9ce107e96467306bb8e4ce936aab59e7baac4",
              "SeoulBikeData.csv", "373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60"),
}
SOURCE_SIZES = {"capital": {"archive": 279992, "member": 1156736},
                "seoul": {"archive": 604298, "member": 604166}}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("source_numeric_invalid")
    return result


def verify_archive(path: Path, name: str) -> bytes:
    filename, expected, member, member_sha256 = ARCHIVES[name]
    if path.name != filename or path.is_symlink() or not path.is_file():
        raise ValueError("source_carrier_invalid")
    raw = path.read_bytes()
    if len(raw) != SOURCE_SIZES[name]["archive"] or sha(raw) != expected:
        raise ValueError("source_digest_mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if names.count(member) != 1 or any(Path(n).name != n for n in names):
            raise ValueError("source_archive_members_invalid")
        content = archive.read(member)
        if len(content) != SOURCE_SIZES[name]["member"] or sha(content) != member_sha256:
            raise ValueError("source_member_digest_mismatch")
        return content


def read_table(data: bytes, expected_rows: int, delimiter: str = ",", encoding: str = "utf-8-sig") -> list[dict]:
    text = data.decode(encoding)
    if delimiter == "auto":
        header = text.splitlines()[0]
        delimiter = ";" if header.count(";") > header.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    if len(rows) != expected_rows or not rows or not rows[0]:
        raise ValueError("source_table_shape_invalid")
    return rows


def capital_rows(rows: list[dict]) -> list[dict]:
    prepared = []
    for row in rows:
        if row["yr"] != "0":
            continue
        date = row["dteday"]
        month = int(date[5:7])
        prepared.append({"entity": date, "year": int(date[:4]), "month": month,
                         "x": [clean_number(row["hr"]), float(__import__("datetime").date.fromisoformat(date).weekday()),
                               float(month), clean_number(row["holiday"]), clean_number(row["workingday"]),
                               clean_number(row["temp"]) * 47.0 - 8.0, clean_number(row["hum"]) * 100.0,
                               clean_number(row["windspeed"]) * 67.0 / 3.6],
                         "y": clean_number(row["cnt"])})
    if len(prepared) < 5000:
        raise ValueError("capital_rows_invalid")
    return prepared


def seoul_rows(rows: list[dict]) -> tuple[list[dict], int]:
    prepared, excluded = [], 0
    for row in rows:
        if row["Functioning Day"].strip().lower() != "yes":
            excluded += 1
            continue
        date = __import__("datetime").datetime.strptime(row["Date"], "%d/%m/%Y").date()
        month = date.month
        holiday = float(row["Holiday"].strip().lower() == "holiday")
        working = float(date.weekday() < 5 and not holiday)
        prepared.append({"entity": date.isoformat(), "year": date.year, "month": month,
                         "x": [clean_number(row["Hour"]), float(date.weekday()), float(month), holiday, working,
                               clean_number(row["Temperature(°C)"]), clean_number(row["Humidity(%)"]),
                               clean_number(row["Wind speed (m/s)"])],
                         "y": clean_number(row["Rented Bike Count"])})
    if len(prepared) < 5000:
        raise ValueError("seoul_rows_invalid")
    return prepared, excluded


def split_dates(rows: list[dict], proportions: tuple[float, ...]) -> tuple[list[dict], ...]:
    """Split ordered whole-day entities; no service date crosses a boundary."""
    days = sorted({row["entity"] for row in rows})
    if len(proportions) < 2 or not math.isclose(sum(proportions), 1.0, abs_tol=1e-12):
        raise ValueError("split_policy_invalid")
    cuts, total = [], 0.0
    for fraction in proportions[:-1]:
        total += fraction
        cuts.append(round(len(days) * total))
    day_sets, start = [], 0
    for end in cuts + [len(days)]:
        day_sets.append(set(days[start:end])); start = end
    partitions = tuple([row for row in rows if row["entity"] in selected] for selected in day_sets)
    if any(not part for part in partitions):
        raise ValueError("split_partition_empty")
    if any(day_sets[i] & day_sets[j] for i in range(len(day_sets)) for j in range(i + 1, len(day_sets))):
        raise ValueError("split_entity_overlap")
    return partitions


def schema() -> dict:
    return {"format": "dope-table-schema", "version": 1, "metadata_visibility": "approved_semantic_labels",
            "columns": [{"id": f"c{i + 1:04d}", "source_name": name, "role": "feature",
                         "dtype": "continuous", "units": unit, "semantic_label": name}
                        for i, (name, unit) in enumerate(zip(FEATURES,
                         ("hour", "weekday", "month", "boolean", "boolean", "Celsius", "%", "m/s")))]
            + [{"id": "c0009", "source_name": "rental_count", "role": "target", "dtype": "continuous",
                "units": "bikes_per_hour", "semantic_label": "hourly rental count"}]}


def write_csv(path: Path, rows: list[dict], header: bool) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if header:
            writer.writerow([*FEATURES, "rental_count"])
        for row in rows:
            writer.writerow([*row["x"], row["y"]])


def native(binary: Path, args: list[str], timeout: int = 90) -> dict:
    if args[0] not in {"doctor", "compile-raw", "inspect-raw", "sample-raw", "sample-raw-isolated"}:
        raise ValueError("native_command_unadmitted")
    env = {} if args[0] == "sample-raw-isolated" else {"RAYON_NUM_THREADS": "2"}
    try:
        run = subprocess.run([str(binary), *args], env=env, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("native_command_failed") from None
    if run.returncode:
        raise ValueError("native_command_failed")
    try:
        return json.loads(run.stdout)
    except (ValueError, UnicodeDecodeError):
        raise ValueError("native_response_invalid") from None


def losses_by_day(truth, prediction, days):
    import numpy as np
    result = []
    for day in sorted(set(days)):
        take = np.asarray([value == day for value in days])
        result.append(float(np.mean((truth[take] - prediction[take]) ** 2)))
    return result


def squared_losses(truth, prediction):
    import numpy as np
    return np.square(truth - prediction).astype(np.float64).tolist()


def gain_lcb(null_losses, real_losses, days, seed=SEED, replicates=2000):
    import numpy as np
    rng = np.random.default_rng(seed)
    null, real = np.asarray(null_losses), np.asarray(real_losses)
    groups = [np.flatnonzero(np.asarray(days) == day) for day in sorted(set(days))]
    draws = rng.integers(0, len(groups), size=(replicates, len(groups)))
    gains = []
    for draw in draws:
        indices = np.concatenate([groups[index] for index in draw])
        gains.append(float(np.mean(null[indices] - real[indices])))
    return float(np.quantile(gains, 0.05, method="lower"))


def run(capital_zip: Path, seoul_zip: Path, binary: Path, output: Path) -> dict:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from research.transfer.tdl import aggregate, drivers, transfer
    started = time.monotonic()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(Path("/tmp")):
        raise ValueError("output_must_be_new_persistent_directory")
    output.mkdir(parents=True)
    stages = []

    def step(name, operation):
        tick = time.monotonic()
        record = {"stage": name, "state": "started", "elapsed_seconds": None}
        stages.append(record)
        try:
            value = operation()
            record["state"] = "completed"
            return value
        except Exception:
            record["state"] = "failed"
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - tick
            with (output / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    cap = step("capital_source_verify_and_parse", lambda: capital_rows(
        read_table(verify_archive(capital_zip, "capital"), 17379)))
    seoul_raw = step("seoul_source_verify_and_parse", lambda: read_table(
        verify_archive(seoul_zip, "seoul"), 8760, "auto", "cp1252"))
    seoul, excluded_nonoperating = seoul_rows(seoul_raw)
    cap_fit, cap_select = split_dates(cap, (.75, .25))
    seoul_fit, seoul_select, seoul_test = split_dates(seoul, (.50, .20, .30))
    if set(row["entity"] for row in cap_fit + cap_select) & set(row["entity"] for row in seoul_fit + seoul_select + seoul_test):
        raise ValueError("cross_source_entity_identity_collision")
    (output / "fit.csv").write_text("", encoding="utf-8")
    write_csv(output / "fit.csv", cap_fit, True)
    (output / "schema.json").write_text(json.dumps(schema(), sort_keys=True) + "\n", encoding="utf-8")
    (output / "split.json").write_text(json.dumps({"format": "bikeshare-source-time-split-v1",
        "source_holdout": ["capital_bikeshare_fit", "seoul_bikeshare_confirmation"],
        "entity_unit": "service_date", "capital_fit_rows": len(cap_fit), "capital_select_rows": len(cap_select),
        "seoul_fit_rows": len(seoul_fit), "seoul_select_rows": len(seoul_select), "seoul_test_rows": len(seoul_test),
        "capital_fit_days": len(set(r["entity"] for r in cap_fit)),
        "capital_select_days": len(set(r["entity"] for r in cap_select)),
        "seoul_fit_days": len(set(r["entity"] for r in seoul_fit)),
        "seoul_select_days": len(set(r["entity"] for r in seoul_select)),
        "seoul_test_days": len(set(r["entity"] for r in seoul_test)),
        "seoul_nonoperating_rows_excluded": excluded_nonoperating,
        "selection_rows_accessed": 0}, sort_keys=True) + "\n", encoding="utf-8")
    binary = binary.resolve(strict=True)
    doctor = step("native_doctor", lambda: native(binary, ["doctor"]))
    if doctor.get("implementation") != "native_rust" or doctor.get("executable_sha256") != sha(binary.read_bytes()):
        raise ValueError("native_binary_identity_invalid")
    X_real = np.asarray([r["x"] for r in seoul_fit], dtype=np.float64)
    y_real = np.asarray([r["y"] for r in seoul_fit], dtype=np.float64)
    X_test = np.asarray([r["x"] for r in seoul_test], dtype=np.float64)
    y_test = np.asarray([r["y"] for r in seoul_test], dtype=np.float64)
    test_days = [r["entity"] for r in seoul_test]

    def model():
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="cholesky"))

    with threadpool_limits(limits=1):
        real_model = step("real_source_reference_fit", lambda: model().fit(X_real, y_real))
        real_pred = step("real_source_reference_predict", lambda: real_model.predict(X_test))
    null_pred = np.full(len(y_test), float(np.mean(y_real)))
    real_day_loss = losses_by_day(y_test, real_pred, test_days)
    null_day_loss = losses_by_day(y_test, null_pred, test_days)
    null_row_loss, real_row_loss = squared_losses(y_test, null_pred), squared_losses(y_test, real_pred)
    real_gain_lcb = step("real_null_cluster_bootstrap", lambda: gain_lcb(null_row_loss, real_row_loss, test_days))
    arms, transfer_rows = [], []
    def model_signature(fitted):
        baseline = float(np.mean((y_test - fitted.predict(X_test)) ** 2))
        importance = []
        for column in range(X_test.shape[1]):
            permuted = X_test.copy()
            order = np.random.default_rng(SEED + 100 + column).permutation(len(permuted))
            permuted[:, column] = X_test[order, column]
            importance.append(max(0.0, float(np.mean((y_test - fitted.predict(permuted)) ** 2)) - baseline))
        return (np.asarray(importance, dtype=np.float64),
                np.asarray(fitted[-1].coef_, dtype=np.float64))

    real_importance, real_sign = model_signature(real_model)
    # Real-real ceilings use three date-block bootstrap contexts, with all operations fixed above.
    import datetime
    train_days = sorted(set(r["entity"] for r in seoul_fit))
    rng = np.random.default_rng(SEED)
    real_signatures = []
    for repeat in range(3):
        chosen = rng.choice(train_days, size=len(train_days), replace=True)
        counts = {day: int(np.sum(chosen == day)) for day in set(chosen)}
        indices = [i for i, row in enumerate(seoul_fit) for _ in range(counts.get(row["entity"], 0))]
        with threadpool_limits(limits=1):
            refit = step(f"real_real_driver_fit_{repeat}", lambda: model().fit(X_real[indices], y_real[indices]))
        real_signatures.append(model_signature(refit))
    ceiling = step("real_real_driver_ceiling", lambda: {
        "mass": min(drivers.mass_agreement(real_importance.tolist(), sig[0].tolist()) for sig in real_signatures),
        "rank": min(drivers.rank_agreement(real_importance.tolist(), sig[0].tolist()) for sig in real_signatures),
        "sign": min(float(np.mean(np.sign(real_sign) == np.sign(sig[1]))) for sig in real_signatures)})

    def evaluate_control(method_name, cap_bytes, multiplier, synthetic, artifact):
        count_rows=len(seoul_fit)*multiplier
        with threadpool_limits(limits=1):
            fitted=step(f"{method_name}_ridge_fit_{cap_bytes}_{multiplier}n", lambda: model().fit(synthetic[:,:8],synthetic[:,8]))
            prediction=fitted.predict(X_test)
        transfer_report=transfer.evaluate({"null_losses":null_row_loss,"real_losses":real_row_loss,
            "synthetic_losses":squared_losses(y_test,prediction),"real_gain_lcb":real_gain_lcb,
            "constant_target":bool(np.ptp(y_real)==0)})
        importance,sign=model_signature(fitted)
        driver_report=drivers.evaluate({"required_components":["mass","rank","sign"],
            "real_importances":real_importance.tolist(),"synthetic_importances":importance.tolist(),
            "real_signs":np.sign(real_sign).astype(int).tolist(),
            "synthetic_signs":np.sign(sign).astype(int).tolist(),"real_real_ceiling":ceiling})
        family=f"{method_name}_{cap_bytes}_{multiplier}n"
        row={"source_family":"seoul_bikeshare_system","dataset":"uci_560_source_holdout",
             "seed":SEED,"family":family,"auditor":"ridge",
             "status":"complete" if transfer_report["status"]=="informative" else transfer_report["status"],
             "informative":transfer_report["status"]=="informative","transfer":transfer_report.get("raw_transfer")}
        aggregate_report=aggregate.evaluate({"required_families":[family],"required_auditors":{family:["ridge"]},
            "required_seeds":[SEED],"rows":[row],"bootstrap_replicates":1000,"bootstrap_seed":SEED})
        arms.append({"method":method_name,"cap_bytes":cap_bytes,"complete_artifact_bytes":len(artifact),
            "generated_multiplier":multiplier,"requested_rows":count_rows,"produced_rows":len(synthetic),
            "consumed_rows":len(synthetic),"artifact_sha256":sha(artifact),"sample_sha256":sha(synthetic.astype("<f8").tobytes()),
            "mse_bikes_squared":float(np.mean((y_test-prediction)**2)),"transfer":transfer_report,"driver":driver_report,
            "driver_interaction":"unavailable_not_estimated","authoritative_aggregate":aggregate_report,
            "qualification":"development_only_unqualified"})

    for cap_bytes in CAPS:
        artifact = output / f"dope-{cap_bytes}.raw"
        compiled = step(f"dope_compile_cap_{cap_bytes}", lambda: native(binary, ["compile-raw", "--csv", str(output / "fit.csv"),
            "--schema", str(output / "schema.json"), "--out", str(artifact), "--seed", str(SEED),
            "--candidate", "chow_liu", "--max-complete-bytes", str(cap_bytes), "--deadline-seconds", "60"]))
        inspected = step(f"dope_inspect_cap_{cap_bytes}", lambda: native(binary, ["inspect-raw", "--artifact", str(artifact)]))
        complete_bytes = inspected["accounting"]["complete_sampling_bytes"]
        if complete_bytes != artifact.stat().st_size or complete_bytes > cap_bytes:
            raise ValueError("artifact_cap_accounting_mismatch")
        for multiplier in (1, 4):
            count = len(seoul_fit) * multiplier
            sample_file = output / f"sample-{cap_bytes}-{multiplier}n.csv"
            draw = step(f"dope_sample_{cap_bytes}_{multiplier}n", lambda: native(binary, ["sample-raw", "--artifact", str(artifact),
                "--rows", str(count), "--seed", str(SEED + 1), "--out", str(sample_file)]))
            replay = step(f"isolated_replay_{cap_bytes}_{multiplier}n", lambda: native(binary, ["sample-raw-isolated", "--artifact", str(artifact),
                "--rows", str(count), "--seed", str(SEED + 1)]))
            synthetic = np.loadtxt(sample_file, delimiter=",", dtype=np.float64)
            if synthetic.shape != (count, 9) or not np.isfinite(synthetic).all():
                raise ValueError("sample_shape_invalid")
            if sha(synthetic.astype("<f8").tobytes()) != replay["sample_sha256"]:
                raise ValueError("isolated_replay_mismatch")
            with threadpool_limits(limits=1):
                syn_model = step(f"ridge_fit_{cap_bytes}_{multiplier}n", lambda: model().fit(synthetic[:, :8], synthetic[:, 8]))
                prediction = syn_model.predict(X_test)
            syn_days = losses_by_day(y_test, prediction, test_days)
            transfer_report = transfer.evaluate({"null_losses": null_row_loss,
                "real_losses": real_row_loss,
                "synthetic_losses": squared_losses(y_test, prediction),
                "real_gain_lcb": real_gain_lcb, "constant_target": bool(np.ptp(y_real) == 0)})
            syn_importance, syn_sign = model_signature(syn_model)
            driver_report = drivers.evaluate({"required_components": ["mass", "rank", "sign"],
                "real_importances": real_importance.tolist(), "synthetic_importances": syn_importance.tolist(),
                "real_signs": np.sign(real_sign).astype(int).tolist(),
                "synthetic_signs": np.sign(syn_sign).astype(int).tolist(), "real_real_ceiling": ceiling})
            row = {"source_family": "seoul_bikeshare_system", "dataset": "uci_560_source_holdout",
                   "seed": SEED, "family": f"dope_{cap_bytes}_{multiplier}n", "auditor": "ridge",
                   "status": "complete" if transfer_report["status"] == "informative" else transfer_report["status"],
                   "informative": transfer_report["status"] == "informative", "transfer": transfer_report.get("raw_transfer")}
            aggregate_report = aggregate.evaluate({"required_families": [row["family"]],
                "required_auditors": {row["family"]: ["ridge"]}, "required_seeds": [SEED],
                "rows": [row], "bootstrap_replicates": 1000, "bootstrap_seed": SEED})
            arms.append({"method":"dope", "cap_bytes": cap_bytes, "complete_artifact_bytes": complete_bytes,
                "generated_multiplier": multiplier, "requested_rows": count, "produced_rows": draw.get("produced_rows", count),
                "consumed_rows": int(synthetic.shape[0]), "artifact_sha256": sha(artifact.read_bytes()),
                "sample_sha256": replay["sample_sha256"], "mse_bikes_squared": float(np.mean((y_test - prediction) ** 2)),
                "transfer": transfer_report, "driver": driver_report, "driver_interaction": "unavailable_not_estimated",
                "authoritative_aggregate": aggregate_report})

    fit_matrix=np.asarray([[*r["x"],r["y"]] for r in cap_fit],dtype=np.float64)
    for method_name, fit_artifact, sample_fn in (
        ("gaussian_copula", gaussian_copula_artifact, gaussian_copula_sample),
        ("compressed_row_subsample", compressed_row_artifact, compressed_row_sample)):
        for cap_bytes in CAPS:
            artifact=step(f"{method_name}_fit_cap_{cap_bytes}",lambda:fit_artifact(fit_matrix,cap_bytes,SEED))
            (output/f"{method_name}-{cap_bytes}.artifact.gz").write_bytes(artifact)
            for multiplier in (1,4):
                count_rows=len(seoul_fit)*multiplier
                synthetic=step(f"{method_name}_sample_{cap_bytes}_{multiplier}n",lambda:sample_fn(artifact,count_rows,SEED+1))
                if synthetic.shape!=(count_rows,9) or not np.isfinite(synthetic).all():
                    raise ValueError("sample_shape_invalid")
                evaluate_control(method_name,cap_bytes,multiplier,synthetic,artifact)
    return {"format": "dope-bikeshare-observation-v1", "scope": "public_observational_source_holdout_development",
        "status": "observed_unqualified", "protocol_sha256": sha((Path(__file__).with_name("protocol.json")).read_bytes()),
        "source_manifest_sha256": sha((Path(__file__).with_name("source-manifest.json")).read_bytes()),
        "split_sha256": sha((output / "split.json").read_bytes()),
        "prepared_fit_view": {"bytes": (output / "fit.csv").stat().st_size,
            "sha256": sha((output / "fit.csv").read_bytes()), "schema_sha256": sha((output / "schema.json").read_bytes())},
        "source_archives": {name: {"filename": ARCHIVES[name][0], "archive_bytes": SOURCE_SIZES[name]["archive"],
             "member": ARCHIVES[name][2], "member_bytes": SOURCE_SIZES[name]["member"],
             "sha256": ARCHIVES[name][1], "member_sha256": ARCHIVES[name][3],
             "license": "CC-BY-4.0", "source_page": "https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset" if name == "capital" else "https://archive.ics.uci.edu/dataset/560/seoul%2Bbike%2Bsharing%2Bdemand"}
             for name in ARCHIVES},
        "native": doctor, "seed": SEED,
        "lineage": {"capital_fit_rows": len(cap_fit), "capital_select_rows": len(cap_select),
            "seoul_fit_rows": len(seoul_fit), "seoul_select_rows": len(seoul_select), "seoul_test_rows": len(seoul_test),
            "seoul_nonoperating_rows_excluded": excluded_nonoperating,
            "seoul_source_holdout": True, "entity_unit": "service_date", "entity_overlap": False,
            "selection_rows_accessed": 0, "target_units": "bikes_per_hour"},
        "reference": {"recipe": "StandardScaler + Ridge(alpha=1.0)", "real_mse_bikes_squared": float(np.mean((y_test-real_pred)**2)),
            "null_mse_bikes_squared": float(np.mean((y_test-null_pred)**2)), "real_gain_lcb": real_gain_lcb,
            "real_loss_by_test_day": real_day_loss, "null_loss_by_test_day": null_day_loss},
        "arms": arms, "driver_real_real_ceiling": ceiling,
        "cost": {"elapsed_seconds": time.monotonic()-started, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "rss_scope": "process ru_maxrss includes interpreter imports and prior calls", "attempts": stages,
            "runtime_versions": {name: __import__(name).__version__ for name in ("numpy", "scipy", "sklearn", "threadpoolctl")}},
        "qualification": {"source_holdout_observed": True, "independent_curator_replication": False,
            "tdl": False, "privacy": False, "modern_panel": False, "foundation_models": False,
            "healthcare_inference": False},
        "limitations": ["Two public systems from one repository curator; this is a source-system holdout, not independent-curator replication.",
            "One fixed Ridge auditor and one seeded fit; no full auditor panel or modern scientific qualification.",
            "Driver report measures coefficient/permutation proxies only; interactions are unavailable.",
            "A weak reference lift or failed driver/transfer gate is reported as incomplete or uninformative, not as a DOPE win."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capital-zip", type=Path, required=True)
    parser.add_argument("--seoul-zip", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run(args.capital_zip, args.seoul_zip, args.binary, args.out)
        (args.out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"state": receipt["status"], "fit_rows": receipt["lineage"]["capital_fit_rows"],
            "test_rows": receipt["lineage"]["seoul_test_rows"], "arms": len(receipt["arms"]),
            "tdl": receipt["qualification"]["tdl"]}, sort_keys=True))
    except (ValueError, OSError, KeyError, ImportError, subprocess.SubprocessError, UnicodeDecodeError) as error:
        known = {"source_carrier_invalid", "source_digest_mismatch", "source_member_digest_mismatch",
                 "source_archive_members_invalid", "source_table_shape_invalid", "capital_rows_invalid",
                 "seoul_rows_invalid", "split_policy_invalid", "split_partition_empty", "split_entity_overlap",
                 "cross_source_entity_identity_collision", "native_command_unadmitted", "native_command_failed",
                 "native_response_invalid", "native_binary_identity_invalid", "artifact_cap_accounting_mismatch",
                 "sample_shape_invalid", "isolated_replay_mismatch", "output_must_be_new_persistent_directory"}
        reason = "source_encoding_invalid" if isinstance(error, UnicodeDecodeError) else (
            str(error) if isinstance(error, ValueError) and str(error) in known else "observational_example_failed")
        if args.out.exists():
            attempts = (args.out / "attempts.jsonl").read_text().splitlines() if (args.out / "attempts.jsonl").exists() else []
            (args.out / "failure.json").write_text(json.dumps({"status":"failed","reason":reason,
                "failed_stage": next((item["stage"] for item in reversed([json.loads(line) for line in attempts])
                                       if item["state"] == "failed"), "between_recorded_stages")})+"\n")
        print(json.dumps({"status":"failed","reason":reason}))
        raise SystemExit(1) from None

if __name__ == "__main__":
    main()
