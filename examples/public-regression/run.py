#!/usr/bin/env python3
"""Pinned, grouped public Airfoil demo. No confirmation or privacy claim."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import inspect
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/291/airfoil%2Bself%2Bnoise.zip"
ARCHIVE_SHA = "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c"
SOURCE_SHA = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"
SOURCE_BYTES = 59984
NAMES = ["frequency", "attack_angle", "chord_length", "free_stream_velocity", "displacement_thickness", "sound_pressure"]
UNITS = ["Hz", "deg", "m", "m/s", "m", "dB"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        def git(*arguments: str) -> str:
            return subprocess.run(["git", "-C", str(root), *arguments], check=True,
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
                "tracked_source_clean": not git("status", "--porcelain", "--untracked-files=no"),
                "scope": "runner_checkout_identity_binary_build_attestation_separate"}
    except (OSError, subprocess.SubprocessError):
        return {"state": "unavailable", "scope": "binary_build_attestation_separate"}


def measured_step(output: Path, attempts: list[dict], stage: str, operation):
    attempt = {"stage": stage, "state": "started", "elapsed_seconds": None}
    started = time.monotonic()
    journal = output / "attempts.jsonl"
    with journal.open("a") as stream:
        stream.write(json.dumps(attempt, allow_nan=False) + "\n")
    try:
        result = operation()
        attempt["state"] = "completed"
        return result
    except Exception:
        attempt["state"] = "failed"
        raise
    finally:
        attempt["elapsed_seconds"] = time.monotonic() - started
        attempts.append(attempt)
        with journal.open("a") as stream:
            stream.write(json.dumps(attempt, allow_nan=False) + "\n")


def read_source(path: Path | None, cache: Path) -> bytes:
    if path is not None:
        if path.is_symlink() or not path.is_file() or path.stat().st_size != SOURCE_BYTES:
            raise ValueError("public_source_carrier_invalid")
        data = path.read_bytes()
    elif cache.exists():
        data = read_source(cache, cache)
    else:
        with urllib.request.urlopen(URL, timeout=30) as response:
            archive_bytes = response.read(1024 * 1024 + 1)
        if digest(archive_bytes) != ARCHIVE_SHA:
            raise ValueError("public_archive_digest_mismatch")
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            if archive.namelist() != ["airfoil_self_noise.dat"]:
                raise ValueError("public_archive_members_mismatch")
            data = archive.read("airfoil_self_noise.dat")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("xb") as stream:
            stream.write(data)
    if len(data) != SOURCE_BYTES or digest(data) != SOURCE_SHA:
        raise ValueError("public_source_digest_mismatch")
    return data


def parse_rows(data: bytes) -> list[list[float]]:
    rows = [[float(value) for value in line.split()] for line in data.decode("ascii").splitlines() if line.strip()]
    if len(rows) != 1503 or any(len(row) != 6 or not all(math.isfinite(v) for v in row) for row in rows):
        raise ValueError("public_source_schema_mismatch")
    if any(row[0] != round(row[0]) for row in rows):
        raise ValueError("public_frequency_type_mismatch")
    return rows


def grouped_split(rows: list[list[float]], seed: int) -> tuple[list[int], list[int], dict]:
    # Frequency is deliberately excluded: repeated frequency measurements of a
    # physical configuration must remain wholly within one arm.
    keys = [tuple(format(row[i], ".17g") for i in [1, 2, 3, 4]) for row in rows]
    groups = sorted(set(keys), key=lambda key: digest(json.dumps([seed, key], separators=(",", ":")).encode()))
    if len(groups) < 10:
        raise ValueError("public_configuration_groups_insufficient")
    fit_groups = set(groups[:int(0.8 * len(groups))])
    fit = [i for i, key in enumerate(keys) if key in fit_groups]
    query = [i for i, key in enumerate(keys) if key not in fit_groups]
    if len(fit) < 100 or len(query) < 50 or set(keys[i] for i in fit) & set(keys[i] for i in query):
        raise ValueError("public_group_split_invalid")
    manifest = {"format": "airfoil-configuration-group-split-v1", "seed": seed,
                "policy": "80_percent_of_hashed_configuration_groups", "frequency_in_group_key": False,
                "configuration_groups": len(groups), "fit_groups": len(fit_groups), "query_groups": len(groups) - len(fit_groups),
                "fit_rows": fit, "query_rows": query, "scope": "public_development_demonstration_not_confirmation"}
    return fit, query, manifest


def schema() -> dict:
    return {"format": "dope-table-schema", "version": 1, "metadata_visibility": "approved_semantic_labels",
            "columns": [{"id": f"c{i+1:04d}", "source_name": name, "role": "target" if i == 5 else "feature",
                         "dtype": "integer" if i == 0 else "continuous", "units": UNITS[i], "semantic_label": name}
                        for i, name in enumerate(NAMES)]}


def native(binary: Path, arguments: list[str], timeout: int = 90) -> dict:
    # Clear ambient import/cache/loader variables for the native process. Every
    # argument is passed literally; query tables are never compilation inputs.
    environment = {} if arguments[0]=="sample-raw-isolated" else {"RAYON_NUM_THREADS":"1"}
    result = subprocess.run([str(binary), *arguments], env=environment, capture_output=True, text=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise ValueError(f"public_native_{arguments[0].replace('-', '_')}_failed")
    return json.loads(result.stdout)


def learner_predictions(fit_x, fit_y, query_x):
    import numpy as np
    import sklearn
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    if sklearn.__version__ != "1.7.2":
        raise ValueError("public_learner_primary_pin_mismatch")
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(fit_x)
        x = scaler.transform(fit_x)
        query = scaler.transform(query_x)
        model = Ridge(alpha=1.0, solver="svd", fit_intercept=True).fit(x, fit_y)
        predictions = model.predict(query)
        # Independently recompute the same frozen ridge functional. This verifies
        # native-library parity for this small CPU recipe, not optional vendors.
        center = np.mean(x, axis=0)
        y_mean = float(np.mean(fit_y))
        xc = x - center
        coef = np.linalg.solve(xc.T @ xc + np.eye(x.shape[1]), xc.T @ (fit_y - y_mean))
        reference = (query - center) @ coef + y_mean
        error = float(np.max(np.abs(reference - predictions)))
        if not np.all(np.isfinite(predictions)) or error > 1e-8:
            raise ValueError("public_learner_native_reference_mismatch")
    return predictions, {"family": "linear_diagnostic", "library": "scikit-learn", "version": sklearn.__version__,
                         "native_source_sha256": digest(Path(inspect.getsourcefile(Ridge)).read_bytes()),
                         "recipe": "fit_arm_standard_scaler_ridge_alpha1_svd_intercept", "threads": 1,
                         "native_reference_max_absolute_error": error, "native_reference": "passed",
                         "consumed_context_rows": len(fit_y)}


def losses(truth, predictions) -> dict:
    gaps = [float(p) - float(y) for p, y in zip(predictions, truth, strict=True)]
    mse = sum(g * g for g in gaps) / len(gaps)
    return {"mse_db_squared": mse, "rmse_db": math.sqrt(mse), "mae_db": sum(abs(g) for g in gaps) / len(gaps)}


def render_report(receipt: dict) -> str:
    safe = html.escape(json.dumps(receipt, indent=2, sort_keys=True))
    table=""
    if "real_loss" in receipt:
        rows=[("Real fitting rows",receipt["fit_rows"],receipt["real_loss"],None),
              ("Fit-mean null",receipt["fit_rows"],receipt["null_loss"],None)]
        rows.extend((f"Synthetic {arm['generated_multiplier']}n",arm["consumed_rows"],arm["loss"],arm["retention"]) for arm in receipt["arms"])
        table="<table><tr><th>Training arm</th><th>Rows consumed</th><th>MSE (dB²)</th><th>RMSE (dB)</th><th>MAE (dB)</th><th>Raw retention</th></tr>"
        for label,n,loss,retention in rows:
            r="—" if retention is None else f"{retention:.4f}"
            table+=f"<tr><td>{label}</td><td>{n}</td><td>{loss['mse_db_squared']:.4f}</td><td>{loss['rmse_db']:.4f}</td><td>{loss['mae_db']:.4f}</td><td>{r}</td></tr>"
        table+="</table>"
        table=f"<p>{receipt['fit_rows']} fitting rows / {receipt['query_rows']} query rows; {receipt['fit_groups']} fitting configurations / {receipt['query_groups']} query configurations. Complete sampling artifact: {receipt['inspection']['accounting']['complete_sampling_bytes']:,} bytes. One declared linear learner.</p>"+table
    return "<!doctype html><meta charset=utf-8><title>DOPE public Airfoil demonstration</title>" \
           "<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px}td,th{padding:10px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>" \
           "<h1>Public Airfoil transfer demonstration</h1><p>Observed public grouped split; " \
           "confirmation, privacy, driver and geometry qualification are unmeasured.</p>" \
           +table+ \
           "<p>The complete artifact includes schema, fitted transforms, framing and native payload. " \
           "A fresh restricted native worker verifies replay. Retention may be negative or exceed one.</p>" \
           f"<details><summary>Reproduction receipt and limitations</summary><pre>{safe}</pre></details>"


def run(binary: Path, data: bytes, output: Path, seed: int) -> dict:
    started = time.monotonic()
    if len(data)!=SOURCE_BYTES or digest(data)!=SOURCE_SHA:
        raise ValueError("public_source_digest_mismatch")
    if not sys.flags.isolated or sys.prefix==sys.base_prefix or "include-system-site-packages = true" in (Path(sys.prefix)/"pyvenv.cfg").read_text().lower():
        raise ValueError("public_demo_requires_isolated_python_environment")
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(Path("/tmp")):
        raise ValueError("public_demo_output_must_be_new_persistent_directory")
    import numpy as np
    output.mkdir(parents=True)
    attempts = []
    def step(stage: str, operation):
        return measured_step(output, attempts, stage, operation)
    rows = parse_rows(data)
    fit, query, split = grouped_split(rows, seed)
    (output / "split.json").write_text(json.dumps(split, sort_keys=True) + "\n")
    (output / "schema.json").write_text(json.dumps(schema(), sort_keys=True) + "\n")
    with (output / "fit.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(NAMES)
        writer.writerows(rows[i] for i in fit)
    binary = binary.resolve(strict=True)
    doctor = step("native_doctor", lambda: native(binary, ["doctor"]))
    if doctor.get("implementation") != "native_rust" or doctor.get("executable_sha256") != digest(binary.read_bytes()):
        raise ValueError("public_demo_requires_native_binary")
    artifact = output / "model.raw"
    compiled = step("native_fit", lambda: native(binary, ["compile-raw", "--csv", str(output / "fit.csv"), "--schema", str(output / "schema.json"),
                               "--out", str(artifact), "--seed", str(seed), "--candidate", "chow_liu",
                               "--max-complete-bytes", "32768", "--deadline-seconds", "60"]))
    inspect = step("native_inspection", lambda: native(binary, ["inspect-raw", "--artifact", str(artifact)]))
    if compiled["accounting"] != inspect["accounting"] or artifact.stat().st_size != inspect["accounting"]["complete_sampling_bytes"]:
        raise ValueError("public_complete_byte_mismatch")
    table = np.asarray(rows, dtype=np.float64)
    real_x, real_y = table[fit, :5], table[fit, 5]
    query_x, truth = table[query, :5], table[query, 5]
    real_pred, recipe = step("real_learner", lambda: learner_predictions(real_x, real_y, query_x))
    real_loss = losses(truth, real_pred)
    null_loss = losses(truth, np.full(len(query), float(np.mean(real_y))))
    arms = []
    for multiplier in [1, 4]:
        n = len(fit) * multiplier
        sample_path = output / f"sample-{multiplier}n.csv"
        sampled = step(f"sampling_{multiplier}n", lambda: native(binary, ["sample-raw", "--artifact", str(artifact), "--rows", str(n), "--seed", str(seed + 1), "--out", str(sample_path)]))
        isolated = step(f"isolated_replay_{multiplier}n", lambda: native(binary, ["sample-raw-isolated", "--artifact", str(artifact), "--rows", str(n), "--seed", str(seed + 1)]))
        synthetic = np.loadtxt(sample_path, delimiter=",", dtype=np.float64)
        if synthetic.shape != (n, 6) or not np.all(np.isfinite(synthetic)):
            raise ValueError("public_sample_shape_invalid")
        if digest(synthetic.astype("<f8").tobytes()) != isolated["sample_sha256"]:
            raise ValueError("public_isolated_replay_mismatch")
        pred, native_recipe = step(f"synthetic_learner_{multiplier}n", lambda: learner_predictions(synthetic[:, :5], synthetic[:, 5], query_x))
        loss = losses(truth, pred)
        rng=np.random.default_rng(seed+multiplier)
        bootstrap=rng.integers(0,len(fit),size=n)
        bootstrap_pred, bootstrap_recipe=step(f"bootstrap_learner_{multiplier}n", lambda: learner_predictions(real_x[bootstrap],real_y[bootstrap],query_x))
        denominator = null_loss["mse_db_squared"] - real_loss["mse_db_squared"]
        retention = None if denominator == 0 else (null_loss["mse_db_squared"] - loss["mse_db_squared"]) / denominator
        arms.append({"generated_multiplier": multiplier, "produced_rows": n, "consumed_rows": n,
                     "loss": loss, "retention": retention,
                     "reference_lift_db_squared":denominator,
                     "reference_informative":denominator>0,
                     "retention_state": "undefined_zero_reference_lift" if denominator == 0 else ("reference_not_better_than_null_raw_ratio_only" if denominator<0 else "observed_unbounded"),
                     "sampling": sampled, "source_free_replay": isolated, "learner": native_recipe,
                     "real_row_bootstrap_diagnostic": {"loss":losses(truth,bootstrap_pred),"learner":bootstrap_recipe,"scope":"matched_context_exposure_not_upper_bound_or_new_independent_information"},
                     "prediction_sha256": digest(np.asarray(pred, dtype="<f8").tobytes())})
    receipt = {"format": "dope-public-airfoil-demo-v1", "scope": "public_grouped_development_demonstration",
               "implementation_sha256": digest(Path(__file__).read_bytes()),
               "source_control": source_identity(),
               "source": {"url": URL, "doi": "10.24432/C5VW2C", "sha256": SOURCE_SHA, "bytes": len(data),
                          "license": "CC-BY-4.0", "attribution": "Brooks, T., Pope, D., & Marcolini, M. (1989). Airfoil Self-Noise. UCI Machine Learning Repository."},
               "source_rows": len(rows), "split_sha256": digest((output / "split.json").read_bytes()),
               "fit_rows": len(fit), "query_rows": len(query), "fit_groups": split["fit_groups"], "query_groups": split["query_groups"],
               "seed": seed, "native": doctor, "artifact_sha256": digest(artifact.read_bytes()), "inspection": inspect,
               "learner": recipe, "real_loss": real_loss, "null_loss": null_loss, "arms": arms,
               "fit_target_variance_db_squared":float(np.var(real_y)),
               "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "threadpoolctl"]},
               "driver": "unmeasured", "geometry": "unmeasured", "privacy": "unmeasured", "confirmation": "not_authorized",
               "optional_foundation_model": "not_run_no_admitted_local_checkpoint", "release": "unqualified",
               "elapsed_seconds": time.monotonic() - started,
               "attempts": attempts,
               "cost_scope": "observed_public_demo_steps_not_full_campaign_or_resident_memory",
               "limitations": ["Public demonstration, not independent-source confirmation.", "One linear diagnostic learner, not a portfolio transfer result.",
                               "Feature support remains fit-range bounded; residual tails are not qualified.", "Private affine/schema state is retained and charged; disclosure attacks are absent."]}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "report.html").write_text(render_report(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("target/public-demo/source/airfoil_self_noise.dat"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    try:
        result = run(args.binary, read_source(args.source, args.cache), args.out, args.seed)
        print(json.dumps({"state": "observed_demo_complete", "complete_sampling_bytes": result["inspection"]["accounting"]["complete_sampling_bytes"],
                          "fit_rows": result["fit_rows"], "query_rows": result["query_rows"], "privacy": result["privacy"]}))
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
        # Preserve any partially written directory, but never echo source paths,
        # headers, category values, or command stderr into a public report.
        reason = str(error) if isinstance(error, ValueError) and str(error).startswith("public_") else "public_demo_failed"
        if args.out.exists():
            (args.out / "failure.json").write_text(json.dumps({"state": "failed", "reason": reason}) + "\n")
        raise SystemExit(reason) from None


if __name__ == "__main__":
    main()
