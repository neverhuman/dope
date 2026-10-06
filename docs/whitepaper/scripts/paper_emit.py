"""Format paper macros and short tables from the panel-stats payload.

Formatting only. The statistics live in compute_panel.py. A missing local
receipt extract becomes the words ``not measured'' rather than a guessed
number.
"""

from __future__ import annotations

import json


def _fit_trace(out):
    path = out / "fit-trace.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _breakable(text):
    """Let a long identifier wrap without changing the characters that are printed.

    The empty group keeps ``\\allowbreak`` from swallowing the next letter.
    A long letter run is split so a two-column line can fill.
    """
    def soften(piece):
        width = 12
        if len(piece) <= width:
            return piece
        parts = [piece[index : index + width] for index in range(0, len(piece), width)]
        return r"\allowbreak{}".join(parts)

    return r"\_\allowbreak{}".join(soften(piece) for piece in text.split("_"))


def _break_hash(value):
    if not value or value == "not measured":
        return value or "not measured"
    pieces = [value[index : index + 8] for index in range(0, len(value), 8)]
    return r"\allowbreak{}".join(pieces)


def _count(trace, key, field):
    block = trace.get(key) or {}
    value = block.get(field)
    if value is None:
        return "not measured"
    return str(value)


def _elapsed(value, tex_bytes):
    if value is None:
        return "not measured"
    number = float(value)
    if abs(number) >= 100:
        return tex_bytes(number)
    if abs(number) >= 10:
        return f"{number:.1f}"
    if abs(number) >= 1:
        return f"{number:.2f}"
    return f"{number:.3g}"


def _byte_median(value, tex_bytes):
    if value is None:
        return "---"
    number = float(value)
    doubled = round(number * 2)
    if abs(number * 2 - doubled) < 1e-6 and doubled % 2 == 1:
        return f"{number:,.1f}".replace(",", "{,}")
    return tex_bytes(number)


def _ratio(pack):
    if not pack or not pack.get("n"):
        return "---"
    return f"{pack['within']}/{pack['n']}"


def _loss_macro(profile, field, sig3):
    if not profile:
        return "not measured"
    return sig3(profile.get(field))


def _threshold_sentence(series, sig3):
    primary = series["0.01"]
    held = []
    for label in ("0.0", "0.001", "0.05"):
        item = series[label]
        same_n = item["n"] == primary["n"]
        same_median = item["median"] is not None and abs(item["median"] - primary["median"]) < 1e-12
        if same_n and same_median:
            held.append(label.replace("0.0", "0") if label == "0.0" else label)
    tail = series["0.1"]
    if len(held) == 3:
        return (
            f"Thresholds 0, 0.001, and 0.05 keep that median and that count. "
            f"At 0.1 the count is {tail['n']} and the median is {sig3(tail['median'])}."
        )
    return "The threshold table is the record of that check."


def emit(payload, out, sig3, tex_p, tex_bytes, command):
    density = payload["blocks"]["density"]["catboost"]
    linear = payload["blocks"]["density"]["linear"]
    neural = payload["blocks"]["neural"]["catboost"]
    forest = payload["blocks"]["forest"]["catboost"]
    side = payload["sidecars"]
    fidelity = payload["fidelity"]["DOPE"]
    dope_bytes = payload["bytes"]["DOPE"]
    gauss_pair = linear["pairs"]["GaussianCopula"]
    ranks = (linear.get("friedman") or {}).get("average_ranks") or {}
    loss = (side.get("loss") or {}).get("profiles") or {}
    loss_a = loss.get("features12_steps2048") or {}
    loss_b = loss.get("features12_steps8192") or {}
    replay = side.get("replay") or {}
    replay_a = (replay.get("features12_steps2048") or {})
    replay_b = (replay.get("features12_steps8192") or {})
    lock = side["lock"]
    sufficiency = side["sufficiency"]
    packs = side["bytes"]
    misses = side["over_cap"]
    cost = side["forest_cost"]
    lines = [
        command("MedDopeCb", sig3(density["methods"]["DOPE"]["median"])),
        command("LoDopeCb", sig3(density["methods"]["DOPE"]["lo"])),
        command("HiDopeCb", sig3(density["methods"]["DOPE"]["hi"])),
        command("NDopeCb", str(density["methods"]["DOPE"]["n"])),
        command("MedGaussCb", sig3(density["pairs"]["GaussianCopula"]["other_median"])),
        command("LoGaussCb", sig3(density["pairs"]["GaussianCopula"]["other_lo"])),
        command("HiGaussCb", sig3(density["pairs"]["GaussianCopula"]["other_hi"])),
        command("MedChowCb", sig3(density["pairs"]["Chow-Liu"]["other_median"])),
        command("LoChowCb", sig3(density["pairs"]["Chow-Liu"]["other_lo"])),
        command("HiChowCb", sig3(density["pairs"]["Chow-Liu"]["other_hi"])),
        command("MedIndCb", sig3(density["pairs"]["independent_marginals"]["other_median"])),
        command("LoIndCb", sig3(density["pairs"]["independent_marginals"]["other_lo"])),
        command("HiIndCb", sig3(density["pairs"]["independent_marginals"]["other_hi"])),
        command("HolmGaussCb", tex_p(density["pairs"]["GaussianCopula"]["holm_p"])),
        command("HolmChowCb", tex_p(density["pairs"]["Chow-Liu"]["holm_p"])),
        command("HolmIndCb", tex_p(density["pairs"]["independent_marginals"]["holm_p"])),
        command("SignHolmGaussCb", tex_p(density["pairs"]["GaussianCopula"].get("sign_holm_p"))),
        command("SignHolmChowCb", tex_p(density["pairs"]["Chow-Liu"].get("sign_holm_p"))),
        command("SignHolmIndCb", tex_p(density["pairs"]["independent_marginals"].get("sign_holm_p"))),
        command("RGaussCb", sig3(density["pairs"]["GaussianCopula"]["rank_biserial"])),
        command("RChowCb", sig3(density["pairs"]["Chow-Liu"]["rank_biserial"])),
        command("RIndCb", sig3(density["pairs"]["independent_marginals"]["rank_biserial"])),
        command("DiffGaussCb", sig3(density["pairs"]["GaussianCopula"]["median_difference"])),
        command("LoDiffGaussCb", sig3(density["pairs"]["GaussianCopula"]["lo"])),
        command("HiDiffGaussCb", sig3(density["pairs"]["GaussianCopula"]["hi"])),
        command("FriedmanPCb", tex_p((density.get("friedman") or {}).get("p"))),
        command("FriedmanCdCb", sig3((density.get("friedman") or {}).get("cd"))),
        command("FriedmanNCb", str((density.get("friedman") or {}).get("n") or 0)),
        command("HolmLinGauss", tex_p(gauss_pair["holm_p"])),
        command("DiffLinGauss", sig3(gauss_pair["median_difference"])),
        command("LoDiffLinGauss", sig3(gauss_pair["lo"])),
        command("HiDiffLinGauss", sig3(gauss_pair["hi"])),
        command("RLinGauss", sig3(gauss_pair["rank_biserial"])),
        command("FriedmanCdLin", sig3((linear.get("friedman") or {}).get("cd"))),
        command("RankDopeLin", sig3(ranks.get("DOPE"))),
        command("RankGaussLin", sig3(ranks.get("GaussianCopula"))),
        command("MedDopeNeuCb", sig3(neural["pairs"]["CTGAN"]["dope_median"])),
        command("LoDopeNeuCb", sig3(neural["pairs"]["CTGAN"]["dope_lo"])),
        command("HiDopeNeuCb", sig3(neural["pairs"]["CTGAN"]["dope_hi"])),
        command("MedCtganCb", sig3(neural["pairs"]["CTGAN"]["other_median"])),
        command("MedTvaeCb", sig3(neural["pairs"]["TVAE"]["other_median"])),
        command("NNeuCb", str(neural["pairs"]["CTGAN"]["n"])),
        command("MedForestCb", sig3(forest["pairs"]["Forest-Flow"]["other_median"])),
        command("LoForestCb", sig3(forest["pairs"]["Forest-Flow"]["other_lo"])),
        command("HiForestCb", sig3(forest["pairs"]["Forest-Flow"]["other_hi"])),
        command("MedDopeForestCb", sig3(forest["pairs"]["Forest-Flow"]["dope_median"])),
        command("NForestCb", str(forest["pairs"]["Forest-Flow"]["n"])),
        command("ForestByteLo", tex_bytes(payload["forest_bytes"]["min"])),
        command("ForestByteHi", tex_bytes(payload["forest_bytes"]["max"])),
        command("ForestByteMedian", _byte_median(side.get("forest_byte_median"), tex_bytes)),
        command("DopeForestByteMedian", _byte_median(side.get("dope_forest_byte_median"), tex_bytes)),
        command("DopeWithinCap", f"{dope_bytes['within_cap']}/{dope_bytes['known']}"),
        command("DopeByteMedian", _byte_median(dope_bytes["median"]["median"], tex_bytes)),
        command("DopeByteLo", tex_bytes(side.get("dope_byte_min"))),
        command("TabSynPanel", "not measured"),
        command("TabDDPMPanel", "not measured"),
        command("PrivacyAttackPanel", "not measured"),
        command("FitSeedVariance", "not measured"),
        command("FullAblationGrid", "not measured"),
        command("CampaignElapsed", "not measured"),
    ]
    byte_rows = []
    order = (
        ("DOPE reference", None),
        ("Gaussian copula", "GaussianCopula"),
        ("Chow--Liu, study", "Chow-Liu"),
        ("Indep.\\ marginals, study", "independent_marginals"),
        ("CTGAN", "CTGAN"),
        ("TVAE", "TVAE"),
    )
    shown = {
        "DOPE reference": {
            "median": dope_bytes["median"]["median"],
            "within": dope_bytes["within_cap"],
            "n": dope_bytes["known"],
        }
    }
    for label, key in order[1:]:
        pack = packs[key]
        shown[label] = {"median": pack["median"], "within": pack["within"], "n": pack["n"]}
        macro = {
            "GaussianCopula": "Gauss",
            "Chow-Liu": "Chow",
            "independent_marginals": "Ind",
            "CTGAN": "Ctgan",
            "TVAE": "Tvae",
        }[key]
        lines.append(command(f"{macro}ByteMedian", _byte_median(pack["median"], tex_bytes)))
        lines.append(command(f"{macro}Within", _ratio(pack)))
    for label, _key in order:
        item = shown[label]
        byte_rows.append(
            f"{label} & {_byte_median(item['median'], tex_bytes)} & {item['within']}/{item['n']} \\\\"
        )
    forest_bytes = payload["forest_bytes"]
    if forest_bytes.get("n"):
        byte_rows.append(
            "Forest-Flow & "
            f"{_byte_median(side.get('forest_byte_median'), tex_bytes)} & "
            f"{forest_bytes['within']}/{forest_bytes['n']} \\\\"
        )
    misses = side["over_cap"]
    if len(misses) >= 1:
        lines.append(command("MissA", _breakable(misses[0]["name"])))
        lines.append(command("MissABytes", tex_bytes(misses[0]["bytes"])))
        lines.append(command("DopeByteHi", tex_bytes(misses[0]["bytes"])))
    else:
        lines.append(command("MissA", "none"))
        lines.append(command("MissABytes", "---"))
        lines.append(command("DopeByteHi", "---"))
    if len(misses) >= 2:
        lines.append(command("MissB", _breakable(misses[1]["name"])))
        lines.append(command("MissBBytes", tex_bytes(misses[1]["bytes"])))
    else:
        lines.append(command("MissB", "none"))
        lines.append(command("MissBBytes", "---"))
    if side["not_informative"]:
        lines.append(command("NotInfName", _breakable(side["not_informative"][0])))
    else:
        lines.append(command("NotInfName", "none"))
    for auditor, stem in (("catboost", "Cb"), ("linear", "Lin"), ("mlp", "Mlp")):
        item = sufficiency[auditor]
        lines.append(command(f"NSuf{stem}", str(item["n"])))
        lines.append(command(f"MedSufA{stem}", sig3(item["median_2048"])))
        lines.append(command(f"MedSufB{stem}", sig3(item["median_8192"])))
        lines.append(command(f"DiffSuf{stem}", sig3(item["median_difference"])))
    fields = (
        ("marginal_ks_mean", "Ks"),
        ("pair_correlation_fidelity", "Corr"),
        ("c2st_auc", "Ctwo"),
    )
    for field, stem in fields:
        item = fidelity[field]
        lines.append(command(f"{stem}Dope", sig3(item["median"])))
        lines.append(command(f"{stem}Lo", sig3(item["lo"])))
        lines.append(command(f"{stem}Hi", sig3(item["hi"])))
        lines.append(command(f"{stem}N", str(item["n"])))
    trace = _fit_trace(out)
    forest_trace = trace.get("forest") or {}
    fit_core = forest_trace.get("fit_seconds_sum", cost.get("forest_fit_core_seconds"))
    sample_op = forest_trace.get("operation_seconds_sum", cost.get("forest_fit_native_sample_operation_seconds"))
    fit_count = forest_trace.get("fit_count", cost.get("forest_new_gpu_fits"))
    lines.append(command("ForestFitCore", _elapsed(fit_core, tex_bytes)))
    lines.append(command("ForestSampleOp", _elapsed(sample_op, tex_bytes)))
    lines.append(command("ForestGpuFits", str(fit_count if fit_count is not None else "not measured")))
    lines.append(command("ArfWatchClosed", _count(trace, "watch", "closed")))
    lines.append(command("ArfWatchOk", _count(trace, "watch", "ok")))
    lines.append(command("ArfWatchPlanned", _count(trace, "watch", "planned")))
    lines.append(command("BeyondPrepared", _count(trace, "beyond", "prepared")))
    lines.append(command("BeyondFailed", _count(trace, "beyond", "failed")))
    lines.append(command("BeyondFamilies", _count(trace, "beyond", "families")))
    lines.append(command("BeyondFeatureRange", _count(trace, "beyond", "feature_range")))
    lines.append(command("BeyondOverlap", _count(trace, "beyond", "overlap")))
    lines.append(command("BeyondInventory", _count(trace, "beyond", "inventory")))
    revision = (trace.get("beyond") or {}).get("revision")
    lines.append(command("BeyondRevision", revision if revision else "not measured"))
    lines.append(command("ReplayMedA", _elapsed(replay_a.get("median"), tex_bytes)))
    lines.append(command("ReplayMedB", _elapsed(replay_b.get("median"), tex_bytes)))
    lines.append(command("ReplayN", str(replay_a.get("n") if replay_a.get("n") is not None else "not measured")))
    lines.append(command("LossTrainStart", _loss_macro(loss_a, "step0_train", sig3)))
    lines.append(command("LossValStart", _loss_macro(loss_a, "step0_validation", sig3)))
    lines.append(command("LossTrainEndA", _loss_macro(loss_a, "final_train", sig3)))
    lines.append(command("LossValEndA", _loss_macro(loss_a, "final_validation", sig3)))
    lines.append(command("LossTrainEndB", _loss_macro(loss_b, "final_train", sig3)))
    lines.append(command("LossValEndB", _loss_macro(loss_b, "final_validation", sig3)))
    lines.append(command("CatalogSha", _break_hash(lock.get("catalog_sha256") or "not measured")))
    lines.append(command("SplitSeed", str(lock.get("split_seed") if lock.get("split_seed") is not None else "not measured")))
    lines.append(command("PreparedN", str(lock.get("prepared") if lock.get("prepared") is not None else "not measured")))
    lines.append(command("EligibleN", str(lock.get("eligible") if lock.get("eligible") is not None else "not measured")))
    lines.append(command("ExcludedN", str(lock.get("excluded") if lock.get("excluded") is not None else "not measured")))
    lines.append(command("UnknownRights", f"{int(lock['unknown_rights']):,}".replace(",", "{,}") if isinstance(lock.get("unknown_rights"), int) else "not measured"))
    lines.append(command("ThrSentence", _threshold_sentence(payload["threshold_robustness"]["DOPE"]["catboost"], sig3)))
    beyond = side.get("beyond") or {}
    lines.append(command("BeyondElapseA", _elapsed(_median_elapsed(beyond, "steps2048"), tex_bytes)))
    lines.append(command("BeyondElapseB", _elapsed(_median_elapsed(beyond, "steps8192"), tex_bytes)))
    (out / "numbers.tex").write_text("".join(lines))
    (out / "byte-table.tex").write_text(
        "\\begin{tabular}{@{}lrr@{}}\n\\toprule\n"
        "Method & Median bytes & Within cap \\\\\n\\midrule\n"
        + "\n".join(byte_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    _write_threshold(payload, out, sig3)
    _write_beyond(beyond, out, sig3, tex_bytes)
    _write_provenance(side, out)


def _median_elapsed(beyond, key):
    values = []
    for family in beyond.get("families") or []:
        slot = family.get(key) or {}
        if isinstance(slot.get("elapsed_seconds"), (int, float)):
            values.append(float(slot["elapsed_seconds"]))
    if not values:
        return None
    values.sort()
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _write_threshold(payload, out, sig3):
    labels = (
        ("DOPE", "DOPE"),
        ("GaussianCopula", "Gaussian"),
        ("Chow-Liu", "Chow--Liu"),
        ("independent_marginals", "Indep."),
    )
    rows = []
    for threshold in ("0.0", "0.001", "0.01", "0.05", "0.1"):
        cells = []
        for method, _label in labels:
            item = payload["threshold_robustness"][method]["catboost"][threshold]
            cells.append(f"{sig3(item['median'])} ({item['n']})")
        shown = "0" if threshold == "0.0" else threshold
        rows.append(f"{shown} & " + " & ".join(cells) + r" \\")
    (out / "threshold-table.tex").write_text(
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Threshold & DOPE & Gaussian & Chow--Liu & Indep. \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def _write_beyond(beyond, out, sig3, tex_bytes):
    shorts = {
        "airfoil_self_noise": "airfoil",
        "musk": "musk",
        "telemonitoring_parkinsons_biomedical_voice_measurements": "telemonitoring",
        "video_transcoding_time_prediction": "video",
        "garments_worker_productivity": "garments",
    }
    rows = []
    for family in beyond.get("families") or []:
        left = family["steps2048"]
        right = family["steps8192"]
        name = shorts.get(family["name"], family["name"].replace("_", r"\_"))
        rows.append(
            f"{name} & {tex_bytes(family['fit_rows'])} & {tex_bytes(left['bytes'])} & "
            f"{tex_bytes(right['bytes'])} & {sig3(left['validation_loss'])} & {sig3(right['validation_loss'])} \\\\"
        )
    if not rows:
        rows.append("not measured & --- & --- & --- & --- & --- \\\\")
    (out / "beyond-fit.tex").write_text(
        "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "Family & Fit rows & 2{,}048 B & 8{,}192 B & Val 2{,}048 & Val 8{,}192 \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def _write_provenance(side, out):
    lock = side["lock"]
    counts = side.get("provenance_counts") or {}
    overlap = ", ".join(f"\\texttt{{{item}}}" for item in lock.get("overlap_ids") or [])
    rows = [
        f"Catalog SHA-256 & \\texttt{{{lock.get('catalog_sha256') or 'not measured'}}} \\\\",
        f"Eligible / prepared / overlap exclusions & {lock.get('eligible')} / {lock.get('prepared')} / {lock.get('excluded')} \\\\",
        f"Unknown-rights catalog entries held out & {lock.get('unknown_rights')} \\\\",
        f"Split & \\texttt{{{str(lock.get('split_kind') or '').replace('_', chr(92)+'_')}}}, seed {lock.get('split_seed')} \\\\",
        f"Overlap lineage ids & {overlap} \\\\",
        f"Final evaluation authorized & {str(lock.get('final_evaluation_authorized')).lower()} \\\\",
        f"S3 worker directories / \\texttt{{train.csv}} / \\texttt{{validation.csv}} & {counts.get('s3_workers', 'not measured')} / {counts.get('s3_train_csv', 'not measured')} / {counts.get('s3_validation_csv', 'not measured')} \\\\",
        f"\\texttt{{test.csv}} under S3 workers & {counts.get('s3_worker_test_csv', 'not measured')} \\\\",
        f"BeyondArena workers / worker \\texttt{{test.csv}} & {counts.get('beyond_workers', 'not measured')} / {counts.get('beyond_worker_test_csv', 'not measured')} \\\\",
        f"Evaluator \\texttt{{test.csv}} files, contents not read & {counts.get('beyond_evaluator_test_csv', 'not measured')} \\\\",
    ]
    (out / "provenance-table.tex").write_text(
        "\\begin{tabular}{@{}lp{11cm}@{}}\n\\toprule\n"
        "Item & Record \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
