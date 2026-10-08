"""Format paper macros and short tables from the panel-stats payload.

Formatting only. The statistics live in compute_panel.py. A missing local
receipt extract becomes the words ``not measured'' rather than a guessed
number.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from public_hardware import banned_hits, origin_counts, origin_sentence, public_hardware


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


def _recorded_seconds(value):
    if value is None:
        return "not recorded"
    return _elapsed(value, lambda number: f"{number:,.1f}".replace(",", "{,}"))


def _recorded_hours(value):
    if value is None:
        return "not recorded"
    number = float(value)
    if number >= 1:
        return f"{number:.2f}"
    return f"{number:.3f}"


def _recorded_mib(value):
    """Resident bytes shown in MiB to one decimal. The JSON keeps the raw bytes."""
    if value is None:
        return "not recorded"
    return f"{float(value) / (1024 * 1024):.1f}"


def _tex_text(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _mfs_value(components):
    """Equation (3). Float arithmetic matches the number checker's illustration."""
    weights = (0.30, 0.20, 0.20, 0.15, 0.10, 0.05)
    epsilon = 1e-6
    total = sum(weights)
    log_score = sum(
        weight * math.log(epsilon + component) for weight, component in zip(weights, components)
    )
    return 100.0 * math.exp(log_score / total)


def _format_mfs(value, places):
    return f"{value:.{places}f}"


def mfs_illustrations():
    """Invented rows. Classic stays two decimals; the epsilon rows keep the tail."""
    classic = _mfs_value((0.90, 0.80, 0.85, 0.70, 0.75, 0.95))
    ones = _mfs_value((1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    at_cap = _mfs_value((0.90, 0.80, 0.85, 0.70, 0.75, 0.0))
    return {
        "classic": _format_mfs(classic, 2),
        "all_ones": _format_mfs(ones, 4),
        "zero_compactness": _format_mfs(at_cap, 4),
    }


def retention_algebra_sentence():
    """Invented losses. Not a measured lineage."""
    null_loss = 1.0
    real_loss = 0.01
    retention = 0.94
    synthetic = real_loss + (1.0 - retention) * (null_loss - real_loss)
    ratio = synthetic / real_loss
    return (
        f"Invented losses ${null_loss:g}$, ${real_loss:.2f}$, and retention "
        f"${retention:.2f}$ give a synthetic-trained loss of ${synthetic:.4f}$, "
        f"which is ${ratio:.2f}$ times the real-trained mean squared error. "
        "The CatBoost median in this paper is not this example."
    )


def _hardware_sentence(row):
    """Prefer the sealed public sentence. Both PDF builds use it."""
    sentence = row.get("hardware")
    if not isinstance(sentence, str) or not sentence:
        sentence = public_hardware(row.get("hosts") or [], row.get("gpu_by_host") or {})
    if banned_hits(sentence):
        raise ValueError("hardware sentence contains a private token")
    return _tex_text(sentence)


def _load_generated(out, name):
    path = out / name
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing; run the paper extract before emit")
    document = json.loads(path.read_text())
    if document.get("official_tests_opened"):
        raise ValueError(f"{name} opened an official test")
    return document


def _time_cell(lines, command, prefix, stem, median, maximum):
    if median is None:
        return r"\NotRecorded"
    med_name = f"{prefix}{stem}Med"
    max_name = f"{prefix}{stem}Max"
    lines.append(command(med_name, _recorded_seconds(median)))
    lines.append(command(max_name, _recorded_seconds(maximum)))
    return f"\\{med_name} (\\{max_name})"


def _memory_cell(lines, command, prefix, ram_bytes, vram_mib):
    if ram_bytes is None and vram_mib is None:
        return r"\NotRecorded"
    ram_name = f"{prefix}PeakRam"
    vram_name = f"{prefix}PeakVram"
    lines.append(command(ram_name, _recorded_mib(ram_bytes)))
    lines.append(command(vram_name, "not recorded" if vram_mib is None else str(int(vram_mib))))
    left = r"\NotRecorded" if ram_bytes is None else f"\\{ram_name}"
    right = r"\NotRecorded" if vram_mib is None else f"\\{vram_name}"
    return f"{left} / {right}"


def _gpu_hour_cell(lines, command, prefix, gpu_hours):
    gpu_name = f"{prefix}GpuHours"
    lines.append(command(gpu_name, _recorded_hours(gpu_hours)))
    return f"\\{gpu_name}"


def _cost_row_is_measured(row):
    """A cost row needs a fit timer, both memory peaks, and GPU hours."""
    return all(
        row.get(key) is not None
        for key in (
            "fit_median_seconds",
            "fit_max_seconds",
            "peak_ram_bytes",
            "peak_vram_mib",
            "gpu_hours",
        )
    )


def _span_cell(median_name, lo_name, hi_name):
    return f"\\{median_name} (\\{lo_name}--\\{hi_name})"


def _architecture_commands(command):
    """Displayed-generator constants, read from the fitter and the data lock."""
    root = Path(__file__).resolve().parents[3]
    rust = (root / "rust/compiler/target_fitting/part_02.rs").read_text()
    shape = re.search(r'"features12_steps2048" => \(12, (\d+), (\d+)\)', rust)
    rate = re.search(r"\.build\(&store, ([0-9.eE+-]+)\)", rust)
    production = re.search(
        r"Ok\(\(12, feature_count\.min\(12\)\.clamp\((\d+), (\d+)\), (\d+)\)\)",
        rust,
    )
    lock = (root / "research/benchmark/results/s3-data.lock.json").read_text()
    split = re.search(r"official_test_grouped_training_(\d+)_(\d+)", lock)
    if shape is None or rate is None or split is None or production is None:
        raise ValueError("architecture source constants are missing")
    steps = f"{int(shape.group(2)):,}".replace(",", "{,}")
    return [
        command("ArchSplitMajor", split.group(1)),
        command("ArchSplitMinor", split.group(2)),
        command("ArchWidth", shape.group(1)),
        command("ArchSteps", steps),
        command("ArchLr", f"{float(rate.group(1)):.3f}"),
        command("ProductionWidthLo", production.group(1)),
        command("ProductionWidthHi", production.group(2)),
        command("ProductionSteps", production.group(3)),
    ]


def _arf_byte_commands(lines, command, arf, tex_bytes):
    """ARF byte span. The within-cap count is used only when the span decides it."""
    artifact = arf.get("artifact_bytes") or {}
    minimum = artifact.get("min")
    maximum = artifact.get("max")
    count = artifact.get("n")
    median = artifact.get("median")
    if artifact.get("field") != "selected_charged_bytes" or None in (minimum, maximum, count, median):
        raise ValueError("ARF charged-byte span is missing")
    if type(count) is not int or count <= 0:
        raise ValueError("ARF charged-byte count is empty")
    cap = 10240
    if minimum > cap:
        within = 0
    elif maximum <= cap:
        within = count
    else:
        raise ValueError("ARF within-cap count is not determined by the byte span")
    lines.append(command("ArfByteLo", f"{int(minimum):,}".replace(",", "{,}")))
    lines.append(command("ArfByteHi", f"{int(maximum):,}".replace(",", "{,}")))
    lines.append(command("ArfByteMedian", _byte_median(median, tex_bytes)))
    return f"ARF & {_byte_median(median, tex_bytes)} & {within}/{count} \\\\"


def _write_compute_cost(out, lines, command, tex_bytes):
    document = _load_generated(out, "compute-cost.json")
    if document.get("missing_field") != "not recorded":
        raise ValueError("compute-cost missing_field is not the required phrase")
    methods = document["methods"]
    lines.append(command("NotRecorded", "not recorded"))
    lines.append(command("FigWidth", "7.16in"))
    lines.append(command("ForestTimeN", str(int(methods["Forest-Flow"]["fit_n"]))))
    for key, prefix in (
        ("GaussianCopula", "Gauss"),
        ("Chow-Liu", "Chow"),
        ("independent_marginals", "Ind"),
    ):
        row = methods[key]
        if row.get("attempt_median_seconds") is None:
            raise ValueError(f"{key} attempt wall_seconds was not recorded")
        lines.append(command(f"{prefix}AttemptMed", _recorded_seconds(row["attempt_median_seconds"])))
        lines.append(command(f"{prefix}AttemptMax", _recorded_seconds(row["attempt_max_seconds"])))
    rows = []
    measured = (
        ("DOPE", "DOPE", "Dope", _span_cell("DopeByteMedian", "DopeByteLo", "DopeByteHi")),
        ("Gaussian copula", "GaussianCopula", "Gauss", _span_cell("GaussByteMedian", "GaussByteLo", "GaussByteHi")),
        ("Chow--Liu", "Chow-Liu", "Chow", _span_cell("ChowByteMedian", "ChowByteLo", "ChowByteHi")),
        ("Independent marginals", "independent_marginals", "Ind", _span_cell("IndByteMedian", "IndByteLo", "IndByteHi")),
        ("CTGAN", "CTGAN", "Ctgan", _span_cell("CtganByteMedian", "CtganByteLo", "CtganByteHi")),
        ("TVAE", "TVAE", "Tvae", _span_cell("TvaeByteMedian", "TvaeByteLo", "TvaeByteHi")),
        ("Forest-Flow", "Forest-Flow", "Forest", _span_cell("ForestByteMedian", "ForestByteLo", "ForestByteHi")),
    )
    for label, key, prefix, byte_cell in measured:
        row = methods[key]
        if not _cost_row_is_measured(row):
            continue
        sentence = _hardware_sentence(row)
        lines.append(command(f"{prefix}Hardware", sentence))
        lines.append(command(f"{prefix}HardwareAnon", sentence))
        fit = _time_cell(lines, command, prefix, "Fit", row["fit_median_seconds"], row["fit_max_seconds"])
        _time_cell(lines, command, prefix, "Samp", row["sample_median_seconds"], row["sample_max_seconds"])
        memory = _memory_cell(lines, command, prefix, row["peak_ram_bytes"], row["peak_vram_mib"])
        hours = _gpu_hour_cell(lines, command, prefix, row["gpu_hours"])
        rows.append(
            f"{label} & {fit} & {memory} & \\{prefix}Hardware & {hours} & {byte_cell} \\\\"
        )
    if not rows:
        raise ValueError("compute-cost table has no measured fit row")
    arf_bytes = _arf_byte_commands(lines, command, methods["ARF"], tex_bytes)
    columns = (
        "@{}"
        ">{\\raggedright\\arraybackslash}p{1.15in}"
        ">{\\raggedright\\arraybackslash}p{0.95in}"
        ">{\\raggedright\\arraybackslash}p{0.95in}"
        ">{\\raggedright\\arraybackslash}p{1.7in}"
        ">{\\raggedright\\arraybackslash}p{0.55in}"
        ">{\\raggedright\\arraybackslash}p{1.35in}@{}"
    )
    table = (
        "\\begin{tabular}{" + columns + "}\n"
        "\\toprule\n"
        "Method & Fit s & RAM / VRAM & Hardware & GPU h & Bytes \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (out / "compute-cost-table.tex").write_text(table)
    return arf_bytes


def _license_bits(entry):
    if not entry or not entry.get("name"):
        return None
    bits = [_tex_text(entry["name"])]
    if entry.get("licensor"):
        bits.append("licensor " + _tex_text(entry["licensor"]))
    if entry.get("licensed_work"):
        bits.append("licensed work " + _tex_text(entry["licensed_work"]))
    if entry.get("copyright"):
        bits.append(_tex_text(entry["copyright"]).rstrip("."))
    return ", ".join(bits)


def _write_availability(out):
    licenses = _load_generated(out, "licenses.json")
    methods = licenses["methods"]
    pmlb = licenses["pmlb"]
    if pmlb.get("spdx") != "MIT" or pmlb.get("note") != "source_license: MIT (PMLB)":
        raise ValueError("PMLB lock record is not MIT")
    repository = licenses["repository"]
    if repository.get("name") != "MIT License":
        raise ValueError("repository license heading is not MIT License")
    ctgan = _license_bits(methods["CTGAN"])
    tvae = _license_bits(methods["TVAE"])
    if ctgan is None or ctgan != tvae:
        raise ValueError("CTGAN and TVAE LICENSE reads disagree")
    copulas = _license_bits(methods["GaussianCopula"])
    tabsyn = _license_bits(methods["TabSyn"])
    tabddpm = _license_bits(methods["TabDDPM"])
    forest = _license_bits(methods["ForestDiffusion/Forest-Flow"])
    arf = _license_bits(methods["ARF"])
    study = _license_bits(repository)
    for label, bits in (
        ("Copulas", copulas),
        ("TabSyn", tabsyn),
        ("TabDDPM", tabddpm),
        ("Forest-Diffusion", forest),
        ("ARF", arf),
    ):
        if not bits:
            raise ValueError(f"{label} LICENSE heading was not read")
    sdv = licenses.get("sdv") or {}
    if sdv.get("name"):
        raise ValueError("SDV license was filled without a pinned LICENSE file")
    paragraph = (
        "Regenerate every \\texttt{generated/} file, the figure PDFs, and both manuscript PDFs "
        "with \\texttt{just paper} from the commit that contains these files. That recipe reads "
        "the committed validation ledgers, original fit metadata, and compressed numeric replay logs. "
        "The replay cost, loss curves, and BeyondArena extracts have file and field mappings in "
        "\\texttt{generated/original-field-map.json}. Those original metadata snapshots were captured "
        "after the historical runs; they do not reconstruct historical custody or replace scored artifacts. "
        "Pinned LICENSE extracts are reread when their local store is available. "
        "It does not open an official test file. "
        "The catalog hash is \\CatalogSha. The grouped training split uses seed $\\SplitSeed$. "
        "The primary displayed DOPE and legacy comparison fits use seed 11 and sample seeds 101, 211, and 307. "
        "Later retained panels document additional ARF and Forest-Flow fit seeds and the measured "
        "TabSyn and TabDDPM subsets separately. "
        "\\texttt{python3 research/benchmark/verify\\_paper\\_numbers.py} exits nonzero when a "
        "displayed macro disagrees with those ledgers. Full neural baseline coverage, the "
        "pre-specified privacy-attack panel, displayed-DOPE fit-seed variance, and the full ablation grid are "
        "named in the prose and omitted from the tables until those runs exist. The journal "
        "fidelity table is a separate "
        "empirical ledger on the grouped validation split. It is not the pre-specified privacy-attack "
        "panel, and it is not differential privacy or HIPAA de-identification.\n\n"
        "PMLB is MIT. Every dataset entry in the data lock records SPDX MIT and the note that "
        "the source license is MIT (PMLB). "
        f"CTGAN and TVAE share one pinned LICENSE file: {ctgan}. "
        f"The Gaussian copula uses the pinned Copulas LICENSE: {copulas}. "
        f"TabSyn is {tabsyn}. "
        f"TabDDPM is {tabddpm}. "
        f"Forest-Diffusion is {forest}. "
        f"ARF is {arf}. "
        f"Chow--Liu and the independent marginals are implementations written for this comparison, under this repository's license: {study}. "
        "The SDV repository LICENSE is not in the pinned snapshot, so that license is not recorded.\n"
    )
    (out / "availability.tex").write_text(paragraph)


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


def _wtl(pair):
    return f"{int(pair['wins'])}/{int(pair['ties'])}/{int(pair['losses'])}"


def _publication_phrase(display):
    """Receipt median from the B-lane publication, or ``not measured``."""
    from publication_rows import phrase

    return phrase(display)


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
        command("WtlGaussCb", _wtl(density["pairs"]["GaussianCopula"])),
        command("WtlLinGauss", _wtl(linear["pairs"]["GaussianCopula"])),
        command("WtlMlpGauss", _wtl(payload["blocks"]["density"]["mlp"]["pairs"]["GaussianCopula"])),
        command("WtlTvaeCb", _wtl(neural["pairs"]["TVAE"])),
        command("WtlTvaeLin", _wtl(payload["blocks"]["neural"]["linear"]["pairs"]["TVAE"])),
        command("WtlTvaeMlp", _wtl(payload["blocks"]["neural"]["mlp"]["pairs"]["TVAE"])),
        command("WtlForestCb", _wtl(forest["pairs"]["Forest-Flow"])),
        command("WtlForestLin", _wtl(payload["blocks"]["forest"]["linear"]["pairs"]["Forest-Flow"])),
        command("WtlForestMlp", _wtl(payload["blocks"]["forest"]["mlp"]["pairs"]["Forest-Flow"])),
        command("ForestByteLo", tex_bytes(payload["forest_bytes"]["min"])),
        command("ForestByteHi", tex_bytes(payload["forest_bytes"]["max"])),
        command("ForestByteMedian", _byte_median(side.get("forest_byte_median"), tex_bytes)),
        command("DopeForestByteMedian", _byte_median(side.get("dope_forest_byte_median"), tex_bytes)),
        command("DopeWithinCap", f"{dope_bytes['within_cap']}/{dope_bytes['known']}"),
        command("DopeByteMedian", _byte_median(dope_bytes["median"]["median"], tex_bytes)),
        command("DopeByteLo", tex_bytes(side.get("dope_byte_min"))),
        command("TabSynPanel", "not measured"),
        command("TabDDPMPanel", "not measured"),
        command("TabSynResult", "measured on a retained validation subset"),
        command("TabDDPMResult", _publication_phrase("TabDDPM")),
        command("ForestPublication", _publication_phrase("Forest-Flow")),
        command("ArfResult", _publication_phrase("ARF")),
        command("PrivacyAttackPanel", "not measured"),
        command("FitSeedVariance", "not measured for the displayed DOPE generator"),
        command("FullAblationGrid", "not measured"),
        command("CampaignElapsed", "not measured"),
    ]
    byte_rows = []
    order = (
        ("DOPE reference", None),
        ("Gaussian copula", "GaussianCopula"),
        ("Chow--Liu", "Chow-Liu"),
        ("Independent marginals", "independent_marginals"),
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
        lines.append(command(f"{macro}ByteLo", tex_bytes(pack["lo"])))
        lines.append(command(f"{macro}ByteHi", tex_bytes(pack["hi"])))
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
    record_path = Path(__file__).resolve().parents[3] / "research/benchmark/results/s3-lineage-record.json"
    ledger = json.loads(record_path.read_text())
    names = [row["display_name"] for row in ledger["rows"]]
    counts = origin_counts(names)
    if sum(counts.values()) != len(names):
        raise ValueError("origin counts do not cover the lineage ledger")
    for bucket, macro in (
        ("feynman", "OriginFeynman"),
        ("strogatz", "OriginStrogatz"),
        ("fri", "OriginFri"),
        ("bng", "OriginBng"),
        ("other", "OriginOther"),
    ):
        lines.append(command(macro, str(counts[bucket])))
    origin = origin_sentence(counts)
    if banned_hits(origin):
        raise ValueError("origin sentence contains a private token")
    lines.append(command("OriginSentence", origin))
    shown = mfs_illustrations()
    lines.append(command("MfsClassic", shown["classic"]))
    lines.append(command("MfsAllOnes", shown["all_ones"]))
    lines.append(command("MfsZeroCompact", shown["zero_compactness"]))
    algebra = retention_algebra_sentence()
    if banned_hits(algebra):
        raise ValueError("retention algebra sentence contains a private token")
    lines.append(command("RetentionAlgebra", algebra))
    lines.append(command(
        "DcrExchange",
        "Under an independent draw, the unbalanced share sits near the fit-view fraction "
        "of the pooled references. The grouped split keeps about four fifths of those rows "
        "in the fit view, so the baseline is near four fifths, not one half.",
    ))
    lines.append(command("ThrSentence", _threshold_sentence(payload["threshold_robustness"]["DOPE"]["catboost"], sig3)))
    lines.extend(_expanded_commands(payload["expanded"], sig3, tex_p, command))
    beyond = side.get("beyond") or {}
    lines.append(command("BeyondElapseA", _elapsed(_median_elapsed(beyond, "steps2048"), tex_bytes)))
    lines.append(command("BeyondElapseB", _elapsed(_median_elapsed(beyond, "steps8192"), tex_bytes)))
    lines.extend(_architecture_commands(command))
    byte_rows.append(_write_compute_cost(out, lines, command, tex_bytes))
    _write_availability(out)
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
    _write_fidelity(payload["expanded"], out, sig3, tex_p)
    _write_loss_beside(payload["expanded"], out, sig3)
    _write_threshold_counts(payload["expanded"], out)
    _write_coreset(payload["expanded"], out)
    _write_hashes(payload["expanded"], out)


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


def _span(summary, sig3):
    if not summary or not summary.get("n"):
        return "---"
    return f"{sig3(summary['median'])} [{sig3(summary['lo'])}, {sig3(summary['hi'])}] ({summary['n']})"


def _contrast(pair, sig3, tex_p):
    if not pair or not pair.get("n"):
        return "---"
    shown = tex_p(pair.get("holm_p"))
    if shown != "---":
        shown = f"${shown}$"
    return (
        "\\shortstack[c]{"
        f"{sig3(pair['median_difference'])} [{sig3(pair['lo'])}, {sig3(pair['hi'])}]"
        "\\\\ "
        f"{shown}; {sig3(pair.get('rank_biserial'))}; {pair['n']}"
        "}"
    )


def _expanded_commands(expanded, sig3, tex_p, command):
    loss = expanded["loss"]
    counts = expanded["threshold_counts"]
    coreset = expanded["coreset"]
    lines = [
        command("CoreMin", str(coreset["fit_rows_min"])),
        command("CoreMax", str(coreset["fit_rows_max"])),
        command("CodeCommit", expanded["source_commit"]),
        command("NThrLowCb", str(counts["0.001"]["catboost"])),
        command("NThrLowLin", str(counts["0.001"]["linear"])),
        command("NThrLowMlp", str(counts["0.001"]["mlp"])),
        command("NThrMidCb", str(counts["0.01"]["catboost"])),
        command("NThrMidLin", str(counts["0.01"]["linear"])),
        command("NThrMidMlp", str(counts["0.01"]["mlp"])),
        command("NThrHighCb", str(counts["0.05"]["catboost"])),
        command("NThrHighLin", str(counts["0.05"]["linear"])),
        command("NThrHighMlp", str(counts["0.05"]["mlp"])),
    ]
    for auditor, suffix in (("catboost", "Cb"), ("linear", "Lin"), ("mlp", "Mlp")):
        block = loss[auditor]
        lines.append(command(f"RmseReal{suffix}", sig3(block["real_rmse"]["median"])))
        lines.append(command(f"RmseSyn{suffix}", sig3(block["synthetic_rmse"]["median"])))
        lines.append(command(f"RtwoReal{suffix}", sig3(block["real_r2"]["median"])))
        lines.append(command(f"RtwoSyn{suffix}", sig3(block["synthetic_r2"]["median"])))
    return lines


def _method_rows(block, labels, sig3):
    rows = []
    for method, label in labels:
        metrics = block["methods"][method]
        cells = [_span(metrics[name], sig3) for name in (
            "alpha_precision", "beta_recall", "share_closer_to_train", "domias_auc")]
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    return rows


def _contrast_rows(block, labels, sig3, tex_p):
    rows = []
    for method, label in labels:
        pairs = block["pairs"][method]
        cells = [_contrast(pairs[name], sig3, tex_p) for name in (
            "alpha_precision", "beta_recall", "share_closer_to_train", "domias_auc")]
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    return rows


def _write_fidelity(expanded, out, sig3, tex_p):
    density_labels = (
        ("DOPE", "DOPE"),
        ("GaussianCopula", "Gaussian"),
        ("Chow-Liu", "Chow--Liu"),
        ("independent_marginals", "Indep."),
    )
    neural_labels = (("DOPE", "DOPE"), ("CTGAN", "CTGAN"), ("TVAE", "TVAE"))
    contrast_labels = (
        ("GaussianCopula", "DOPE$-$Gaussian"),
        ("Chow-Liu", "DOPE$-$Chow--Liu"),
        ("independent_marginals", "DOPE$-$Indep."),
    )
    neural_contrasts = (("CTGAN", "DOPE$-$CTGAN"), ("TVAE", "DOPE$-$TVAE"))
    header = (
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Method & $\\alpha$-prec. & $\\beta$-recall & \\shortstack{DCR share\\\\(unbalanced)} & DOMIAS \\\\\n"
        "\\multicolumn{5}{@{}l@{}}{\\emph{Median [interval] (lineages).}} \\\\\n\\midrule\n"
    )
    body = "\n".join(_method_rows(expanded["blocks"]["density"], density_labels, sig3))
    body += "\n\\midrule\n" + "\n".join(_method_rows(expanded["blocks"]["neural"], neural_labels, sig3))
    contrast = (
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Contrast & $\\alpha$-prec. & $\\beta$-recall & \\shortstack{DCR share\\\\(unbalanced)} & DOMIAS \\\\\n"
        "\\multicolumn{5}{@{}l@{}}{\\emph{Line 2: Holm $p$; rank-biserial $r$; $n$. DOPE minus comparator.}} \\\\\n"
        "\\midrule\n"
        + "\n".join(_contrast_rows(expanded["blocks"]["density"], contrast_labels, sig3, tex_p))
        + "\n\\midrule\n"
        + "\n".join(_contrast_rows(expanded["blocks"]["neural"], neural_contrasts, sig3, tex_p))
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (out / "fidelity-privacy-table.tex").write_text(
        header + body + "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{0.5em}\n" + contrast
    )


def _write_threshold_counts(expanded, out):
    counts = expanded["threshold_counts"]
    rows = []
    for key in ("0.001", "0.01", "0.05"):
        row = counts[key]
        rows.append(f"{key} & {row['catboost']} & {row['linear']} & {row['mlp']} \\\\")
    (out / "threshold-counts.tex").write_text(
        "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n"
        "Threshold & CatBoost & Linear & MLP \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def _write_loss_beside(expanded, out, sig3):
    rows = []
    for auditor, label in (("catboost", "CatBoost"), ("linear", "Linear"), ("mlp", "MLP")):
        block = expanded["loss"][auditor]
        rows.append(
            f"{label} & {block['n']} & {sig3(block['real_rmse']['median'])} & "
            f"{sig3(block['synthetic_rmse']['median'])} & {sig3(block['real_r2']['median'])} & "
            f"{sig3(block['synthetic_r2']['median'])} \\\\"
        )
    (out / "loss-beside-retention.tex").write_text(
        "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "Auditor & $n$ & RMSE real & RMSE synth. & "
        "\\shortstack{training-mean\\\\skill, real} & \\shortstack{training-mean\\\\skill, synth.} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def _bin_phrase(coreset):
    parts = []
    for bins_, count in sorted(coreset["bins"].items(), key=lambda item: int(item[0])):
        noun = "lineage" if count == 1 else "lineages"
        parts.append(f"{count} {noun} at {bins_} bins")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _write_coreset(expanded, out):
    coreset = expanded["coreset"]
    bins = _bin_phrase(coreset)
    text = (
        "The fit view is not the full PMLB table. Frozen catalog metadata draws each "
        "official training and test file by quota stratification on the target distribution "
        "and high-variance feature quantile bins, aiming at "
        f"{coreset['train_target_rows']} training rows and {coreset['test_target_rows']} test rows, "
        f"with at most {coreset['final_max_rows']} rows in total and at least "
        f"{coreset['minimum_final_rows']} rows. A source smaller than that quota keeps every row "
        f"and still reserves a test share of at least {coreset['minimum_test_to_train_ratio_when_small']}. "
        f"Of the {coreset['lineages']} prepared lineages, {coreset['not_downsampled']} were already "
        "inside the quota. The other lineages use a hybrid coreset that keeps the "
        f"observed feature and target range; the bin counts are {bins}. "
        "The generator then groups identical official-training rows and assigns those groups "
        f"with seed {coreset['split_seed']} so that about 80\\% of the rows become the fit view "
        "and the rest become the validation holdout. Duplicate rows are not split across that cut. "
        f"On these lineages the fit view has {coreset['fit_rows_min']} to {coreset['fit_rows_max']} rows. "
        "The bound keeps every lineage on one worker view and leaves the official test sealed. "
        "It is not a second draw at scoring time.\n"
    )
    (out / "coreset-paragraph.tex").write_text(text)


def _break_path(path):
    pieces = []
    for piece in path.split("/"):
        escaped = piece.replace("_", r"\_")
        bits = [escaped[index : index + 16] for index in range(0, len(escaped), 16)]
        pieces.append(r"\allowbreak{}".join(bits))
    return r"/\allowbreak{}".join(pieces)


def _write_hashes(expanded, out):
    rows = [
        f"Code commit of the measured harness & \\texttt{{{_break_hash(expanded['source_commit'])}}} \\\\",
        f"Harness SHA-256 & \\texttt{{{_break_hash(expanded['harness_sha256'])}}} \\\\",
    ]
    for item in expanded["hashes"]:
        rows.append(
            f"\\texttt{{{_break_path(item['path'])}}} & \\texttt{{{_break_hash(item['sha256'])}}} \\\\"
        )
    (out / "artifact-hashes.tex").write_text(
        "{\\scriptsize\n"
        "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{8.6cm}"
        ">{\\raggedright\\arraybackslash}p{8.6cm}@{}}\n\\toprule\n"
        "Ledger or lock & SHA-256 \\\\\n\\midrule\n\\endhead\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{longtable}\n}\n"
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
