//! `dope fit --format auto|v7` and the `DPA1` branch of `dope sample`.

use crate::args::{FitArgs, SampleArgs};
use crate::style::{group, Style};
use crate::v7_weights::load_model;
use dope_core::auto::{
    fit_auto, is_dpa1, parse_dpa1, AutoOptions, AutoReport, CandidateReport, Encoder, V7Decode,
    V7Decoder, DEFAULT_AUTO_BUDGET,
};
use dope_core::{DopeError, EmissionMode, FitOptions, ModelSize, RawTable};

fn e(err: DopeError) -> String {
    format!(
        "{err}\n  hint: {}\n  docs: {}",
        err.repair_hint(),
        err.docs_url()
    )
}

/// Fit within `--budget` (else the `--size` cap, else 2048 bytes) and write the `DPA1`
/// artifact: the better candidate (`--format auto`) or the V7 packet (`--format v7`).
/// `format1` carries the Format 1 flags of this command.
pub(crate) fn run_fit_auto(
    a: &FitArgs,
    table: &RawTable,
    target: usize,
    format1: FitOptions,
) -> Result<(), String> {
    let budget = format1
        .budget
        .or(format1.size.map(ModelSize::budget))
        .unwrap_or(DEFAULT_AUTO_BUDGET);
    let encoder = (a.format == "v7").then_some(Encoder::V7);
    let model = load_model(a.v7_weights.as_deref())?;
    let opts = AutoOptions {
        budget,
        format1,
        encoder,
    };
    let fitted = fit_auto(table, target, &opts, &model).map_err(e)?;
    std::fs::write(&a.out, &fitted.bytes).map_err(|x| x.to_string())?;
    if let Some(p) = &a.report {
        let text = serde_json::to_string_pretty(&fitted.report).map_err(|x| x.to_string())?;
        std::fs::write(p, text).map_err(|x| x.to_string())?;
    }
    if a.json {
        let text = serde_json::to_string(&fitted.report).map_err(|x| x.to_string())?;
        println!("{text}");
    }
    if !a.quiet {
        eprintln!("{}", summary(&fitted.report));
    }
    Ok(())
}

/// One line with the choice, byte counts and internal scores; no cell, header or extremum.
pub(crate) fn summary(r: &AutoReport) -> String {
    let other = |name: &str, c: &CandidateReport| match (c.requested, c.bytes, &c.error) {
        (false, _, _) => format!("{name} not requested"),
        (_, Some(b), _) => format!("{name} fitted ({b} B payload)"),
        (_, None, Some(err)) => {
            let kind = err.split(':').next().unwrap_or("error");
            format!("{name} failed ({kind})")
        }
        (_, None, None) => format!("{name} not fitted"),
    };
    let check = match (r.check.score_format1, r.check.score_v7) {
        (Some(f), Some(v)) => format!("internal check format1 {f:.3} vs v7 {v:.3}"),
        _ => format!("internal check {}", r.check.note),
    };
    let head = if r.mode == "auto" {
        format!("auto: chose {}", r.encoder)
    } else {
        format!("{} only: wrote {}", r.mode, r.encoder)
    };
    format!(
        "{head} ({} of {} bytes); {}; {}; {check}",
        r.bytes,
        r.budget,
        other("format1", &r.format1),
        other("v7", &r.v7)
    )
}

/// `dope sample` on a `DPA1` file. A Format 1 payload is returned so the Format 1 path
/// samples it unchanged; a V7 payload is decoded here (`None`) with the embedded weights,
/// the `--v7-weights` file or the zero-init decoder. Other files are returned as they are.
pub(crate) fn sample_dpa1(
    a: &SampleArgs,
    bytes: Vec<u8>,
    style: Style,
) -> Result<Option<Vec<u8>>, String> {
    let not_v7 = |payload: Vec<u8>| match a.v7_weights.is_some() || a.v7_zero_init {
        true => Err("--v7-weights and --v7-zero-init apply only to V7 artifacts".to_string()),
        false => Ok(Some(payload)),
    };
    if !is_dpa1(&bytes) {
        return not_v7(bytes);
    }
    let dpa = parse_dpa1(&bytes).map_err(e)?;
    if dpa.encoder == Encoder::Format1 {
        return not_v7(dpa.payload.to_vec());
    }
    let legacy = EmissionMode::parse(&a.mode).map_err(e)? == EmissionMode::Legacy;
    if !legacy || a.given.is_some() || a.space != "raw" {
        return Err(
            "a V7 artifact samples raw rows only (no --mode, --given or --space ml)".into(),
        );
    }
    let rows = if a.rows.trim().eq_ignore_ascii_case("auto") {
        1000
    } else {
        a.rows
            .trim()
            .parse::<usize>()
            .map_err(|_| "--rows must be a positive integer or 'auto'".to_string())?
    };
    let model = match a.v7_zero_init {
        true => None,
        false => Some(load_model(a.v7_weights.as_deref())?),
    };
    let decoder = match &model {
        Some(m) => V7Decoder::open(&dpa, m),
        None => V7Decoder::open_weightless(&dpa),
    }
    .map_err(e)?;
    let how = model.as_ref().map_or(V7Decode::ZeroInit, V7Decode::Trained);
    decoder
        .write_csv(how, rows, a.seed, crate::writer(&a.out)?)
        .map_err(e)?;
    if let (Some(p), true) = (&a.out, style.enabled) {
        println!(
            "{} sampled {} synthetic rows ({} features, V7 packet{}) from {} -> {}",
            style.green_bold("✓"),
            style.bold(group(rows)),
            decoder.features(),
            if a.v7_zero_init { ", zero-init" } else { "" },
            style.bold(a.model.display().to_string()),
            style.cyan_bold(p.display().to_string())
        );
    }
    Ok(None)
}
