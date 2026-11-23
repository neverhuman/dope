//! `dope fit --format auto|v7|v8|v8-analytic|auto-v8` and the `DPA1` branch of `dope sample`.

use crate::args::{FitArgs, SampleArgs};
use crate::style::{group, Style};
use crate::v7_weights::load_model;
use crate::v8_weights::load_v8_model;
use dope_core::auto::v8::AutoV8Options;
use dope_core::auto::{
    fit_auto, fit_dpa1_v8, is_dpa1, parse_dpa1, AutoOptions, AutoReport, CandidateReport, Encoder,
    V7Decode, V7Decoder, V8Decoder, V8DpaMode, V8DpaReport, DEFAULT_AUTO_BUDGET,
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

/// Fit a `DSE8` v1 (`--format v8|v8-analytic|auto-v8`) `DPA1` tag-8 artifact within `--budget`.
pub(crate) fn run_fit_v8(
    a: &FitArgs,
    table: &RawTable,
    target: usize,
    format1: FitOptions,
) -> Result<(), String> {
    let budget = format1
        .budget
        .or(format1.size.map(ModelSize::budget))
        .unwrap_or(DEFAULT_AUTO_BUDGET);
    let mode = match a.format.as_str() {
        "v8-analytic" => V8DpaMode::V8Analytic,
        "auto-v8" => V8DpaMode::AutoV8,
        _ => V8DpaMode::V8,
    };
    let model = load_v8_model(a.v8_weights.as_deref())?;
    let opts = AutoV8Options {
        budget,
        format1,
        enable_trees: !a.no_trees,
        ..Default::default()
    };
    let fitted = fit_dpa1_v8(table, target, &opts, model.as_ref(), mode).map_err(e)?;
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
        eprintln!("{}", summary_v8(&fitted.report));
    }
    Ok(())
}

fn summary_v8(r: &V8DpaReport) -> String {
    format!(
        "{}: chose {} ({} of {} bytes); {}",
        r.mode, r.encoder, r.bytes, r.budget, r.note
    )
}

/// `dope sample` on a `DPA1` file. A Format 1 payload is returned so the Format 1 path
/// samples it unchanged; V7 (tag 7) and V8 (tag 8) payloads are decoded here (`None`).
pub(crate) fn sample_dpa1(
    a: &SampleArgs,
    bytes: Vec<u8>,
    style: Style,
) -> Result<Option<Vec<u8>>, String> {
    let not_v7 = |payload: Vec<u8>| {
        if a.v7_weights.is_some() || a.v7_zero_init {
            return Err("--v7-weights and --v7-zero-init apply only to V7 artifacts".to_string());
        }
        if a.v8_weights.is_some() || a.v8_zero_init {
            return Err("--v8-weights and --v8-zero-init apply only to V8 artifacts".to_string());
        }
        Ok(Some(payload))
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
            "a V7/V8 DPA1 artifact samples raw rows only (no --mode, --given or --space ml)".into(),
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
    if dpa.encoder == Encoder::V8 {
        if a.v7_weights.is_some() || a.v7_zero_init {
            return Err("--v7-weights and --v7-zero-init apply only to V7 artifacts".to_string());
        }
        let model = match a.v8_zero_init {
            true => None,
            false => load_v8_model(a.v8_weights.as_deref())?,
        };
        let decoder = V8Decoder::open(&dpa, model.as_ref(), a.v8_zero_init).map_err(e)?;
        decoder
            .write_csv(
                model.as_ref(),
                a.v8_zero_init,
                rows,
                a.seed,
                crate::writer(&a.out)?,
            )
            .map_err(e)?;
        if let (Some(p), true) = (&a.out, style.enabled) {
            println!(
                "{} sampled {} synthetic rows ({} features, V8 packet{}) from {} -> {}",
                style.green_bold("✓"),
                style.bold(group(rows)),
                decoder.features(),
                if a.v8_zero_init || !decoder.is_neural() {
                    ", analytic"
                } else {
                    ""
                },
                style.bold(a.model.display().to_string()),
                style.cyan_bold(p.display().to_string())
            );
        }
        return Ok(None);
    }
    if a.v8_weights.is_some() || a.v8_zero_init {
        return Err("--v8-weights and --v8-zero-init apply only to V8 artifacts".to_string());
    }
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
