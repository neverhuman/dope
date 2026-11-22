//! CLI dispatch for Fidelity Engine 2.0 Format 2 (`dope fit --format 2`, `dope sample`).

use crate::args::{FitArgs, SampleArgs};
use crate::style::{print_sample_saved_report, Style};
use dope_core::schema::Task;
use dope_core::structure::RuleMode;
use dope_core::{
    decode_format2_program, fit_v2, output, EmissionMode, EmitOptions, FitV2Options, RawTable,
};
use std::io::Write;
use std::path::PathBuf;

pub(crate) fn run_fit_v2(
    a: &FitArgs,
    table: &RawTable,
    target: usize,
    task: Option<Task>,
    budget: Option<usize>,
) -> Result<(), String> {
    let e = |err: dope_core::DopeError| {
        format!(
            "{err}\n  hint: {}\n  docs: {}",
            err.repair_hint(),
            err.docs_url()
        )
    };
    let rule_mode = match a.rules.as_str() {
        "structural" | "auto" => RuleMode::Structural,
        "all" => RuleMode::All,
        "off" | "none" => RuleMode::None,
        _ => return Err("--rules must be structural, all, or off".into()),
    };
    let teacher_exchange_dir = if let Some(rest) = a.teacher.strip_prefix("external:") {
        Some(PathBuf::from(rest))
    } else if a.teacher == "native" {
        None
    } else {
        return Err("--teacher must be 'native' or 'external:<DIR>'".into());
    };
    let opts = FitV2Options {
        budget,
        size: a.size,
        seed: a.seed,
        task,
        anonymize: a.anonymize,
        rule_mode,
        support_k: a.k_anon,
        teacher_exchange_dir,
    };
    let fitted = fit_v2(table, target, &opts).map_err(e)?;
    std::fs::write(&a.out, &fitted.bytes).map_err(|x| x.to_string())?;
    if let Some(p) = &a.report {
        std::fs::write(
            p,
            serde_json::to_string_pretty(&fitted.report).unwrap_or_default(),
        )
        .map_err(|x| x.to_string())?;
    }
    if a.json {
        println!(
            "{}",
            serde_json::to_string(&fitted.report).map_err(|x| x.to_string())?
        );
    } else if !a.quiet {
        println!(
            "DOPE Format 2 saved -> {} ({} B / {} B cap, rules={}, catf_trees={}, student_trees={}, ESS={:.2})",
            a.out.display(),
            fitted.report.bytes,
            fitted.report.budget_bytes,
            fitted.report.rules_mined,
            fitted.report.catf_trees,
            fitted.report.student_trees,
            fitted.report.projection_ess,
        );
    }
    Ok(())
}

pub(crate) fn run_sample_v2(
    a: &SampleArgs,
    bytes: &[u8],
    writer: Box<dyn Write>,
    style: Style,
) -> Result<(), String> {
    let e = |err: dope_core::DopeError| {
        format!(
            "{err}\n  hint: {}\n  docs: {}",
            err.repair_hint(),
            err.docs_url()
        )
    };
    let prog = decode_format2_program(bytes).map_err(e)?;
    let mode = EmissionMode::parse(&a.mode).map_err(e)?;
    let req_rows = if a.rows.trim().eq_ignore_ascii_case("auto") {
        prog.auto_rows(mode)
    } else {
        a.rows
            .trim()
            .parse::<usize>()
            .map_err(|_| "--rows must be a positive integer or 'auto'".to_string())?
    };
    let (samples, sidecar_data) = if let Some(given_csv) = &a.given {
        let raw = RawTable::read(given_csv).map_err(e)?;
        let data = prog.schema.to_model(&raw).map_err(e)?;
        (prog.sample_given(&data.x, a.seed).map_err(e)?, None)
    } else {
        let emitted = prog
            .sample_with_mode(&EmitOptions {
                mode,
                rows: req_rows,
                seed: a.seed,
                table_id: a.table_id,
                ..Default::default()
            })
            .map_err(e)?;
        let sc = emitted.weights.zip(emitted.pair_ids);
        (emitted.samples, sc)
    };
    let emitted_rows = samples.y.len();
    prog.write_raw(&samples, writer).map_err(e)?;
    if let Some((weights, pair_ids)) = sidecar_data {
        let sidecar_path = a.sidecar.clone().or_else(|| {
            a.out.as_ref().map(|p| {
                let mut s = p.as_os_str().to_os_string();
                s.push(".sidecar.csv");
                PathBuf::from(s)
            })
        });
        if let Some(sp) = sidecar_path {
            let sw =
                std::io::BufWriter::new(std::fs::File::create(&sp).map_err(|x| x.to_string())?);
            output::write_distill_sidecar(&weights, &pair_ids, sw).map_err(e)?;
        }
    }
    if let Some(p) = &a.out {
        print_sample_saved_report(
            &a.model,
            p,
            emitted_rows,
            prog.features(),
            &prog.schema.target.name,
            &a.space,
            style,
        );
    }
    Ok(())
}
