//! `dope` — compile tables into tiny generative programs.

use clap::Parser;
use dope_core::encode::{MlEncoder, Scaling};
use dope_core::eval::{evaluate, frontier, EvalOptions, FrontierOptions};
use dope_core::schema::{infer, IntakeOptions, Task};
use dope_core::{
    byte_report, decode, device, fit, output, sample, FitOptions, RawTable, SampleOptions,
};
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

mod args;
use args::*;

fn parse_bytes(s: &str) -> Result<usize, String> {
    let t = s.trim().to_ascii_lowercase();
    let t = t.trim_end_matches("ib").trim_end_matches('b');
    let (num, mul) = match t.chars().last() {
        Some('k') => (&t[..t.len() - 1], 1024.0),
        Some('m') => (&t[..t.len() - 1], 1024.0 * 1024.0),
        _ => (t, 1.0),
    };
    num.parse::<f64>()
        .map(|v| (v * mul).round() as usize)
        .map_err(|_| format!("bad byte size: {s}"))
}

fn writer(path: &Option<PathBuf>) -> Result<Box<dyn Write>, String> {
    Ok(match path {
        Some(p) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(p).map_err(|e| e.to_string())?,
        )),
        None => Box::new(std::io::BufWriter::new(std::io::stdout())),
    })
}

fn group(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn run(cli: Cli) -> Result<(), String> {
    let e = |err: dope_core::DopeError| {
        format!(
            "{err}\n  hint: {}\n  docs: {}",
            err.repair_hint(),
            err.docs_url()
        )
    };
    match cli.command {
        Command::Fit(a) => {
            let table = RawTable::read(&a.csv).map_err(e)?;
            let target = table.column_index(&a.target).map_err(e)?;
            let task = match a.task.as_deref() {
                None => None,
                Some("regression") => Some(Task::Regression),
                Some("classification") => Some(Task::Classification),
                Some(_) => return Err("--task must be regression or classification".into()),
            };
            let plan = match &a.plan {
                None => None,
                Some(p) => {
                    let v: Vec<usize> =
                        p.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                    if v.len() != 3 {
                        return Err("--plan must be knots,inputs,hidden".into());
                    }
                    Some((v[0], v[1], v[2]))
                }
            };
            let opts = FitOptions {
                budget: a.budget.as_deref().map(parse_bytes).transpose()?,
                seed: a.seed,
                task,
                anonymize: a.anonymize,
                device: device::parse(&a.device).map_err(e)?,
                plan,
                dependence: a.dependence.clone(),
                tolerance: a.tolerance,
                regimes: a.regimes,
                trees: !a.no_trees,
                audit: !a.no_audit,
                ..Default::default()
            };
            let fitted = fit(&table, target, &opts).map_err(e)?;
            std::fs::write(&a.out, &fitted.bytes).map_err(|x| x.to_string())?;
            if let Some(p) = &a.report {
                std::fs::write(
                    p,
                    serde_json::to_string_pretty(&fitted.report).unwrap_or_default(),
                )
                .map_err(|x| x.to_string())?;
            }
            if !a.quiet {
                let r = &fitted.report;
                println!(
                    "compiled {} -> {}  ({} bytes, budget {})",
                    a.csv.display(),
                    a.out.display(),
                    group(r.bytes),
                    group(r.budget)
                );
                println!(
                    "  task         {} · {} rows × {} features",
                    r.task,
                    group(r.rows),
                    r.features
                );
                println!(
                    "  program      {} knots · {} dependence × {} regime(s) · net {}→{} (bits w{} b{} β{} v{}) · {} trees of depth {}",
                    r.knots, r.dependence, r.regimes, r.inputs, r.hidden, r.bits[0], r.bits[1], r.bits[2], r.bits[3], r.trees, r.depth
                );
                println!(
                    "  stored gain  {:.3} of the null loss explained on held-out rows",
                    r.stored_gain
                );
                if let Some(a) = r.self_audit {
                    println!(
                        "  self-audit   robust retention {a:.3} (linear and tree auditors trained on samples, scored on held-out training rows)"
                    );
                }
                println!(
                    "  drivers      {}",
                    r.screened
                        .iter()
                        .take(6)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                println!(
                    "  time         {:.1} s on {}",
                    r.seconds,
                    device::describe(&opts.device)
                );
            }
        }
        Command::Sample(a) => {
            let program =
                decode(&std::fs::read(&a.model).map_err(|x| x.to_string())?).map_err(e)?;
            let opts = SampleOptions {
                rows: a.rows,
                seed: a.seed,
                device: device::parse(&a.device).map_err(e)?,
                ..Default::default()
            };
            let s = sample(&program, &opts).map_err(e)?;
            let w = writer(&a.out)?;
            match a.space.as_str() {
                "raw" => output::write_raw(&program, &s, w).map_err(e)?,
                "ml" => {
                    let scaling =
                        Scaling::parse(&a.scaling).ok_or("--scaling must be unit or quantile")?;
                    output::write_ml(&MlEncoder::new(&program, scaling), &s, w).map_err(e)?
                }
                _ => return Err("--space must be raw or ml".into()),
            }
        }
        Command::Predict(a) => {
            let program =
                decode(&std::fs::read(&a.model).map_err(|x| x.to_string())?).map_err(e)?;
            let table = RawTable::read(&a.csv).map_err(e)?;
            let data = program.schema.to_model(&table).map_err(e)?;
            let n = table.rows();
            let pred = program.predict_model(&data.x, n);
            let mut w = writer(&a.out)?;
            let headers: Vec<String> = match &program.schema.target.kind {
                dope_core::schema::Kind::Numeric(_) => {
                    vec![format!("{}_pred", program.schema.target.name)]
                }
                dope_core::schema::Kind::Categorical(c) => {
                    c.levels.iter().map(|l| format!("p({l})")).collect()
                }
            };
            writeln!(w, "{}", headers.join(",")).map_err(|x| x.to_string())?;
            for row in pred {
                let cells: Vec<String> = if row.len() == 1 {
                    vec![dope_core::schema::format_number(
                        program.regression_to_raw(row[0]),
                    )]
                } else {
                    row.iter().map(|p| format!("{p:.6}")).collect()
                };
                writeln!(w, "{}", cells.join(",")).map_err(|x| x.to_string())?;
            }
        }
        Command::Inspect(a) => {
            let bytes = std::fs::read(&a.model).map_err(|x| x.to_string())?;
            let program = decode(&bytes).map_err(e)?;
            let report = byte_report(&bytes).map_err(e)?;
            if a.json {
                let v = serde_json::json!({
                    "bytes": report,
                    "task": program.schema.task,
                    "features": program.schema.features.iter().map(|f| &f.name).collect::<Vec<_>>(),
                    "target": program.schema.target.name,
                    "dependence": program.dependence.name(),
                    "screened": program.screened.iter().map(|&j| &program.schema.features[j].name).collect::<Vec<_>>(),
                    "net": [program.net.inputs, program.net.hidden, program.net.outputs],
                    "dropped": program.schema.dropped,
                });
                println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            } else {
                println!(
                    "{}  {} bytes ({} brotli program + {} safetensors framing)",
                    a.model.display(),
                    group(report.total),
                    group(report.stored),
                    group(report.header)
                );
                println!("  inflated, the program holds:");
                println!("  {:<18} {:>7} bytes", "metadata", group(report.metadata));
                for (name, b) in &report.tensors {
                    println!("  {:<18} {:>7} bytes", name, group(*b));
                }
                println!(
                    "  task         {:?}, target `{}`",
                    program.schema.task, program.schema.target.name
                );
                println!("  features     {}", program.features());
                println!(
                    "  dependence   {} × {} regime(s)",
                    program.dependence.name(),
                    program.regimes.k()
                );
                println!(
                    "  net          {} inputs → {} hidden → {} outputs",
                    program.net.inputs, program.net.hidden, program.net.outputs
                );
                println!(
                    "  trees        {} oblivious trees of depth {} ({}-bit leaves)",
                    program.forest.trees(),
                    program.forest.depth,
                    program.forest.bits
                );
                if !program.schema.dropped.is_empty() {
                    println!(
                        "  dropped      {} identifier-like column(s)",
                        program.schema.dropped.len()
                    );
                }
                println!("  rows stored  none");
            }
        }
        Command::Encode(a) => {
            let table = RawTable::read(&a.csv).map_err(e)?;
            let scaling = Scaling::parse(&a.scaling).ok_or("--scaling must be unit or quantile")?;
            let program = match &a.model {
                Some(m) => decode(&std::fs::read(m).map_err(|x| x.to_string())?).map_err(e)?,
                None => {
                    let t = a
                        .target
                        .as_deref()
                        .ok_or("--target is required without --model")?;
                    let target = table.column_index(t).map_err(e)?;
                    dope_core::eval::schema_program(
                        &infer(&table, target, &IntakeOptions::default()).map_err(e)?,
                        &table,
                    )
                    .map_err(e)?
                }
            };
            let data = program.schema.to_model(&table).map_err(e)?;
            let s = dope_core::Samples {
                x: data.x,
                y: data.y,
            };
            output::write_ml(&MlEncoder::new(&program, scaling), &s, writer(&a.out)?).map_err(e)?;
        }
        Command::Retention(a) => {
            let opts = EvalOptions {
                auditors: a
                    .auditors
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
                factor: a.factor,
                seed: a.seed,
            };
            let model = std::fs::read(&a.model).map_err(|x| x.to_string())?;
            let train = RawTable::read(&a.train).map_err(e)?;
            let test = RawTable::read(&a.test).map_err(e)?;
            let synth = a
                .synth
                .as_ref()
                .map(|p| RawTable::read(p))
                .transpose()
                .map_err(e)?;
            let result = evaluate(&model, &train, &test, synth.as_ref(), &opts).map_err(e)?;
            println!("{}", result.render());
            if let Some(p) = &a.json {
                std::fs::write(p, serde_json::to_string_pretty(&result).unwrap_or_default())
                    .map_err(|x| x.to_string())?;
            }
        }
        Command::Frontier(a) => {
            let train = RawTable::read(&a.train).map_err(e)?;
            let test = RawTable::read(&a.test).map_err(e)?;
            let target = train.column_index(&a.target).map_err(e)?;
            let budgets = a
                .budgets
                .split(',')
                .map(parse_bytes)
                .collect::<Result<Vec<_>, _>>()?;
            let opts = FrontierOptions {
                budgets,
                tau: a.tau,
                eval: EvalOptions {
                    auditors: a
                        .auditors
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect(),
                    factor: 4.0,
                    seed: a.seed,
                },
            };
            let result = frontier(&train, target, &test, &opts).map_err(e)?;
            println!("{}", result.render());
            if let Some(p) = &a.json {
                std::fs::write(p, serde_json::to_string_pretty(&result).unwrap_or_default())
                    .map_err(|x| x.to_string())?;
            }
        }
        Command::Throughput(a) => {
            let program =
                decode(&std::fs::read(&a.model).map_err(|x| x.to_string())?).map_err(e)?;
            let dev = device::parse(&a.device).map_err(e)?;
            let mut best = f64::INFINITY;
            for r in 0..a.repeats.max(1) {
                let opts = SampleOptions {
                    rows: a.rows,
                    seed: r as u64,
                    chunk: a.chunk,
                    device: dev.clone(),
                };
                let t = std::time::Instant::now();
                let s = sample(&program, &opts).map_err(e)?;
                best = best.min(t.elapsed().as_secs_f64());
                assert_eq!(s.y.len(), a.rows);
            }
            println!(
                "{{\"rows\": {}, \"seconds\": {:.4}, \"rows_per_second\": {:.0}, \"device\": \"{}\"}}",
                a.rows,
                best,
                a.rows as f64 / best,
                device::describe(&dev)
            );
        }
        Command::Doctor => {
            println!("dope {}", env!("CARGO_PKG_VERSION"));
            println!("  backends   {}", device::backends().join(", "));
            println!(
                "  default    {}",
                device::describe(&device::parse("auto").map_err(e)?)
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}
