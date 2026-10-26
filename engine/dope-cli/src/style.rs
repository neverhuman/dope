//! Zero-dependency ANSI terminal styling for interactive CLI output.
//!
//! Honors `--color auto|always|never`, `NO_COLOR` (https://no-color.org/), and
//! `CLICOLOR_FORCE`. In `auto` mode, styling is enabled only when `stdout` is an
//! interactive terminal (`std::io::IsTerminal`), preserving byte-for-byte plain
//! text for pipes, scripts, `--json` modes, and CI test harnesses.

use std::io::IsTerminal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColorChoice {
    Auto,
    Always,
    Never,
}

impl ColorChoice {
    pub(crate) fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "always" | "on" | "true" => Ok(Self::Always),
            "never" | "off" | "false" | "none" => Ok(Self::Never),
            _ => Err("--color must be auto, always, or never".into()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Style {
    pub(crate) enabled: bool,
}

impl Style {
    pub(crate) fn from_choice(choice: ColorChoice) -> Self {
        let enabled = match choice {
            ColorChoice::Always => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => {
                if std::env::var_os("NO_COLOR").is_some() {
                    false
                } else if std::env::var_os("CLICOLOR_FORCE")
                    .map(|v| v != "0" && !v.is_empty())
                    .unwrap_or(false)
                {
                    true
                } else {
                    std::io::stdout().is_terminal()
                }
            }
        };
        Self { enabled }
    }

    pub(crate) fn wrap(&self, code: &str, text: impl AsRef<str>) -> String {
        let t = text.as_ref();
        if self.enabled {
            format!("\x1b[{code}m{t}\x1b[0m")
        } else {
            t.to_string()
        }
    }

    pub(crate) fn bold(&self, text: impl AsRef<str>) -> String {
        self.wrap("1", text)
    }

    pub(crate) fn dim(&self, text: impl AsRef<str>) -> String {
        self.wrap("2", text)
    }

    pub(crate) fn cyan_bold(&self, text: impl AsRef<str>) -> String {
        self.wrap("1;36", text)
    }

    pub(crate) fn green_bold(&self, text: impl AsRef<str>) -> String {
        self.wrap("1;32", text)
    }

    pub(crate) fn yellow_bold(&self, text: impl AsRef<str>) -> String {
        self.wrap("1;33", text)
    }

    pub(crate) fn red_bold(&self, text: impl AsRef<str>) -> String {
        self.wrap("1;31", text)
    }

    pub(crate) fn blue(&self, text: impl AsRef<str>) -> String {
        self.wrap("34", text)
    }

    pub(crate) fn green(&self, text: impl AsRef<str>) -> String {
        self.wrap("32", text)
    }

    pub(crate) fn retention_badge(&self, r: f64) -> String {
        let raw = format!("{r:.3}");
        if !self.enabled {
            return raw;
        }
        if r >= 0.90 {
            self.green_bold(format!("{raw} (≥T90)"))
        } else if r >= 0.80 {
            self.cyan_bold(format!("{raw} (≥D80)"))
        } else if r >= 0.50 {
            self.yellow_bold(raw)
        } else {
            self.red_bold(raw)
        }
    }

    pub(crate) fn budget_bar(&self, used: usize, budget: usize, width: usize) -> String {
        let cap = budget.max(1);
        let ratio = (used as f64 / cap as f64).clamp(0.0, 1.0);
        let filled = ((ratio * width as f64).round() as usize).min(width);
        let empty = width.saturating_sub(filled);
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
        let pct = format!("{:.1}%", ratio * 100.0);
        if self.enabled {
            format!("[{}] {}", self.cyan_bold(bar), self.dim(pct))
        } else {
            format!("[{bar}] {pct}")
        }
    }
}

pub(crate) fn group(n: usize) -> String {
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

pub(crate) fn print_fit_report(
    csv: &std::path::Path,
    out: &std::path::Path,
    r: &dope_core::FitReport,
    dev_desc: &str,
    style: Style,
) {
    let drv = r
        .screened
        .iter()
        .take(6)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if !style.enabled {
        println!(
            "compiled {} -> {}  ({} bytes, budget {})",
            csv.display(),
            out.display(),
            group(r.bytes),
            group(r.budget)
        );
        println!(
            "  task         {} · {} rows × {} features",
            r.task,
            group(r.rows),
            r.features
        );
        println!("  program      {} knots · {} dependence × {} regime(s) · net {}→{} (bits w{} b{} β{} v{}) · {} trees of depth {}", r.knots, r.dependence, r.regimes, r.inputs, r.hidden, r.bits[0], r.bits[1], r.bits[2], r.bits[3], r.trees, r.depth);
        println!(
            "  parameters   {} total scalar slots ({} network, {} forest)",
            group(r.parameters.total),
            group(r.parameters.network),
            group(r.parameters.forest)
        );
        println!(
            "  stored gain  {:.3} of the null loss explained on held-out rows",
            r.stored_gain
        );
        if let Some(a) = r.self_audit {
            println!("  self-audit   robust retention {a:.3} (linear and tree auditors trained on samples, scored on held-out training rows)");
        }
        println!("  drivers      {drv}");
        println!("  time         {:.1} s on {dev_desc}", r.seconds);
    } else {
        println!(
            "{} compiled {} -> {}  ({} / {} bytes) {}",
            style.green_bold("✓"),
            style.bold(csv.display().to_string()),
            style.cyan_bold(out.display().to_string()),
            style.bold(group(r.bytes)),
            group(r.budget),
            style.budget_bar(r.bytes, r.budget, 12)
        );
        println!(
            "  {}         {} · {} rows × {} features · {}",
            style.dim("task"),
            style.bold(&r.task),
            group(r.rows),
            r.features,
            style.green("0 source rows stored")
        );
        println!("  {}      {} knots · {} dependence × {} regime(s) · net {}→{} (bits w{} b{} β{} v{}) · {} trees of depth {}", style.dim("program"), r.knots, r.dependence, r.regimes, r.inputs, r.hidden, r.bits[0], r.bits[1], r.bits[2], r.bits[3], r.trees, r.depth);
        println!(
            "  {}   {} total scalar slots ({} network, {} forest)",
            style.dim("parameters"),
            style.bold(group(r.parameters.total)),
            group(r.parameters.network),
            group(r.parameters.forest)
        );
        println!(
            "  {}  {} of the null loss explained on held-out rows",
            style.dim("stored gain"),
            style.bold(format!("{:.3}", r.stored_gain))
        );
        if let Some(a) = r.self_audit {
            println!("  {}   robust retention {} (linear and tree auditors trained on samples, scored on held-out training rows)", style.dim("self-audit"), style.retention_badge(a));
        }
        println!("  {}      {}", style.dim("drivers"), style.cyan_bold(drv));
        println!(
            "  {}         {:.1} s on {dev_desc}",
            style.dim("time"),
            r.seconds
        );
    }
}

pub(crate) fn print_inspect_report(
    model_path: &std::path::Path,
    program: &dope_core::Program,
    report: &dope_core::ByteReport,
    style: Style,
) {
    if !style.enabled {
        println!(
            "{}  {} bytes ({} brotli program + {} safetensors framing)",
            model_path.display(),
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
            "  parameters   {} total scalar slots",
            group(program.parameters().total)
        );
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
    } else {
        println!(
            "{} {}  {} bytes ({} brotli program + {} safetensors framing)",
            style.cyan_bold("●"),
            style.bold(model_path.display().to_string()),
            style.green_bold(group(report.total)),
            group(report.stored),
            group(report.header)
        );
        println!("  {}", style.dim("inflated, the program holds:"));
        println!(
            "  {:<18} {:>7} bytes",
            style.dim("metadata"),
            group(report.metadata)
        );
        for (name, b) in &report.tensors {
            println!("  {:<18} {:>7} bytes", style.blue(name), group(*b));
        }
        println!(
            "  {}         {:?}, target `{}`",
            style.dim("task"),
            program.schema.task,
            style.cyan_bold(&program.schema.target.name)
        );
        println!("  {}     {}", style.dim("features"), program.features());
        println!(
            "  {}   {} total scalar slots",
            style.dim("parameters"),
            style.bold(group(program.parameters().total))
        );
        println!(
            "  {}   {} × {} regime(s)",
            style.dim("dependence"),
            program.dependence.name(),
            program.regimes.k()
        );
        println!(
            "  {}          {} inputs → {} hidden → {} outputs",
            style.dim("net"),
            program.net.inputs,
            program.net.hidden,
            program.net.outputs
        );
        println!(
            "  {}        {} oblivious trees of depth {} ({}-bit leaves)",
            style.dim("trees"),
            program.forest.trees(),
            program.forest.depth,
            program.forest.bits
        );
        if !program.schema.dropped.is_empty() {
            println!(
                "  {}      {} identifier-like column(s)",
                style.dim("dropped"),
                program.schema.dropped.len()
            );
        }
        println!(
            "  {}  {}",
            style.dim("rows stored"),
            style.green_bold("none")
        );
    }
}

pub(crate) fn print_eval_report(result: &dope_core::eval::EvalResult, style: Style) {
    if !style.enabled {
        println!("{}", result.render());
    } else {
        println!("{} retention on {} held-out real rows ({} task, artifact {} bytes, {} synthetic vs {} real training rows)", style.cyan_bold("●"), style.bold(result.test_rows.to_string()), style.bold(&result.task), style.green_bold(group(result.artifact_bytes)), group(result.synth_rows), group(result.train_rows));
        println!(
            "  {}",
            style.dim("auditor   null loss   real loss  synth loss  retention")
        );
        for aud in &result.auditors {
            let ret_str = if aud.informative {
                style.retention_badge(aud.retention)
            } else {
                style.dim("n/a")
            };
            println!(
                "  {:<8} {:>10.5} {:>11.5} {:>11.5}  {}",
                style.bold(&aud.auditor),
                aud.null_loss,
                aud.real_loss,
                aud.synth_loss,
                ret_str
            );
        }
        if let Some(r) = result.robust_retention {
            println!(
                "  {} {}",
                style.bold("robust retention (min over families):"),
                style.retention_badge(r)
            );
        }
    }
}
