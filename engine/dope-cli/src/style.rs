//! Zero-dependency ANSI terminal styling for interactive CLI output.
//!
//! Honors `--color auto|always|never`, `NO_COLOR` (https://no-color.org/), and
//! `CLICOLOR_FORCE`. In `auto` mode, styling is enabled only when `stdout` is an
//! interactive terminal (`std::io::IsTerminal`), preserving byte-for-byte plain
//! text for pipes, scripts, `--json` modes, and CI test harnesses.

use std::io::IsTerminal;
use std::path::Path;

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
                    .is_some_and(|v| v != "0" && !v.is_empty())
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
    csv: &Path,
    out: &Path,
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
    let (b_str, bud_str, rows_str) = (group(r.bytes), group(r.budget), group(r.rows));
    let (pt, pn, pf) = (
        group(r.parameters.total),
        group(r.parameters.network),
        group(r.parameters.forest),
    );
    let [w0, w1, w2, w3] = r.bits;
    let cd = style.bold(csv.display().to_string());
    let od = style.cyan_bold(out.display().to_string());
    if !style.enabled {
        println!("compiled {cd} -> {od}  ({b_str} bytes, budget {bud_str})");
        println!(
            "  task         {} · {rows_str} rows × {} features",
            r.task, r.features
        );
    } else {
        let ok = style.green_bold("✓");
        let bb = style.bold(&b_str);
        let bar = style.budget_bar(r.bytes, r.budget, 12);
        println!("{ok} compiled {cd} -> {od}  ({bb} / {bud_str} bytes) {bar}");
        let tk = style.bold(&r.task);
        let zr = style.green("0 source rows stored");
        println!(
            "  {}         {tk} · {rows_str} rows × {} features · {zr}",
            style.dim("task"),
            r.features
        );
    }
    println!(
        "  {}      {} knots · {} dependence × {} regime(s) · net {}→{} (bits w{w0} b{w1} β{w2} v{w3}) · {} trees of depth {}",
        style.dim("program"), r.knots, r.dependence, r.regimes, r.inputs, r.hidden, r.trees, r.depth
    );
    println!(
        "  {}   {} total scalar slots ({pn} network, {pf} forest)",
        style.dim("parameters"),
        style.bold(&pt)
    );
    println!(
        "  {}  {} of the null loss explained on held-out rows",
        style.dim("stored gain"),
        style.bold(format!("{:.3}", r.stored_gain))
    );
    if let Some(a) = r.self_audit {
        println!(
            "  {}   robust retention {} (linear and tree auditors trained on samples, scored on held-out training rows)",
            style.dim("self-audit"),
            style.retention_badge(a)
        );
    }
    println!("  {}      {}", style.dim("drivers"), style.cyan_bold(drv));
    println!(
        "  {}         {:.1} s on {dev_desc}",
        style.dim("time"),
        r.seconds
    );
}

pub(crate) fn print_inspect_report(
    model_path: &Path,
    program: &dope_core::Program,
    report: &dope_core::ByteReport,
    style: Style,
) {
    let pfx = if style.enabled {
        format!("{} ", style.cyan_bold("●"))
    } else {
        String::new()
    };
    let mp = style.bold(model_path.display().to_string());
    let tb_col = style.green_bold(group(report.total));
    let (sto, hdr, meta) = (
        group(report.stored),
        group(report.header),
        group(report.metadata),
    );
    println!("{pfx}{mp}  {tb_col} bytes ({sto} brotli program + {hdr} safetensors framing)");
    println!("  {}", style.dim("inflated, the program holds:"));
    println!("  {:<18} {:>7} bytes", style.dim("metadata"), meta);
    for (name, b) in &report.tensors {
        println!("  {:<18} {:>7} bytes", style.blue(name), group(*b));
    }
    let tgt = style.cyan_bold(&program.schema.target.name);
    println!(
        "  {}         {:?}, target `{tgt}`",
        style.dim("task"),
        program.schema.task
    );
    println!("  {}     {}", style.dim("features"), program.features());
    let pt = style.bold(group(program.parameters().total));
    println!("  {}   {pt} total scalar slots", style.dim("parameters"));
    let (dep, rk) = (program.dependence.name(), program.regimes.k());
    println!("  {}   {dep} × {rk} regime(s)", style.dim("dependence"));
    let (ni, nh, no) = (program.net.inputs, program.net.hidden, program.net.outputs);
    println!(
        "  {}          {ni} inputs → {nh} hidden → {no} outputs",
        style.dim("net")
    );
    let (nt, td, tb) = (
        program.forest.trees(),
        program.forest.depth,
        program.forest.bits,
    );
    println!(
        "  {}        {nt} oblivious trees of depth {td} ({tb}-bit leaves)",
        style.dim("trees")
    );
    if !program.schema.dropped.is_empty() {
        let nd = program.schema.dropped.len();
        println!(
            "  {}      {nd} identifier-like column(s)",
            style.dim("dropped")
        );
    }
    println!(
        "  {}  {}",
        style.dim("rows stored"),
        style.green_bold("none")
    );
}

pub(crate) fn print_eval_report(result: &dope_core::eval::EvalResult, style: Style) {
    if !style.enabled {
        println!("{}", result.render());
    } else {
        println!(
            "{} retention on {} held-out real rows ({} task, artifact {} bytes, {} synthetic vs {} real training rows)",
            style.cyan_bold("●"),
            style.bold(result.test_rows.to_string()),
            style.bold(&result.task),
            style.green_bold(group(result.artifact_bytes)),
            group(result.synth_rows),
            group(result.train_rows)
        );
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
            let nm = style.bold(&aud.auditor);
            let (nl, rl, sl) = (aud.null_loss, aud.real_loss, aud.synth_loss);
            println!("  {nm:<8} {nl:>10.5} {rl:>11.5} {sl:>11.5}  {ret_str}");
        }
        if let Some(r) = result.robust_retention {
            let lbl = style.bold("robust retention (min over families):");
            println!("  {lbl} {}", style.retention_badge(r));
        }
    }
}

pub(crate) fn print_sample_saved_report(
    model_path: &Path,
    out_path: &Path,
    rows: usize,
    features: usize,
    target_name: &str,
    space: &str,
    style: Style,
) {
    if style.enabled {
        let (ok, nr) = (style.green_bold("✓"), style.bold(group(rows)));
        let (tg, sp) = (style.cyan_bold(target_name), style.bold(space));
        let mp = style.bold(model_path.display().to_string());
        let op = style.cyan_bold(out_path.display().to_string());
        println!("{ok} sampled {nr} synthetic rows ({features} features, target `{tg}`, {sp} space) from {mp} -> {op}");
    }
}

pub(crate) fn print_frontier_report(result: &dope_core::eval::FrontierResult, style: Style) {
    if !style.enabled {
        println!("{}", result.render());
    } else {
        let dot = style.cyan_bold("●");
        println!(
            "{dot} retention-vs-bytes frontier (threshold τ = {:.0}%)",
            result.tau * 100.0
        );
        println!(
            "  {}",
            style.dim("budget     bytes   budget bar         robust retention   per auditor")
        );
        for p in &result.points {
            let per: Vec<String> = p
                .retention
                .iter()
                .map(|(a, r)| format!("{} {r:.3}", style.bold(a)))
                .collect();
            let rob = p
                .robust_retention
                .map_or_else(|| style.dim("n/a"), |r| style.retention_badge(r));
            let (bud, b_col) = (group(p.budget), style.bold(group(p.bytes)));
            let (bar, pj) = (style.budget_bar(p.bytes, p.budget, 10), per.join(" · "));
            println!("  {bud:>7} {b_col:>8}   {bar}   {rob:>16}   {pj}");
        }
        match result.tdl_bytes {
            Some(b) => {
                let (ok, gb) = (style.green_bold("✓"), style.green_bold(group(b)));
                println!("  {ok} TDL@{:.0}% = {gb} bytes (smallest tested budget reaching the threshold)", result.tau * 100.0);
            }
            None => {
                let wn = style.yellow_bold("!");
                println!(
                    "  {wn} TDL@{:.0}% not reached in the tested range",
                    result.tau * 100.0
                );
            }
        }
    }
}

pub(crate) fn print_mlc_v3_weights_report(
    weights_path: &Path,
    m: &dope_core::mlc_v3::MlcV3WeightInventory,
    style: Style,
) {
    let p = style.bold(weights_path.display().to_string());
    let b = style.green_bold(group(m.byte_len));
    let (tc, pc) = (
        style.bold(group(m.tensor_count)),
        style.bold(group(m.parameter_count)),
    );
    println!(
        "{} {p}  {b} bytes ({tc} tensors · {pc} parameters)",
        style.cyan_bold("●")
    );
    println!("  {}       dope-mlc-v3-safetensors", style.dim("format"));
    let rt = style.green_bold("rust-candle-cpu · 0 Python/LibTorch deps");
    println!("  {}      {rt}", style.dim("runtime"));
    println!("  {}       {}", style.dim("sha256"), m.sha256);
}

pub(crate) fn print_mlc_v3_wire_report(
    wire_path: &Path,
    env: &dope_core::mlc_v3::Dml3Envelope,
    style: Style,
) {
    let enc = &env.encoding;
    let p = style.bold(wire_path.display().to_string());
    let (wb, cb) = (
        style.green_bold(group(env.wire_bytes)),
        group(env.tier_cap_bytes),
    );
    let (wf, lv) = (style.bold(&env.wire_format), style.cyan_bold(&enc.level));
    let bar = style.budget_bar(env.wire_bytes, env.tier_cap_bytes, 12);
    println!(
        "{} {p}  {wb} / {cb} bytes ({wf} · tier `{lv}`) {bar}",
        style.cyan_bold("●")
    );
    let (tk, kc) = (style.bold(&enc.task), enc.target_bundle.k_classes);
    let (nt, pu, zr) = (
        group(enc.n_train),
        enc.p_use,
        style.green("0 source rows stored"),
    );
    println!(
        "  {}         {tk} (k_classes = {kc}) · {nt} train rows × {pu} features · {zr}",
        style.dim("task")
    );
    let (cw, v3p, csc) = (
        enc.code_width,
        enc.target_bundle.include_v3_primitives,
        env.scaler.is_some(),
    );
    println!(
        "  {}     code_width = {cw} · v3_primitives = {v3p} · csc1_scaler = {csc}",
        style.dim("encoding")
    );
    let crc = env
        .crc32
        .map_or_else(|| "none".to_string(), |c| format!("0x{c:08x}"));
    let cap = if env.within_tier_cap {
        style.green_bold("true")
    } else {
        style.red_bold("false")
    };
    println!(
        "  {}    within_tier_cap = {cap} · crc32 = {crc} · flags = 0x{:02x}",
        style.dim("integrity"),
        env.compressed_flags
    );
}

pub(crate) fn print_mlc_v3_sample_report(
    wire_path: &Path,
    out_path: &Path,
    rows: usize,
    features: usize,
    space: &str,
    env: &dope_core::mlc_v3::Dml3Envelope,
    style: Style,
) {
    if style.enabled {
        let (ok, nr, sp) = (
            style.green_bold("✓"),
            style.bold(group(rows)),
            style.bold(space),
        );
        let wp = style.bold(wire_path.display().to_string());
        let (wf, lv, wb) = (
            &env.wire_format,
            style.cyan_bold(&env.encoding.level),
            group(env.wire_bytes),
        );
        let op = style.cyan_bold(out_path.display().to_string());
        println!("{ok} sampled {nr} synthetic rows ({features} features + target, {sp} space) from {wp} ({wf} · {lv} · {wb} B) -> {op}");
    }
}
