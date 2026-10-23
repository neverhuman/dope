//! Markdown rendering of the landmark replay: the landmark plan's section 2.2 tables A–D and the
//! section 5.5a representation check.

use super::{
    ext, predictors, profile, selection, table_picks, LandmarkRow, Program, RawlinRow, LANDMARKS,
    PICK_COLUMNS, PREDICTORS, PROGS,
};

fn f3(x: Option<f64>) -> String {
    x.map_or("—".to_string(), |v| format!("{v:.3}"))
}

/// Tables A, B and C of the landmark plan's §2.2, plus the per-program detail.
pub fn render(rows: &[LandmarkRow]) -> String {
    let mut s = String::from(
        "A. Which program does each audit pick? (external T2 retention of the pick)\n\n",
    );
    s += "| Table | external T2 retention, best program |";
    for (_, label) in PICK_COLUMNS {
        s += &format!(" pick by {label} |");
    }
    s += "\n|---|---|---|---|---|---|\n";
    for r in rows {
        let (best, e, picks) = table_picks(r);
        let lo = e.values().copied().fold(f64::INFINITY, f64::min);
        let hi = e.values().copied().fold(f64::NEG_INFINITY, f64::max);
        s += &format!(
            "| `{}` | {best} {:.3} (range {lo:.3}–{hi:.3}) |",
            r.table, e[best]
        );
        for p in picks {
            s += &format!(" {p} {:.3} |", e[p]);
        }
        s.push('\n');
    }
    let sel = selection(rows);
    s += "\nB. Selection quality\n\n";
    s += "| Predictor | mean selection regret | best-program hits | mean within-table Spearman | pooled Spearman | mean bias vs T2 |\n";
    s += "|---|---:|---:|---:|---:|---:|\n";
    for name in PREDICTORS {
        let d = &sel.predictors[name];
        s += &format!(
            "| {name} | {:.4} | {}/{} | {:.2} | {} | {} |\n",
            d.regret,
            d.hits,
            d.tables,
            d.within_spearman,
            d.pooled_spearman.map_or("—".into(), |v| format!("{v:.2}")),
            d.bias.map_or("—".into(), |v| format!("{v:+.3}"))
        );
    }
    s += &format!("| always low | {:.4} | | | | |\n", sel.always_low);
    s += &format!("| always ultra | {:.4} | | | | |\n", sel.always_ultra);
    s += &format!(
        "\nT2 seed-to-seed std (median over programs): {:.4}\n",
        sel.t2_seed_std
    );
    s += "\nC. Real CV landmark profiles (normalized gain 1 - L/L_null; Brier for classification)\n\n";
    let keys = [
        "linear", "step", "pairs", "rff", "rows", "fourier", "gbdt", "knn1", "knn15", "mlp",
    ];
    s += &format!(
        "| Table | {} | linearity | interaction | repeat-CV max diff |\n|---|{}\n",
        keys.join(" | "),
        "---:|".repeat(keys.len() + 3)
    );
    for r in rows {
        let pr = profile(r);
        let cells: Vec<String> = keys
            .iter()
            .map(|m| format!("{:.2}", r.real_profile[*m]))
            .collect();
        s += &format!(
            "| {} | {} | {:.2} | {:+.2} | {:.3} |\n",
            r.table,
            cells.join(" | "),
            pr.linearity,
            pr.interaction,
            pr.repeat_max_diff
        );
    }
    s += "\nPer program\n\n| Table | program | external T2 (median of 3 seeds) | self-audit | 2-learner T1 | family-min T1 | strong-min T1 | signature gap |\n";
    s += "|---|---|---:|---:|---:|---:|---:|---:|\n";
    for r in rows {
        for p in PROGS {
            let pr = predictors(r.program(p), &r.real_profile);
            s += &format!(
                "| {} | {p} | {:.3} | {} | {} | {} | {} | {} |\n",
                r.table,
                ext(r.program(p)),
                f3(pr["self_audit"]),
                f3(pr["panel2_min"]),
                f3(pr["family_min"]),
                f3(pr["strong_min"]),
                f3(pr["signature_gap"])
            );
        }
    }
    s
}

/// Per-landmark T1 retention and per-family T2 retention (sampling seed 0) for two programs
/// of one table: the §2.2 D diagnostics.
pub fn render_diff(rows: &[LandmarkRow], table: &str, from: &str, to: &str) -> Option<String> {
    let r = rows.iter().find(|r| r.table == table)?;
    let (a, b) = (r.programs.get(from)?, r.programs.get(to)?);
    let mut s = format!("| {table} | {from} | {to} |\n|---|---:|---:|\n");
    for fam in ["linear", "gbdt", "mlp"] {
        let get = |p: &Program| p.t2_family.first().and_then(|m| m.get(fam).copied());
        s += &format!(
            "| T2 auditor {fam} (seed 0) | {} | {} |\n",
            f3(get(a)),
            f3(get(b))
        );
    }
    for m in LANDMARKS {
        s += &format!(
            "| T1 landmark {m} | {} | {} |\n",
            f3(a.tstr[m].ret),
            f3(b.tstr[m].ret)
        );
    }
    Some(s)
}

/// The §5.5a representation check: robust-scaled linear landmark (T1), the unclipped
/// mean/sd-scaled linear landmark (T1) and the engine's linear auditor (T2, seed 0).
pub fn render_rawlin(rows: &[LandmarkRow], rawlin: &[RawlinRow]) -> String {
    let mut s = String::from(
        "| Table | program | robust-scaled linear landmark, T1 | mean/sd-scaled linear, T1 | engine linear auditor, T2 seed 0 |\n|---|---|---:|---:|---:|\n",
    );
    for r in rows {
        let raw = rawlin.iter().find(|x| x.table == r.table);
        for p in PROGS {
            let prog = r.program(p);
            s += &format!(
                "| {} | {p} | {} | {} | {} |\n",
                r.table,
                f3(prog.tstr["linear"].ret),
                f3(raw.and_then(|x| x.programs.get(p).copied().flatten())),
                f3(prog
                    .t2_family
                    .first()
                    .and_then(|m| m.get("linear").copied()))
            );
        }
    }
    s
}
