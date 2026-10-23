//! Landmark probe receipts and their replay.
//!
//! The receipts were produced by a Python prototype of a 16-landmark panel (scikit-learn
//! trees, kNN and a small MLP; NumPy ridge bases) outside this repository. That learner set
//! cannot be reproduced bit-for-bit in Rust; the native panel (`dope landmarks`) replaces it.
//! This module recomputes every summary the landmark plan cites from the receipts alone.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::stats::{mean, median, spearman, std};
use crate::Result;

mod render;

pub use render::{render, render_diff, render_rawlin};

/// Landmark order of the prototype (and of every map in the receipts).
pub const LANDMARKS: [&str; 16] = [
    "linear", "log", "square", "tanh", "abs", "fourier", "rows", "step", "pairs", "rff", "stump",
    "tree3", "gbdt", "knn1", "knn15", "mlp",
];
pub const FAMILIES: [(&str, &[&str]); 4] = [
    (
        "additive",
        &[
            "linear", "log", "square", "tanh", "abs", "fourier", "rows", "step", "pairs",
        ],
    ),
    ("kernel_local", &["rff", "knn15", "knn1"]),
    ("tree", &["stump", "tree3", "gbdt"]),
    ("neural", &["mlp"]),
];
pub const PROGS: [&str; 5] = ["low", "med", "high", "ultra", "sum"];
pub const PREDICTORS: [&str; 8] = [
    "self_audit",
    "panel2_min",
    "family_min",
    "family_min_x_knn1",
    "strong_min",
    "gain_weighted",
    "pooled_strong",
    "signature_gap",
];

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Tstr {
    pub real: f64,
    pub syn: f64,
    pub null: f64,
    pub ret: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Program {
    pub bytes: u64,
    pub self_audit: Option<f64>,
    pub tstr: BTreeMap<String, Tstr>,
    pub syn_profile: BTreeMap<String, f64>,
    pub t2_robust: Vec<Option<f64>>,
    pub t2_family: Vec<BTreeMap<String, f64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LandmarkRow {
    pub table: String,
    pub task: String,
    pub n_train: u64,
    pub d: u64,
    pub real_profile: BTreeMap<String, f64>,
    pub real_profile_repeat: BTreeMap<String, f64>,
    pub programs: BTreeMap<String, Program>,
}

impl LandmarkRow {
    pub fn program(&self, p: &str) -> &Program {
        &self.programs[p]
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawlinRow {
    pub table: String,
    pub programs: BTreeMap<String, Option<f64>>,
}

pub fn load(path: &Path) -> Result<Vec<LandmarkRow>> {
    crate::load_jsonl(path)
}

pub fn load_rawlin(path: &Path) -> Result<Vec<RawlinRow>> {
    crate::load_jsonl(path)
}

fn clip(x: f64) -> f64 {
    x.clamp(-1.0, 1.5)
}

/// Every predictor's score for one program; `None` where the predictor is undefined.
pub fn predictors(
    p: &Program,
    real_profile: &BTreeMap<String, f64>,
) -> BTreeMap<&'static str, Option<f64>> {
    let t = |m: &str| p.tstr[m];
    let gain = |m: &str| t(m).null - t(m).real;
    let l0 = t(LANDMARKS[0]).null;
    let inf: Vec<&str> = LANDMARKS
        .into_iter()
        .filter(|m| t(m).ret.is_some())
        .collect();
    let ret = |m: &str| t(m).ret.unwrap_or(0.0);
    let gmax = LANDMARKS
        .iter()
        .map(|m| gain(m))
        .fold(f64::NEG_INFINITY, f64::max);
    let strong: Vec<&str> = inf
        .iter()
        .copied()
        .filter(|m| gain(m) >= 0.5 * gmax)
        .collect();
    let mut out = BTreeMap::new();
    out.insert("self_audit", p.self_audit);
    let two: Vec<f64> = ["linear", "gbdt"].iter().filter_map(|m| t(m).ret).collect();
    out.insert("panel2_min", two.into_iter().reduce(f64::min));
    out.insert(
        "strong_min",
        strong.iter().map(|m| clip(ret(m))).reduce(f64::min),
    );
    let (mut num, mut den) = (0.0, 0.0);
    for m in &inf {
        num += gain(m) * clip(ret(m));
    }
    for m in &inf {
        den += gain(m);
    }
    out.insert("gain_weighted", (!inf.is_empty()).then(|| num / den));
    let (mut a, mut b) = (0.0, 0.0);
    for m in &strong {
        a += l0 - t(m).syn;
    }
    for m in &strong {
        b += l0 - t(m).real;
    }
    out.insert("pooled_strong", (!strong.is_empty()).then(|| a / b));
    let family = |skip_knn1: bool| {
        FAMILIES
            .iter()
            .filter_map(|(_, ms)| {
                let ms: Vec<&str> = ms
                    .iter()
                    .copied()
                    .filter(|m| inf.contains(m) && !(skip_knn1 && *m == "knn1"))
                    .collect();
                if ms.is_empty() {
                    return None;
                }
                let (mut n, mut d) = (0.0, 0.0);
                for m in &ms {
                    n += gain(m) * clip(ret(m));
                }
                for m in &ms {
                    d += gain(m);
                }
                Some(n / d)
            })
            .reduce(f64::min)
    };
    out.insert("family_min", family(false));
    out.insert("family_min_x_knn1", family(true));
    let gaps: Vec<f64> = LANDMARKS
        .iter()
        .filter(|m| real_profile[**m] > 0.01)
        .map(|m| (p.syn_profile[*m] - real_profile[*m]).abs())
        .collect();
    out.insert("signature_gap", (!gaps.is_empty()).then(|| -mean(&gaps)));
    out
}

/// External T2 retention: median robust retention over the sampling seeds.
pub fn ext(p: &Program) -> f64 {
    let v: Vec<f64> = p.t2_robust.iter().flatten().copied().collect();
    median(&v)
}

/// Highest score; ties go to the smaller program (PROGS order).
pub fn pick(vals: &[(&'static str, f64)]) -> &'static str {
    let top = vals.iter().map(|v| v.1).fold(f64::NEG_INFINITY, f64::max);
    vals.iter()
        .find(|v| v.1 >= top - 1e-12)
        .map(|v| v.0)
        .unwrap_or(PROGS[0])
}

fn scores(r: &LandmarkRow, name: &str) -> Vec<(&'static str, f64)> {
    PROGS
        .iter()
        .filter_map(|p| predictors(r.program(p), &r.real_profile)[name].map(|v| (*p, v)))
        .collect()
}

#[derive(Debug, Clone)]
pub struct Selection {
    pub regret: f64,
    pub hits: usize,
    pub tables: usize,
    pub within_spearman: f64,
    pub pooled_spearman: Option<f64>,
    pub bias: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SelectionSummary {
    pub predictors: BTreeMap<&'static str, Selection>,
    pub always_low: f64,
    pub always_ultra: f64,
    pub t2_seed_std: f64,
}

pub fn selection(rows: &[LandmarkRow]) -> SelectionSummary {
    let mut out = BTreeMap::new();
    for name in PREDICTORS {
        let (mut regret, mut hits, mut within) = (Vec::new(), 0, Vec::new());
        let (mut pp, mut pe) = (Vec::new(), Vec::new());
        for r in rows {
            let e: BTreeMap<&str, f64> = PROGS.iter().map(|p| (*p, ext(r.program(p)))).collect();
            let best = e.values().copied().fold(f64::NEG_INFINITY, f64::max);
            let vals = scores(r, name);
            if vals.is_empty() {
                continue;
            }
            let chosen = pick(&vals);
            regret.push(best - e[chosen]);
            hits += usize::from(e[chosen] >= best - 1e-12);
            let a: Vec<f64> = vals.iter().map(|v| v.1).collect();
            let b: Vec<f64> = vals.iter().map(|v| e[v.0]).collect();
            if let Some(s) = spearman(&a, &b) {
                within.push(s);
            }
            pp.extend(a);
            pe.extend(b);
        }
        let diff: Vec<f64> = pp.iter().zip(&pe).map(|(a, b)| a - b).collect();
        out.insert(
            name,
            Selection {
                regret: mean(&regret),
                hits,
                tables: rows.len(),
                within_spearman: mean(&within),
                pooled_spearman: spearman(&pp, &pe),
                bias: (name != "signature_gap").then(|| mean(&diff)),
            },
        );
    }
    let (mut low, mut ultra, mut noise) = (Vec::new(), Vec::new(), Vec::new());
    for r in rows {
        let e: Vec<f64> = PROGS.iter().map(|p| ext(r.program(p))).collect();
        let best = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        low.push(best - e[0]);
        ultra.push(best - e[3]);
        for p in PROGS {
            let v: Vec<f64> = r.program(p).t2_robust.iter().flatten().copied().collect();
            noise.push(std(&v));
        }
    }
    SelectionSummary {
        predictors: out,
        always_low: mean(&low),
        always_ultra: mean(&ultra),
        t2_seed_std: median(&noise),
    }
}

/// Table A: the best program by T2 and each audit's pick, with the pick's T2 retention.
pub const PICK_COLUMNS: [(&str, &str); 4] = [
    ("self_audit", "engine self-audit"),
    ("panel2_min", "2 learners, clean T1"),
    ("family_min", "16-landmark family-min, T1"),
    ("signature_gap", "signature gap (no real holdout)"),
];

pub fn table_picks(
    r: &LandmarkRow,
) -> (&'static str, BTreeMap<&'static str, f64>, Vec<&'static str>) {
    let e: BTreeMap<&'static str, f64> = PROGS.iter().map(|p| (*p, ext(r.program(p)))).collect();
    let mut best = PROGS[0];
    for p in PROGS {
        if e[p] > e[best] {
            best = p;
        }
    }
    let picks = PICK_COLUMNS
        .iter()
        .map(|(n, _)| pick(&scores(r, n)))
        .collect();
    (best, e, picks)
}

#[derive(Debug, Clone)]
pub struct ProfileRow {
    pub table: String,
    pub linearity: f64,
    pub interaction: f64,
    pub repeat_max_diff: f64,
}

/// Table C summaries of the real CV landmark profile.
pub fn profile(r: &LandmarkRow) -> ProfileRow {
    let rp = &r.real_profile;
    let gbest = LANDMARKS
        .iter()
        .map(|m| rp[*m])
        .fold(f64::NEG_INFINITY, f64::max);
    let additive = FAMILIES[0]
        .1
        .iter()
        .map(|m| rp[*m])
        .fold(f64::NEG_INFINITY, f64::max);
    let ratio = |x: f64| if gbest > 0.0 { x / gbest } else { 0.0 };
    ProfileRow {
        table: r.table.clone(),
        linearity: ratio(rp["linear"]),
        interaction: ratio(rp["gbdt"] - additive),
        repeat_max_diff: LANDMARKS
            .iter()
            .map(|m| (rp[*m] - r.real_profile_repeat[*m]).abs())
            .fold(0.0, f64::max),
    }
}
