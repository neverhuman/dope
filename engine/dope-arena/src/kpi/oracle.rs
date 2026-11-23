//! E0 oracle arms (plan v3, package C1): where does the DSE7 step-0 packet lose the KPI?
//!
//! Every arm is scored on the cell's identical rows, like any control. These are oracles, not
//! encodings: every arm but `oracle_packet` reads A's rows, so all are logged as `control`.
//!
//! | arm | X rows | Y |
//! |---|---|---|
//! | `oracle_packet` | DSE7 step-0 decode of A's packet (≤ 2,048 B) | the packet's linear target decoder |
//! | `oracle_real_x_packet_y` | A | the packet's linear target decoder applied to A's X |
//! | `oracle_packet_x_tree_y` | packet decode | 40 depth-3 oblivious trees fitted on A |
//! | `oracle_real_x_tree_y` | A | the same 40 trees |
//! | `oracle_packet_x_gam_y` | packet decode | boosted stumps fitted on A (additive main effects) |
//! | `oracle_packet_x_teacher_y` | packet decode | a depth-5 booster on A, early-stopped on a 20 % inner split |
//! | `oracle_exact_marginal_rank12` | Gaussian copula over X and Y: exact A marginals, rank-12 factor dependence | (one of the copula columns) |
//! | `oracle_exact_marginal_fullrank` | the same copula with the full correlation matrix | (one of the copula columns) |
//!
//! Read side by side: `packet` vs `real_x_packet_y` prices the X side at the packet's own
//! conditional; `packet` vs `packet_x_tree_y` prices the conditional at the packet's X;
//! `real_x_tree_y` is the small-tree ceiling with nothing synthetic but the labels. The copula
//! arms say how far exact marginals plus Gaussian dependence go with no packet at all.
//!
//! Labels are always sampled: Bernoulli or categorical draws from the logits, regression with a
//! Gaussian residual (the packet's `resid_sd`, or the tree model's residual SD on A). Trees read
//! every column through A's mid-rank empirical CDF, so real and decoded X share one input map.
//! A model is fitted once per cell from a structural seed; sample seeds only change the draws.

mod copula;

use self::copula::copula_rows;
use super::celldata::{CellData, Rows};
use super::cellplan::Cell;
use super::mat::Mat;
use super::seeds::{crn_seed, KpiRng, STRUCTURAL_SAMPLE_SEED};
use dope_arena::ArenaResult;
use dope_core::forest::{boost, code, predict_columns, BoostOptions, Forest, Split};
use dope_core::single_encoding::{
    fit_dse7_packet, sample_zero_init_rows, unpack_dse7_packet, Dse7Header, ScaleCardV7,
    SingleEncV7Packet, MAX_CLASSES,
};
use dope_core::target::Head;

pub const PACKET: &str = "oracle_packet";
pub const REAL_X_PACKET_Y: &str = "oracle_real_x_packet_y";
pub const PACKET_X_TREE_Y: &str = "oracle_packet_x_tree_y";
pub const REAL_X_TREE_Y: &str = "oracle_real_x_tree_y";
pub const PACKET_X_GAM_Y: &str = "oracle_packet_x_gam_y";
pub const PACKET_X_TEACHER_Y: &str = "oracle_packet_x_teacher_y";
pub const EXACT_RANK12: &str = "oracle_exact_marginal_rank12";
pub const EXACT_FULL: &str = "oracle_exact_marginal_fullrank";
pub const ARMS: [&str; 8] = [
    PACKET,
    REAL_X_PACKET_Y,
    PACKET_X_TREE_Y,
    REAL_X_TREE_Y,
    PACKET_X_GAM_Y,
    PACKET_X_TEACHER_Y,
    EXACT_RANK12,
    EXACT_FULL,
];
/// Factor rank of `oracle_exact_marginal_rank12` (the DSE7 loading rank).
pub const COPULA_RANK: usize = 12;
/// The packet budget of every packet-side arm: the v2 headline.
pub const BUDGET: usize = 2048;
pub const REASON_PACKET: &str = "oracle_packet_fit_failed";

pub fn is_oracle(arm_id: &str) -> bool {
    ARMS.contains(&arm_id)
}

fn task_id(head: Head) -> u8 {
    match head {
        Head::Regression => 0,
        Head::Binary => 1,
        Head::Multiclass(_) => 2,
    }
}

fn columns(rows: &Rows) -> Vec<Vec<f64>> {
    (0..rows.p)
        .map(|j| {
            (0..rows.n)
                .map(|i| f64::from(rows.x[i * rows.p + j]))
                .collect()
        })
        .collect()
}

fn seed(cell: &Cell, sample_seed: u64, purpose: &str) -> u64 {
    crn_seed(
        &cell.panel_sha256,
        &cell.dataset_hash,
        cell.data_seed,
        sample_seed,
        purpose,
    )
}

// -- the packet -----------------------------------------------------------------------------
struct Packet {
    pkt: SingleEncV7Packet,
    hdr: Dse7Header,
    card: ScaleCardV7,
}

fn fit_packet(a: &Rows, head: Head) -> Option<Packet> {
    let y: Vec<f64> = a.y.iter().map(|&v| f64::from(v)).collect();
    let bytes = fit_dse7_packet(&columns(a), &y, task_id(head), BUDGET).ok()?;
    let (pkt, hdr, card) = unpack_dse7_packet(&bytes).ok()?;
    Some(Packet { pkt, hdr, card })
}

/// A sampled target from packet logits, as `dope_v7_emit` samples it.
fn packet_label(p: &Packet, logits: &[f32], y_reg: f32, rng: &mut KpiRng) -> f32 {
    match p.hdr.task_id {
        0 => (f64::from(p.card.y_mean) + f64::from(p.card.y_sd) * f64::from(y_reg)) as f32,
        1 => {
            let p1 = 1.0 / (1.0 + (-f64::from(logits[0].clamp(-10.0, 10.0))).exp());
            let idx = usize::from(rng.uniform() < p1);
            p.card.classes.get(idx).copied().unwrap_or(idx as i32) as f32
        }
        _ => {
            let k = (p.hdr.k_eff as usize).clamp(1, MAX_CLASSES);
            let f: Vec<f64> = logits[..k].iter().map(|&v| f64::from(v)).collect();
            let idx = categorical(&f, rng);
            p.card.classes.get(idx).copied().unwrap_or(idx as i32) as f32
        }
    }
}

/// The packet's linear target decoder on real rows (packet coordinates, step-0 weights).
fn packet_logits_on(p: &Packet, a: &Rows) -> Vec<[f32; MAX_CLASSES]> {
    let x = p.card.forward_x(&columns(a));
    (0..a.n)
        .map(|i| {
            let mut logits = [0.0f32; MAX_CLASSES];
            for (k, out) in logits.iter_mut().enumerate() {
                let mut lin = p.pkt.b_base[k];
                for (c, col) in x.iter().enumerate() {
                    let keep = (1.0 - p.pkt.is_const[c]) * p.pkt.col_mask[c];
                    lin += col[i] * keep * p.pkt.w_base[k][c];
                }
                *out = lin;
            }
            logits
        })
        .collect()
}

// -- trees ----------------------------------------------------------------------------------
/// A's mid-rank empirical CDF per column: the input map of every tree arm.
struct Cdf {
    sorted: Vec<Vec<f64>>,
}

impl Cdf {
    fn fit(a: &Rows) -> Self {
        let sorted = columns(a)
            .into_iter()
            .map(|mut col| {
                col.retain(|v| v.is_finite());
                col.sort_by(f64::total_cmp);
                col
            })
            .collect();
        Self { sorted }
    }

    fn code(&self, j: usize, v: f64) -> u8 {
        let s = &self.sorted[j];
        if !v.is_finite() || s.is_empty() {
            return code(0.5);
        }
        let lo = s.partition_point(|&w| w < v);
        let hi = s.partition_point(|&w| w <= v);
        code((lo + hi) as f64 / (2 * s.len()) as f64)
    }

    /// Column-major codes (`codes[j * n + i]`) of `n` rows read through `get(i, j)`.
    fn codes(&self, n: usize, get: impl Fn(usize, usize) -> f32) -> Vec<u8> {
        let p = self.sorted.len();
        let mut out = vec![0u8; p * n];
        for j in 0..p {
            for i in 0..n {
                out[j * n + i] = self.code(j, f64::from(get(i, j)));
            }
        }
        out
    }
}

struct TreeModel {
    head: Head,
    forest: Forest,
    base: Vec<f64>,
    y_mean: f64,
    y_sd: f64,
    resid_sd: f64,
}

struct TreeRecipe {
    depth: usize,
    trees: usize,
    eta: f64,
    patience: usize,
    min_rows: usize,
    inner_val: bool,
}

fn recipe(arm_id: &str) -> TreeRecipe {
    match arm_id {
        PACKET_X_GAM_Y => TreeRecipe {
            depth: 1,
            trees: 200,
            eta: 0.2,
            patience: 20,
            min_rows: 10,
            inner_val: true,
        },
        PACKET_X_TEACHER_Y => TreeRecipe {
            depth: 5,
            trees: 300,
            eta: 0.1,
            patience: 30,
            min_rows: 10,
            inner_val: true,
        },
        // the small-tree oracle: exactly 40 depth-3 trees, no early stop
        _ => TreeRecipe {
            depth: 3,
            trees: 40,
            eta: 0.2,
            patience: 40,
            min_rows: 10,
            inner_val: false,
        },
    }
}

fn prior(head: Head, y: &[f64]) -> Vec<f64> {
    let n = y.len().max(1) as f64;
    match head {
        Head::Regression => vec![0.0],
        Head::Binary => {
            let p = (y.iter().sum::<f64>() / n).clamp(1e-3, 1.0 - 1e-3);
            vec![(p / (1.0 - p)).ln()]
        }
        Head::Multiclass(k) => {
            let mut counts = vec![0.5f64; k];
            for &v in y {
                if let Some(c) = counts.get_mut(v as usize) {
                    *c += 1.0;
                }
            }
            let total: f64 = counts.iter().sum();
            counts.iter().map(|c| (c / total).ln()).collect()
        }
    }
}

fn fit_trees(a: &Rows, head: Head, cdf: &Cdf, r: &TreeRecipe, fit_seed: u64) -> TreeModel {
    let raw: Vec<f64> = a.y.iter().map(|&v| f64::from(v)).collect();
    let (y_mean, y_sd) = if head == Head::Regression {
        let m = raw.iter().sum::<f64>() / raw.len().max(1) as f64;
        let v = raw.iter().map(|y| (y - m).powi(2)).sum::<f64>() / raw.len().max(1) as f64;
        (m, v.sqrt().max(1e-9))
    } else {
        (0.0, 1.0)
    };
    let y: Vec<f64> = raw.iter().map(|v| (v - y_mean) / y_sd).collect();
    let base = prior(head, &y);
    let mut order: Vec<usize> = (0..a.n).collect();
    let n_val = if r.inner_val {
        (a.n / 5).max(1).min(a.n - 1)
    } else {
        0
    };
    if r.inner_val {
        KpiRng::new(fit_seed ^ 0x5EED_0F1A_11CE).shuffle(&mut order);
    }
    let (val_rows, fit_rows) = order.split_at(n_val);
    let part = |rows: &[usize]| {
        let codes = cdf.codes(rows.len(), |i, j| a.x[rows[i] * a.p + j]);
        let ys: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
        let b: Vec<f64> = (0..rows.len()).flat_map(|_| base.iter().copied()).collect();
        (codes, b, ys)
    };
    let (fc, fb, fy) = part(fit_rows);
    let (vc, vb, vy) = if r.inner_val {
        part(val_rows)
    } else {
        (fc.clone(), fb.clone(), fy.clone())
    };
    let opts = BoostOptions {
        depth: r.depth,
        bits: 6,
        eta: r.eta,
        lambda: 1.0,
        max_trees: r.trees,
        patience: r.patience,
        min_rows: r.min_rows,
        sample_rows: 20_000,
        seed: fit_seed,
    };
    let fit = Split {
        codes: &fc,
        base: &fb,
        y: &fy,
    };
    let val = Split {
        codes: &vc,
        base: &vb,
        y: &vy,
    };
    let boosted = boost(head, a.p, &fit, &val, &opts);
    let best = boosted
        .val_path
        .iter()
        .enumerate()
        .min_by(|x, y| x.1.total_cmp(y.1))
        .map_or(0, |(t, _)| t);
    let forest = if r.inner_val {
        boosted.forest.truncate(best)
    } else {
        boosted.forest
    };
    let mut model = TreeModel {
        head,
        forest,
        base,
        y_mean,
        y_sd,
        resid_sd: 0.0,
    };
    if head == Head::Regression {
        let all = cdf.codes(a.n, |i, j| a.x[i * a.p + j]);
        let f = model.outputs(&all, a.n, a.p);
        let sq: f64 = f.iter().zip(&y).map(|(p, t)| (p - t).powi(2)).sum();
        model.resid_sd = (sq / a.n.max(1) as f64).sqrt();
    }
    model
}

impl TreeModel {
    fn outputs(&self, codes: &[u8], n: usize, d: usize) -> Vec<f64> {
        let mut out = predict_columns(&self.forest, codes, d, n);
        let o = self.base.len();
        for (i, v) in out.iter_mut().enumerate() {
            *v += self.base[i % o];
        }
        out
    }

    fn label(&self, f: &[f64], rng: &mut KpiRng) -> f32 {
        match self.head {
            Head::Regression => {
                (self.y_mean + self.y_sd * (f[0] + self.resid_sd * rng.normal())) as f32
            }
            Head::Binary => {
                let p1 = 1.0 / (1.0 + (-f[0].clamp(-30.0, 30.0)).exp());
                f32::from(u8::from(rng.uniform() < p1))
            }
            Head::Multiclass(_) => categorical(f, rng) as f32,
        }
    }
}

/// One categorical draw from logits.
fn categorical(logits: &[f64], rng: &mut KpiRng) -> usize {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = logits.iter().map(|&v| (v - top).max(-50.0).exp()).collect();
    let u = rng.uniform() * w.iter().sum::<f64>();
    let mut acc = 0.0;
    for (k, &p) in w.iter().enumerate() {
        acc += p;
        if u <= acc {
            return k;
        }
    }
    w.len().saturating_sub(1)
}

// -- arms -----------------------------------------------------------------------------------
/// The synthetic table of one oracle arm, or the reason the cell has none.
pub fn generate(
    arm_id: &str,
    cell: &Cell,
    data: &CellData,
    sample_seed: u64,
    multiplier: u32,
) -> ArenaResult<Result<Mat, &'static str>> {
    let (a, head) = (&data.a, data.head);
    let m = a.n * multiplier as usize;
    let mut rng = KpiRng::new(seed(
        cell,
        sample_seed,
        &format!("oracle|{arm_id}|m{multiplier}|labels"),
    ));
    if matches!(arm_id, EXACT_RANK12 | EXACT_FULL) {
        let rank = (arm_id == EXACT_RANK12).then_some(COPULA_RANK);
        return Ok(Ok(Mat::new(m, a.p, copula_rows(a, rank, m, &mut rng))?));
    }
    let decode_seed = seed(cell, sample_seed, &format!("oracle|decode|m{multiplier}"));
    let fit_seed = seed(
        cell,
        STRUCTURAL_SAMPLE_SEED,
        &format!("oracle|fit|{arm_id}"),
    );
    let real_x = matches!(arm_id, REAL_X_PACKET_Y | REAL_X_TREE_Y);
    let needs_packet = !matches!(arm_id, REAL_X_TREE_Y);
    let packet = if needs_packet {
        match fit_packet(a, head) {
            Some(p) => Some(p),
            None => return Ok(Err(REASON_PACKET)),
        }
    } else {
        None
    };
    // X rows (raw units, `a.p` per row) and, for the packet arm, its logits and regression draw.
    let mut x: Vec<f32> = Vec::with_capacity(m * a.p);
    let mut packet_out: Vec<([f32; MAX_CLASSES], f32)> = Vec::new();
    if real_x {
        for i in 0..m {
            x.extend_from_slice(a.row(i % a.n));
        }
    } else {
        let p = packet.as_ref().expect("packet arms fit a packet");
        for row in sample_zero_init_rows(&p.pkt, m, decode_seed) {
            let raw = p.card.inverse_row_x(&row.x);
            x.extend_from_slice(&raw[..a.p]);
            packet_out.push((row.logits, row.y_reg));
        }
    }
    let y: Vec<f32> = match arm_id {
        PACKET => {
            let p = packet.as_ref().expect("packet");
            packet_out
                .iter()
                .map(|(l, yr)| packet_label(p, l, *yr, &mut rng))
                .collect()
        }
        REAL_X_PACKET_Y => {
            let p = packet.as_ref().expect("packet");
            let logits = packet_logits_on(p, a);
            (0..m)
                .map(|i| {
                    let l = &logits[i % a.n];
                    let y_reg = l[0] + p.pkt.resid_sd * rng.normal() as f32;
                    packet_label(p, l, y_reg, &mut rng)
                })
                .collect()
        }
        _ => {
            let cdf = Cdf::fit(a);
            let model = fit_trees(a, head, &cdf, &recipe(arm_id), fit_seed);
            let codes = cdf.codes(m, |i, j| x[i * a.p + j]);
            let f = model.outputs(&codes, m, a.p);
            let o = model.base.len();
            (0..m)
                .map(|i| model.label(&f[i * o..(i + 1) * o], &mut rng))
                .collect()
        }
    };
    let mut body = Vec::with_capacity(m * (a.p + 1));
    for (i, &yi) in y.iter().enumerate() {
        body.extend_from_slice(&x[i * a.p..(i + 1) * a.p]);
        body.push(yi);
    }
    Ok(Ok(Mat::new(m, a.p, body)?))
}

#[cfg(test)]
mod tests;
