//! Shared state of one fit: data splits, cached marginals, networks and
//! forests, and assembly of candidate programs into artifacts.

use super::{FitOptions, DEFAULT_BUDGET, FOREST_OVERHEAD, MAX_TREES, TREE_LEAF_BITS};
use crate::artifact::{decode, encode};
use crate::copula::Dependence;
use crate::error::Result;
use crate::forest::{boost, predict_columns, BoostOptions, Boosted, Forest, Split};
use crate::marginal::Marginal;
use crate::noise::{fit_regression, fit_temperature, Noise, RESIDUAL_KNOTS};
use crate::program::Program;
use crate::quant::QTensor;
use crate::regimes::Regimes;
use crate::schema::{Kind, ModelData, Schema};
use crate::stats::quantile_sorted;
use crate::target::{allocate_bits, loss, Head, QuantNet};
use crate::train::{train, TrainOptions};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone)]
pub(super) struct NetEntry {
    pub(super) q: QuantNet,
    pub(super) forest: Forest,
    pub(super) noise: Noise,
    pub(super) val_loss: f64,
    pub(super) range: (f64, f64),
}

/// Everything shared by the candidate programs of one fit.
pub(super) struct Ctx<'a> {
    pub(super) schema: Schema,
    pub(super) data: &'a ModelData,
    pub(super) head: Head,
    pub(super) rows: Vec<usize>,
    pub(super) fit_rows: Vec<usize>,
    pub(super) val_rows: Vec<usize>,
    pub(super) y_fit: Vec<f64>,
    pub(super) y_val: Vec<f64>,
    pub(super) missing: Vec<f64>,
    pub(super) order: Vec<usize>,
    pub(super) impute: Vec<f64>,
    pub(super) opts: &'a FitOptions,
    pub(super) marginals: HashMap<usize, Vec<Marginal>>,
    pub(super) nets: HashMap<(usize, usize), NetEntry>,
    /// Column-major input codes of the fitting and validation rows (trees).
    pub(super) codes_fit: Vec<u8>,
    pub(super) codes_val: Vec<u8>,
    pub(super) boosts: HashMap<(usize, usize), Boosted>,
}

impl Ctx<'_> {
    pub(super) fn d(&self) -> usize {
        self.schema.features.len()
    }

    pub(super) fn column(&self, j: usize, rows: &[usize]) -> Vec<f64> {
        rows.iter().map(|&r| self.data.x[j][r]).collect()
    }

    pub(super) fn marginals(&mut self, k: usize) -> Vec<Marginal> {
        if !self.marginals.contains_key(&k) {
            let m = (0..self.d())
                .map(|j| {
                    let levels = match &self.schema.features[j].kind {
                        Kind::Categorical(c) => Some(c.levels.len()),
                        Kind::Numeric(_) => None,
                    };
                    Marginal::fit(&self.column(j, &self.rows), k, levels)
                })
                .collect();
            self.marginals.insert(k, m);
        }
        self.marginals[&k].clone()
    }

    pub(super) fn inputs(&self, s: usize, rows: &[usize]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows.len() * s);
        for &r in rows {
            for &f in &self.order[..s] {
                let v = self.data.x[f][r];
                out.push(if v.is_finite() { v } else { self.impute[f] } - 0.5);
            }
        }
        out
    }

    pub(super) fn net(&mut self, s: usize, h: usize) -> Result<&NetEntry> {
        if !self.nets.contains_key(&(s, h)) {
            let entry = self.train_net(s, h)?;
            self.nets.insert((s, h), entry);
        }
        Ok(&self.nets[&(s, h)])
    }

    pub(super) fn train_net(&self, s: usize, h: usize) -> Result<NetEntry> {
        let t0 = Instant::now();
        let (head, opts) = (self.head, self.opts);
        let (xf, xv) = (
            self.inputs(s, &self.fit_rows),
            self.inputs(s, &self.val_rows),
        );
        let n_val = self.y_val.len();
        let topts = TrainOptions {
            hidden: h,
            max_steps: opts.max_steps,
            seed: opts.seed,
            ..Default::default()
        };
        let net = train(
            head,
            s,
            (&xf, &self.y_fit),
            (&xv, &self.y_val),
            &topts,
            &opts.device,
        )?;
        let t1 = Instant::now();
        let q = allocate_bits(
            &net,
            head,
            (&xf, &self.y_fit),
            (&xv, &self.y_val),
            opts.tolerance,
        );
        if std::env::var_os("DOPE_TRACE").is_some() {
            let vl = loss(head, &net.forward(&xv, n_val), &self.y_val);
            eprintln!(
                "net s={s} h={h}: train {:.2}s, allocate {:.2}s, float val loss {vl:.5}",
                (t1 - t0).as_secs_f64(),
                t1.elapsed().as_secs_f64()
            );
        }
        let qn = q.net();
        let out_val = qn.forward(&xv, n_val);
        let val_loss = loss(head, &out_val, &self.y_val);
        let noise = match head {
            Head::Regression => {
                // Held-out residuals, unless the validation split is tiny.
                let (xr, yr) = if n_val >= 50 {
                    (&xv, &self.y_val)
                } else {
                    (&xf, &self.y_fit)
                };
                let pred = qn.forward(xr, yr.len());
                let r: Vec<f64> = yr.iter().zip(&pred).map(|(y, p)| y - p).collect();
                fit_regression(&r, xr, s)
            }
            _ => fit_temperature(head, &out_val, &self.y_val),
        };
        let mut sorted = self.y_fit.clone();
        sorted.sort_by(f64::total_cmp);
        let range = (
            quantile_sorted(&sorted, 0.0005),
            quantile_sorted(&sorted, 0.9995),
        );
        Ok(NetEntry {
            q,
            forest: Forest::empty(head.outputs()),
            noise,
            val_loss,
            range,
        })
    }

    pub(super) fn empty_net(&self) -> NetEntry {
        let o = self.head.outputs();
        NetEntry {
            q: QuantNet {
                inputs: 0,
                hidden: 0,
                outputs: o,
                w: QTensor::quantize(&[], 8),
                b: QTensor::quantize(&[], 8),
                beta: QTensor::quantize(&[], 8),
                v: QTensor::quantize(&[], 8),
                a: vec![0.0; o],
            },
            forest: Forest::empty(o),
            noise: match self.head {
                Head::Regression => Noise::Gaussianish {
                    coef: vec![0.0],
                    lo: 0.0,
                    hi: 1.0,
                    knots: vec![0.0; RESIDUAL_KNOTS],
                },
                _ => Noise::Temperature(1.0),
            },
            val_loss: f64::INFINITY,
            range: (0.0, 1.0),
        }
    }

    /// Tree depth and leaf bits for this budget.
    pub(super) fn forest_shape(&self) -> (usize, u8) {
        let budget = self.opts.budget.unwrap_or(DEFAULT_BUDGET);
        let depth = match budget {
            b if b < 8 * 1024 => 4,
            b if b < 32 * 1024 => 5,
            _ => 6,
        };
        (depth, TREE_LEAF_BITS)
    }

    /// Boost a forest on the quantized network `(s, h)` once per fit (cached);
    /// every allocation then keeps the best prefix that fits its bytes.
    pub(super) fn boost(&mut self, s: usize, h: usize) {
        if self.boosts.contains_key(&(s, h)) || !self.opts.trees || self.d() == 0 {
            return;
        }
        let (head, d) = (self.head, self.d());
        let (depth, bits) = self.forest_shape();
        let budget = self.opts.budget.unwrap_or(DEFAULT_BUDGET);
        let per_tree = Forest::byte_len(1, depth, head.outputs(), bits, d);
        let max_trees = (budget / per_tree).clamp(1, MAX_TREES);
        let nominal = (budget / 2 / per_tree).max(1);
        let qn = self.nets[&(s, h)].q.net();
        let (xf, xv) = (
            self.inputs(s, &self.fit_rows),
            self.inputs(s, &self.val_rows),
        );
        let (bf, bv) = (
            qn.forward(&xf, self.y_fit.len()),
            qn.forward(&xv, self.y_val.len()),
        );
        let opts = BoostOptions {
            depth,
            bits,
            eta: (3.0 / (nominal as f64).sqrt()).clamp(0.08, 0.3),
            max_trees,
            patience: (nominal / 10).max(30),
            seed: self.opts.seed,
            ..Default::default()
        };
        let boosted = boost(
            head,
            d,
            &Split {
                codes: &self.codes_fit,
                base: &bf,
                y: &self.y_fit,
            },
            &Split {
                codes: &self.codes_val,
                base: &bv,
                y: &self.y_val,
            },
            &opts,
        );
        if std::env::var_os("DOPE_TRACE").is_some() {
            let p = &boosted.val_path;
            let best = p.iter().copied().fold(f64::INFINITY, f64::min);
            eprintln!(
                "forest s={s} h={h}: {} trees of depth {depth} boosted, val loss {:.5} -> {best:.5}",
                p.len() - 1,
                p[0]
            );
        }
        self.boosts.insert((s, h), boosted);
    }

    /// Network `(s, h)` plus the best forest prefix within `room` bytes.
    pub(super) fn with_forest(&self, s: usize, h: usize, e: &NetEntry, room: usize) -> NetEntry {
        let Some(boosted) = self.boosts.get(&(s, h)) else {
            return e.clone();
        };
        let (depth, bits) = self.forest_shape();
        let per_tree = Forest::byte_len(1, depth, self.head.outputs(), bits, self.d());
        let fit = room.saturating_sub(FOREST_OVERHEAD) / per_tree;
        let path = &boosted.val_path;
        let best = (0..path.len().min(fit + 1))
            .min_by(|&a, &b| path[a].total_cmp(&path[b]))
            .unwrap_or(0);
        if best == 0 || path[best] >= path[0] - 1e-9 {
            return e.clone();
        }
        self.finish(s, e, boosted.forest.truncate(best))
    }

    /// Keep the first `trees` trees of `e`'s forest.
    pub(super) fn retrim(&self, s: usize, e: &NetEntry, trees: usize) -> NetEntry {
        self.finish(s, e, e.forest.truncate(trees))
    }

    /// An entry with `forest` on top of `e`'s network: held-out loss, and the
    /// noise model refitted on the combined predictions.
    pub(super) fn finish(&self, s: usize, e: &NetEntry, forest: Forest) -> NetEntry {
        let (head, d) = (self.head, self.d());
        let qn = e.q.net();
        let n_val = self.y_val.len();
        let (xf, xv) = (
            self.inputs(s, &self.fit_rows),
            self.inputs(s, &self.val_rows),
        );
        let combined = |x: &[f64], codes: &[u8], n: usize| -> Vec<f64> {
            let mut out = qn.forward(x, n);
            for (a, b) in out.iter_mut().zip(predict_columns(&forest, codes, d, n)) {
                *a += b;
            }
            out
        };
        let out_val = combined(&xv, &self.codes_val, n_val);
        let noise = match head {
            Head::Regression => {
                let (xr, yr, pred) = if n_val >= 50 {
                    (&xv, &self.y_val, out_val.clone())
                } else {
                    (
                        &xf,
                        &self.y_fit,
                        combined(&xf, &self.codes_fit, self.y_fit.len()),
                    )
                };
                let r: Vec<f64> = yr.iter().zip(&pred).map(|(y, p)| y - p).collect();
                fit_regression(&r, xr, s)
            }
            _ => fit_temperature(head, &out_val, &self.y_val),
        };
        NetEntry {
            q: e.q.clone(),
            val_loss: loss(head, &out_val, &self.y_val),
            forest,
            noise,
            range: e.range,
        }
    }

    pub(super) fn assemble(
        &self,
        k: usize,
        margs: Vec<Marginal>,
        s: usize,
        e: &NetEntry,
        dep: &(Regimes, Dependence),
    ) -> Result<(Vec<u8>, Program)> {
        let program = Program {
            schema: self.schema.clone(),
            marginals: margs,
            missing: self.missing.clone(),
            dependence: dep.1.clone(),
            regimes: dep.0.clone(),
            screened: self.order[..s].to_vec(),
            net: e.q.net(),
            forest: e.forest.clone(),
            head: self.head,
            noise: e.noise.clone(),
            target_range: e.range,
        };
        let bytes = encode(&program, &e.q, k)?;
        let decoded = decode(&bytes)?;
        Ok((bytes, decoded))
    }
}
