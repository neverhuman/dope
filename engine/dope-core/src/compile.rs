//! `fit`: compile a table into a byte-budgeted program.
//!
//! Byte allocation is *target-first*: after the schema and marginals, a
//! minimal dependence model is reserved, the conditional model is sized to
//! the budget (up to three capacities are compared on held-out rows), and
//! the remaining bytes upgrade the dependence model (tree → factor → full)
//! in order of held-out likelihood. This follows the retention
//! decomposition (paper, Thm. 1): conditional-mean error enters retention
//! directly, covariate error only through a learner's extrapolation.

use crate::budget::{ModelSize, ParameterCount};
use crate::copula::Dependence;
use crate::error::{DopeError, Result};
use crate::eval::SelfAudit;
use crate::forest::column_codes;
use crate::program::Program;
use crate::regimes::{Regimes, MAX_REGIMES, REGIME_LADDER};
use crate::rng::Rng;
use crate::schema::{infer, IntakeOptions, Kind, Schema, Task};
use crate::screen::{associations, latent_correlation, ranking};
use crate::table::RawTable;
use crate::target::null_loss;
use candle_core::Device;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Instant;

/// Capacity ladder (knots, inputs, hidden units), richest first.
pub const PLANS: [(usize, usize, usize); 11] = [
    (33, 32, 48),
    (17, 24, 32),
    (17, 16, 24),
    (9, 12, 16),
    (9, 10, 12),
    (7, 8, 8),
    (5, 6, 6),
    (5, 4, 4),
    (3, 3, 2),
    (3, 2, 0),
    (3, 1, 0),
];

/// Default budget when none is given (bytes, whole file).
pub const DEFAULT_BUDGET: usize = 4096;

/// Shares of the budget the dependence model may reserve while the conditional
/// model is sized (target-first allocation, paper Prop. 2); the self-audit
/// picks among the resulting programs.
pub const DEPENDENCE_SHARES: [f64; 3] = [0.05, 0.15, 0.3];

/// Most trees a forest may hold.
pub const MAX_TREES: usize = 1500;

/// Bits per leaf code (symmetric codes, one f16 scale per tree).
pub const TREE_LEAF_BITS: u8 = 6;

/// Header and metadata bytes added by a non-empty forest (tensor entry + meta).
const FOREST_OVERHEAD: usize = 96;

#[derive(Clone, Debug)]
pub struct FitOptions {
    /// Maximum artifact size in bytes (whole safetensors file).
    pub budget: Option<usize>,
    /// Optional preset; an explicit byte budget overrides its byte cap.
    pub size: Option<ModelSize>,
    /// Maximum complete-generator scalar count (ParameterCount version 1).
    pub max_parameters: Option<usize>,
    pub seed: u64,
    pub task: Option<Task>,
    /// Replace column names and category labels by positional codes.
    pub anonymize: bool,
    pub max_rows: usize,
    /// Quantization may give up at most this share of the stored gain.
    pub tolerance: f64,
    pub max_steps: usize,
    pub device: Device,
    /// Force a (knots, inputs, hidden) plan (ablations).
    pub plan: Option<(usize, usize, usize)>,
    /// Force a dependence family: independent | tree | factorN | full.
    pub dependence: Option<String>,
    /// Force the number of regimes (1 = plain Gaussian copula).
    pub regimes: Option<usize>,
    /// Add boosted oblivious trees to the conditional model.
    pub trees: bool,
    /// Choose the byte split between dependence and the conditional model by
    /// self-audited retention on the validation rows.
    pub audit: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            budget: None,
            size: None,
            max_parameters: None,
            seed: 0,
            task: None,
            anonymize: false,
            max_rows: 200_000,
            tolerance: 0.005,
            max_steps: 1500,
            device: Device::Cpu,
            plan: None,
            dependence: None,
            regimes: None,
            trees: true,
            audit: true,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FitReport {
    pub task: String,
    pub rows: usize,
    pub features: usize,
    pub budget: usize,
    pub bytes: usize,
    pub size: Option<ModelSize>,
    pub max_parameters: Option<usize>,
    pub parameters: ParameterCount,
    pub knots: usize,
    pub inputs: usize,
    pub hidden: usize,
    pub bits: [u8; 4],
    /// Boosted trees in the conditional model, and their depth.
    pub trees: usize,
    pub depth: usize,
    pub dependence: String,
    pub regimes: usize,
    pub screened: Vec<String>,
    /// 1 − held-out loss / null loss of the *stored, quantized* conditional model.
    pub stored_gain: f64,
    /// Robust retention of fast linear and tree auditors trained on the
    /// program's rows, scored on the compiler's own validation rows.
    pub self_audit: Option<f64>,
    pub seconds: f64,
    pub device: String,
}

pub struct Fitted {
    pub bytes: Vec<u8>,
    pub program: Program,
    pub report: FitReport,
}

fn anonymize(schema: &mut Schema) {
    for (j, f) in schema.features.iter_mut().enumerate() {
        f.name = format!("x{j}");
        if let Kind::Categorical(c) = &mut f.kind {
            c.levels = (0..c.levels.len()).map(|k| format!("L{k}")).collect();
        }
    }
    schema.target.name = "y".into();
    if let Kind::Categorical(c) = &mut schema.target.kind {
        c.levels = (0..c.levels.len()).map(|k| format!("C{k}")).collect();
    }
    schema.dropped.clear();
}

mod allocate;
mod ctx;

use allocate::{allocate, Candidate};
use ctx::Ctx;

/// Compile `raw` with column `target` into an artifact.
pub fn fit(raw: &RawTable, target: usize, opts: &FitOptions) -> Result<Fitted> {
    let start = Instant::now();
    let mut resolved = opts.clone();
    resolved.budget = Some(
        opts.budget
            .or_else(|| opts.size.map(ModelSize::budget))
            .unwrap_or(DEFAULT_BUDGET),
    );
    let opts = &resolved;
    if opts.budget == Some(0) || opts.max_parameters == Some(0) {
        return Err(DopeError::Budget(
            "byte and parameter caps must be positive".into(),
        ));
    }
    let intake = IntakeOptions {
        task: opts.task,
        ..Default::default()
    };
    let mut schema = infer(raw, target, &intake)?;
    let data = schema.to_model(raw)?;
    if opts.anonymize {
        anonymize(&mut schema);
    }
    let head = Program::head_for(&schema);
    let d = schema.features.len();
    let mut rows: Vec<usize> = (0..data.rows())
        .filter(|&r| data.y[r].is_finite())
        .collect();
    if rows.len() < 20 {
        return Err(DopeError::Input(
            "need at least 20 rows with an observed target".into(),
        ));
    }
    let mut rng = Rng::new(opts.seed ^ 0x5eed);
    rng.shuffle(&mut rows);
    rows.truncate(opts.max_rows);
    let n_val = ((rows.len() as f64 * 0.15).round() as usize).clamp(10, 5000);
    let (val_rows, fit_rows) = (rows[..n_val].to_vec(), rows[n_val..].to_vec());
    let y_fit: Vec<f64> = fit_rows.iter().map(|&r| data.y[r]).collect();
    let y_val: Vec<f64> = val_rows.iter().map(|&r| data.y[r]).collect();
    let missing = (0..d)
        .map(|j| {
            rows.iter().filter(|&&r| !data.x[j][r].is_finite()).count() as f64 / rows.len() as f64
        })
        .collect();
    let x_fit: Vec<Vec<f64>> = (0..d)
        .map(|j| fit_rows.iter().map(|&r| data.x[j][r]).collect())
        .collect();
    let order = ranking(&associations(&x_fit, &y_fit, head, 5000));
    let mut ctx = Ctx {
        schema,
        data: &data,
        head,
        rows,
        fit_rows,
        val_rows,
        y_fit,
        y_val,
        missing,
        order,
        impute: Vec::new(),
        opts,
        marginals: HashMap::new(),
        nets: HashMap::new(),
        codes_fit: Vec::new(),
        codes_val: Vec::new(),
        boosts: HashMap::new(),
    };
    let trace = std::env::var_os("DOPE_TRACE").is_some();
    if trace {
        eprintln!("intake+screen {:.2}s", start.elapsed().as_secs_f64());
    }
    let base = ctx.marginals(9);
    ctx.impute = base.iter().map(|m| m.at(0.0)).collect();
    ctx.codes_fit = column_codes(&data.x, &ctx.fit_rows, &ctx.impute);
    ctx.codes_val = column_codes(&data.x, &ctx.val_rows, &ctx.impute);

    // Dependence candidates (regimes × family), ranked by held-out pseudo-likelihood.
    let (r_fit, _, sds) = latent_correlation(&base, &data.x, &ctx.fit_rows, 20_000);
    let standardized = |rows: &[usize]| -> Vec<f64> {
        let mut out = vec![0.0; rows.len() * d];
        for j in 0..d {
            let (s, _) = base[j].scores(&ctx.column(j, rows));
            for (i, v) in s.into_iter().enumerate() {
                out[i * d + j] = v / sds[j];
            }
        }
        out
    };
    let sub = |rows: &[usize], cap: usize| -> Vec<usize> {
        rows.iter()
            .step_by(rows.len().div_ceil(cap).max(1))
            .copied()
            .collect()
    };
    let (fit_sub, val_sub) = (sub(&ctx.fit_rows, 5000), sub(&ctx.val_rows, 2000));
    let (fit_scores, val_scores) = (standardized(&fit_sub), standardized(&val_sub));
    let mut deps: Vec<(f64, (Regimes, Dependence))> = Vec::new();
    let ks: Vec<usize> = match opts.regimes {
        Some(k) => vec![k.clamp(1, MAX_REGIMES)],
        None => REGIME_LADDER
            .iter()
            .copied()
            .filter(|&k| fit_sub.len() >= 50 * k)
            .collect(),
    };
    for k in ks {
        let reg = if k == 1 {
            Regimes::single(d)
        } else {
            Regimes::fit(&fit_scores, fit_sub.len(), d, k, opts.seed)
        };
        if k > 1 && reg.k() == 1 {
            continue;
        }
        let r = if k == 1 {
            r_fit.clone()
        } else {
            let mut w = reg.within_correlation(&fit_scores, fit_sub.len(), d);
            for a in 0..d {
                for b in 0..d {
                    if a != b {
                        w[a * d + b] = (w[a * d + b] / (sds[a] * sds[b])).clamp(-0.995, 0.995);
                    }
                }
            }
            crate::stats::nearest_correlation(&w, d, 1e-3)
        };
        for dep in candidates(&r, d) {
            let ll = reg.log_likelihood(&dep, &val_scores, val_sub.len(), d);
            deps.push((ll, (reg.clone(), dep)));
        }
    }
    if let Some(force) = &opts.dependence {
        deps.retain(|(_, (_, dep))| dep.name() == *force);
        if deps.is_empty() {
            return Err(DopeError::Input(
                "unknown or unavailable dependence family".into(),
            ));
        }
    }
    deps.sort_by(|a, b| b.0.total_cmp(&a.0));
    if trace {
        eprintln!("marginals+dependence {:.2}s", start.elapsed().as_secs_f64());
    }
    let budget = opts.budget.unwrap_or(DEFAULT_BUDGET);
    let null_val = null_loss(head, &ctx.y_val);
    // How much of the budget may the dependence model take while the
    // conditional model is sized? Each share yields one candidate program; the
    // compiler keeps the one with the best self-audited robust retention on its
    // own validation rows.
    let shares: Vec<f64> = match std::env::var("DOPE_DEP_SHARE")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(v) => vec![v],
        None if opts.audit => DEPENDENCE_SHARES.to_vec(),
        None => vec![DEPENDENCE_SHARES[DEPENDENCE_SHARES.len() / 2]],
    };
    let mut best: Option<(f64, Candidate)> = None;
    let mut audit: Option<SelfAudit> = None;
    let mut seen: Vec<Vec<u8>> = Vec::new();
    for &share in &shares {
        let Some(cand) = allocate(&mut ctx, &deps, &r_fit, budget, share)? else {
            continue;
        };
        if seen.contains(&cand.bytes) {
            continue;
        }
        seen.push(cand.bytes.clone());
        let score = if shares.len() > 1 {
            if audit.is_none() {
                audit = Some(SelfAudit::new(
                    &cand.program,
                    &data,
                    &ctx.fit_rows,
                    &ctx.val_rows,
                    opts.seed,
                )?);
            }
            audit
                .as_ref()
                .map_or(Ok(f64::NAN), |a| a.retention(&cand.program))?
        } else {
            f64::NAN
        };
        if trace {
            eprintln!(
                "share {share}: {} bytes, {} x {}, net {}->{}, {} trees, self-audit {score:.4}",
                cand.bytes.len(),
                cand.program.dependence.name(),
                cand.program.regimes.k(),
                cand.s,
                cand.h,
                cand.entry.forest.trees()
            );
        }
        let better = match &best {
            None => true,
            Some((b, _)) => score > *b + 1e-3,
        };
        if better {
            best = Some((score, cand));
        }
    }
    let Some((
        self_audit,
        Candidate {
            k,
            s,
            h,
            entry,
            bytes,
            program,
        },
    )) = best
    else {
        return Err(DopeError::Budget(format!(
            "no candidate fits {budget} bytes and the requested parameter cap; increase the caps or use --anonymize"
        )));
    };
    let q = &entry.q;
    let report = FitReport {
        task: format!("{:?}", ctx.schema.task).to_lowercase(),
        rows: ctx.rows.len(),
        features: d,
        budget,
        bytes: bytes.len(),
        size: opts.size,
        max_parameters: opts.max_parameters,
        parameters: program.parameters(),
        knots: k,
        inputs: s,
        hidden: h,
        bits: [q.w.bits, q.b.bits, q.beta.bits, q.v.bits],
        trees: entry.forest.trees(),
        depth: entry.forest.depth,
        dependence: program.dependence.name(),
        regimes: program.regimes.k(),
        screened: ctx.order[..s]
            .iter()
            .map(|&j| ctx.schema.features[j].name.clone())
            .collect(),
        stored_gain: if null_val > 0.0 {
            1.0 - entry.val_loss / null_val
        } else {
            0.0
        },
        self_audit: self_audit.is_finite().then_some(self_audit),
        seconds: start.elapsed().as_secs_f64(),
        device: format!("{:?}", opts.device.location()),
    };
    Ok(Fitted {
        bytes,
        program,
        report,
    })
}

fn candidates(r: &[f64], d: usize) -> Vec<Dependence> {
    let mut c = vec![Dependence::Independent];
    if d >= 2 {
        c.push(Dependence::fit_tree(r, d));
        for rank in [1, 2, 4, 8].into_iter().filter(|&q| q < d) {
            c.push(Dependence::fit_factor(r, d, rank));
        }
        c.push(Dependence::fit_full(r, d));
    }
    c
}
