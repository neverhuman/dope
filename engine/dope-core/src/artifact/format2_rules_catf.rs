//! Pure-Rust bounds-checked binary codecs for `RuleSet` and `CatfModel` (Format 2).

use super::format2_bin::{BinReader, BinWriter};
use crate::catf::fit::{standard_z_bins, CatfConditional, CatfPredicate, CatfTree};
use crate::catf::CatfModel;
use crate::error::{DopeError, Result};
use crate::marginal::v2::ColumnMarginalV2;
use crate::stats::norm_ppf;
use crate::structure::{RuleMode, RuleSet, StructuralRule};
use half::f16;
use std::collections::BTreeMap;

pub(crate) fn encode_rules_section(rules: &RuleSet) -> Vec<u8> {
    let mut w = BinWriter::new();
    w.u8(match rules.mode {
        RuleMode::None => 0,
        RuleMode::Structural => 1,
        RuleMode::All => 2,
    });
    w.u32(rules.rules.len() as u32);
    for rule in &rules.rules {
        match rule {
            StructuralRule::Fd1 {
                parent,
                child,
                map,
                epsilon,
            } => {
                w.u8(0);
                w.u32(*parent as u32);
                w.u32(*child as u32);
                w.f32(*epsilon);
                w.u32(map.len() as u32);
                for (&k, &v) in map {
                    w.i64(k);
                    w.f64(v);
                }
            }
            StructuralRule::Fd2 {
                parent_a,
                parent_b,
                child,
                map,
                epsilon,
            } => {
                w.u8(1);
                w.u32(*parent_a as u32);
                w.u32(*parent_b as u32);
                w.u32(*child as u32);
                w.f32(*epsilon);
                w.u32(map.len() as u32);
                for (&(ka, kb), &v) in map {
                    w.i64(ka);
                    w.i64(kb);
                    w.f64(v);
                }
            }
            StructuralRule::Arith {
                pa,
                pb,
                child,
                a,
                b,
                c,
                tol,
                epsilon,
            } => {
                w.u8(2);
                w.u32(*pa as u32);
                w.u32(*pb as u32);
                w.u32(*child as u32);
                w.f64(*a);
                w.f64(*b);
                w.f64(*c);
                w.f32(*tol);
                w.f32(*epsilon);
            }
            StructuralRule::Bin {
                parent,
                child,
                edges,
                bin_values,
                epsilon,
            } => {
                w.u8(3);
                w.u32(*parent as u32);
                w.u32(*child as u32);
                w.f32_slice(edges);
                w.f64_slice(bin_values);
                w.f32(*epsilon);
            }
            StructuralRule::Ineq {
                lo_col,
                hi_col,
                hi_cap,
                epsilon,
            } => {
                w.u8(4);
                w.u32(*lo_col as u32);
                w.u32(*hi_col as u32);
                w.f64(*hi_cap);
                w.f32(*epsilon);
            }
            StructuralRule::Excl {
                col_a,
                col_b,
                prob_a_nonzero,
                epsilon,
            } => {
                w.u8(5);
                w.u32(*col_a as u32);
                w.u32(*col_b as u32);
                w.f32(*prob_a_nonzero);
                w.f32(*epsilon);
            }
            StructuralRule::Miss {
                parent,
                child,
                trigger_value,
                epsilon,
            } => {
                w.u8(6);
                w.u32(*parent as u32);
                w.u32(*child as u32);
                match trigger_value {
                    None => w.u8(0),
                    Some(tv) => {
                        w.u8(1);
                        w.f64(*tv);
                    }
                }
                w.f32(*epsilon);
            }
        }
    }
    w.0
}

pub(crate) fn decode_rules_section(bytes: &[u8]) -> Result<RuleSet> {
    let mut r = BinReader::new(bytes);
    let mode = match r.u8()? {
        0 => RuleMode::None,
        1 => RuleMode::Structural,
        2 => RuleMode::All,
        _ => return Err(DopeError::Artifact("format2 invalid rule mode".into())),
    };
    let n = r.len()?;
    let mut rules = Vec::with_capacity(n);
    for _ in 0..n {
        let rule = match r.u8()? {
            0 => {
                let parent = r.u32()? as usize;
                let child = r.u32()? as usize;
                let epsilon = r.f32()?;
                let m = r.len()?;
                let mut map = BTreeMap::new();
                for _ in 0..m {
                    let k = r.i64()?;
                    let v = r.f64()?;
                    map.insert(k, v);
                }
                StructuralRule::Fd1 {
                    parent,
                    child,
                    map,
                    epsilon,
                }
            }
            1 => {
                let parent_a = r.u32()? as usize;
                let parent_b = r.u32()? as usize;
                let child = r.u32()? as usize;
                let epsilon = r.f32()?;
                let m = r.len()?;
                let mut map = BTreeMap::new();
                for _ in 0..m {
                    let ka = r.i64()?;
                    let kb = r.i64()?;
                    let v = r.f64()?;
                    map.insert((ka, kb), v);
                }
                StructuralRule::Fd2 {
                    parent_a,
                    parent_b,
                    child,
                    map,
                    epsilon,
                }
            }
            2 => StructuralRule::Arith {
                pa: r.u32()? as usize,
                pb: r.u32()? as usize,
                child: r.u32()? as usize,
                a: r.f64()?,
                b: r.f64()?,
                c: r.f64()?,
                tol: r.f32()?,
                epsilon: r.f32()?,
            },
            3 => StructuralRule::Bin {
                parent: r.u32()? as usize,
                child: r.u32()? as usize,
                edges: r.f32_vec()?,
                bin_values: r.f64_vec()?,
                epsilon: r.f32()?,
            },
            4 => StructuralRule::Ineq {
                lo_col: r.u32()? as usize,
                hi_col: r.u32()? as usize,
                hi_cap: r.f64()?,
                epsilon: r.f32()?,
            },
            5 => StructuralRule::Excl {
                col_a: r.u32()? as usize,
                col_b: r.u32()? as usize,
                prob_a_nonzero: r.f32()?,
                epsilon: r.f32()?,
            },
            6 => {
                let parent = r.u32()? as usize;
                let child = r.u32()? as usize;
                let trigger_value = if r.u8()? == 1 { Some(r.f64()?) } else { None };
                let epsilon = r.f32()?;
                StructuralRule::Miss {
                    parent,
                    child,
                    trigger_value,
                    epsilon,
                }
            }
            _ => return Err(DopeError::Artifact("format2 invalid rule tag".into())),
        };
        rules.push(rule);
    }
    Ok(RuleSet { mode, rules })
}

fn recompute_z_bins(marginal: &ColumnMarginalV2, k_bins: usize) -> (Vec<f64>, Vec<f64>) {
    match marginal {
        ColumnMarginalV2::Discrete { values, probs, .. } if !values.is_empty() => {
            let l = values.len();
            let mut edges = Vec::with_capacity(l + 1);
            edges.push(f64::NEG_INFINITY);
            let mut cum = 0.0;
            for &p in &probs[..l.saturating_sub(1)] {
                cum += p;
                edges.push(norm_ppf(cum.clamp(1e-9, 1.0 - 1e-9)));
            }
            edges.push(f64::INFINITY);
            let centers = (0..l)
                .map(|i| {
                    let lo = if edges[i].is_finite() { edges[i] } else { -2.5 };
                    let hi = if edges[i + 1].is_finite() {
                        edges[i + 1]
                    } else {
                        2.5
                    };
                    0.5 * (lo + hi)
                })
                .collect();
            (edges, centers)
        }
        _ => standard_z_bins(k_bins.max(2)),
    }
}

pub(crate) fn encode_catf_section(catf: &CatfModel) -> Vec<u8> {
    let mut w = BinWriter::new();
    w.usize_slice(&catf.order);
    w.u32(catf.is_nominal.len() as u32);
    for &b in &catf.is_nominal {
        w.u8(u8::from(b));
    }
    // `nom_levels` is deterministically recomputed from `marginals` on decode (0 bytes!).
    w.u32(0);
    w.u32(catf.conditionals.len() as u32);
    for c in &catf.conditionals {
        w.u32(c.col_idx as u32);
        // Compact v2.2 conditional flags: bit0=is_discrete, bit1=recompute_z_bins, bit2=f16_copula.
        let compact_copula = c.predecessors.len() <= u16::MAX as usize
            && c.predecessors.iter().all(|&p| p <= u16::MAX as usize)
            && c.copula_weights.iter().all(|&w| w.abs() <= 1.25);
        let flags = u8::from(c.is_discrete) | 2 | if compact_copula { 4 } else { 0 };
        w.u8(flags);
        w.u32(c.k_bins as u32);
        if compact_copula {
            w.u16(c.predecessors.len() as u16);
            for &p in &c.predecessors {
                w.u16(p as u16);
            }
            for &cw in &c.copula_weights {
                w.u16(f16::from_f64(cw).to_bits());
            }
        } else {
            w.usize_slice(&c.predecessors);
            w.f32_slice(&c.copula_weights);
        }
        w.f32(c.copula_cond_sd);
        w.f32(c.base_missing_prob);
        w.u32(c.trees.len() as u32);
        for t in &c.trees {
            w.u32(t.predicates.len() as u32);
            for p in &t.predicates {
                match *p {
                    CatfPredicate::NumericLess { feat, threshold } => {
                        w.u8(0);
                        w.u32(feat as u32);
                        w.f32(threshold);
                    }
                    CatfPredicate::NominalIn { feat, mask } => {
                        w.u8(1);
                        w.u32(feat as u32);
                        w.u64(mask);
                    }
                    CatfPredicate::IsMissing { feat } => {
                        w.u8(2);
                        w.u32(feat as u32);
                    }
                }
            }
            w.u8(u8::from(t.hermite_mode));
            w.u32(t.n_states as u32);
            w.f32_slice(&t.leaf_values);
        }
    }
    w.0
}

pub(crate) fn decode_catf_section(
    bytes: &[u8],
    marginals: Vec<ColumnMarginalV2>,
    rules: RuleSet,
) -> Result<CatfModel> {
    let mut r = BinReader::new(bytes);
    let order = r.usize_vec()?;
    let n_nom = r.len()?;
    let mut is_nominal = Vec::with_capacity(n_nom);
    for _ in 0..n_nom {
        is_nominal.push(r.u8()? != 0);
    }
    let n_lv = r.len()?;
    let nom_levels = if n_lv == 0 {
        marginals
            .iter()
            .map(|m| match m {
                ColumnMarginalV2::Discrete { values, .. } => values.clone(),
                _ => Vec::new(),
            })
            .collect()
    } else {
        let mut out = Vec::with_capacity(n_lv);
        for _ in 0..n_lv {
            out.push(r.f64_vec()?);
        }
        out
    };
    let n_cond = r.len()?;
    let mut conditionals = Vec::with_capacity(n_cond);
    for _ in 0..n_cond {
        let col_idx = r.u32()? as usize;
        let flags = r.u8()?;
        let is_discrete = (flags & 1) != 0;
        let recompute_bins = (flags & 2) != 0;
        let compact_copula = (flags & 4) != 0;
        let (
            predecessors,
            copula_weights,
            copula_cond_sd,
            base_missing_prob,
            k_bins,
            z_edges,
            z_centers,
        ) = if recompute_bins {
            let k_bins = r.u32()? as usize;
            let (preds, weights) = if compact_copula {
                let np = r.u16()? as usize;
                let mut ps = Vec::with_capacity(np);
                for _ in 0..np {
                    ps.push(r.u16()? as usize);
                }
                let mut ws = Vec::with_capacity(np);
                for _ in 0..np {
                    ws.push(f16::from_bits(r.u16()?).to_f64());
                }
                (ps, ws)
            } else {
                (r.usize_vec()?, r.f32_vec()?)
            };
            let sd = r.f32()?;
            let pm = r.f32()?;
            let (ze, zc) = marginals.get(col_idx).map_or_else(
                || standard_z_bins(k_bins.max(2)),
                |m| recompute_z_bins(m, k_bins),
            );
            (preds, weights, sd, pm, k_bins, ze, zc)
        } else {
            // Legacy v2.0 layout fallback (kept for clean error/format safety).
            let preds = r.usize_vec()?;
            let ws = r.f32_vec()?;
            let sd = r.f32()?;
            let pm = r.f32()?;
            let k_bins = r.u32()? as usize;
            let ze = r.f64_vec()?;
            let zc = r.f32_vec()?;
            (preds, ws, sd, pm, k_bins, ze, zc)
        };
        let nt = r.len()?;
        let mut trees = Vec::with_capacity(nt);
        for _ in 0..nt {
            let np = r.len()?;
            let mut predicates = Vec::with_capacity(np);
            for _ in 0..np {
                let p = match r.u8()? {
                    0 => CatfPredicate::NumericLess {
                        feat: r.u32()? as usize,
                        threshold: r.f32()?,
                    },
                    1 => CatfPredicate::NominalIn {
                        feat: r.u32()? as usize,
                        mask: r.u64()?,
                    },
                    2 => CatfPredicate::IsMissing {
                        feat: r.u32()? as usize,
                    },
                    _ => return Err(DopeError::Artifact("format2 invalid catf predicate".into())),
                };
                predicates.push(p);
            }
            let hermite_mode = r.u8()? != 0;
            let n_states = r.u32()? as usize;
            let leaf_values = r.f32_vec()?;
            trees.push(CatfTree {
                predicates,
                hermite_mode,
                leaf_values,
                n_states,
            });
        }
        conditionals.push(CatfConditional {
            col_idx,
            predecessors,
            copula_weights,
            copula_cond_sd,
            base_missing_prob,
            is_discrete,
            k_bins,
            z_edges,
            z_centers,
            trees,
        });
    }
    Ok(CatfModel {
        order,
        conditionals,
        marginals,
        is_nominal,
        nom_levels,
        rules,
    })
}
