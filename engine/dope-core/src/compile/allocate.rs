//! Target-first allocation of the byte budget for one dependence share.

use super::ctx::{Ctx, NetEntry};
use super::PLANS;
use crate::copula::Dependence;
use crate::error::Result;
use crate::program::Program;
use crate::regimes::Regimes;
use crate::target::null_loss;

/// Rounds of re-growing the forest into the measured (compressed) slack.
const GROW_ROUNDS: usize = 3;

/// One complete program for a given dependence share.
pub(super) struct Candidate {
    pub(super) k: usize,
    pub(super) s: usize,
    pub(super) h: usize,
    pub(super) entry: NetEntry,
    pub(super) bytes: Vec<u8>,
    pub(super) program: Program,
}

/// Target-first allocation for one dependence share: reserve the most likely
/// dependence model costing at most `share` of the budget, size the network
/// (and its trees) on the rest, then spend what is left on dependence.
pub(super) fn allocate(
    ctx: &mut Ctx,
    deps: &[(f64, (Regimes, Dependence))],
    r_fit: &[f64],
    budget: usize,
    share: f64,
) -> Result<Option<Candidate>> {
    let (head, d, opts) = (ctx.head, ctx.d(), ctx.opts);
    let empty = ctx.empty_net();
    let reserve = {
        let margs = ctx.marginals(9);
        let indep = (Regimes::single(d), Dependence::Independent);
        let floor = ctx.assemble(9, margs.clone(), 0, &empty, &indep)?.0.len();
        let cap = (share * budget as f64) as usize;
        let mut pick = None;
        for (_, dep) in deps {
            let extra = ctx
                .assemble(9, margs.clone(), 0, &empty, dep)?
                .0
                .len()
                .saturating_sub(floor);
            if extra <= cap {
                pick = Some(dep.clone());
                break;
            }
        }
        match (pick, &opts.dependence, d >= 2) {
            (Some(p), _, _) => p,
            (None, None, true) => (Regimes::single(d), Dependence::fit_factor(r_fit, d, 1)),
            _ => deps[deps.len() - 1].1.clone(),
        }
    };
    let plans: Vec<(usize, usize, usize)> = opts.plan.map_or_else(|| PLANS.to_vec(), |p| vec![p]);
    let forced = opts.plan.is_some();
    let null_val = null_loss(head, &ctx.y_val);
    // Candidates span the ladder: the richest network that fits, then every
    // second plan below it (leaner networks leave more bytes for trees).
    let mut fitting: Vec<(usize, usize, usize, NetEntry)> = Vec::new();
    let mut next = 0;
    for (idx, &(k, s, h)) in plans.iter().enumerate() {
        if idx < next {
            continue;
        }
        let (k, s) = (k.max(3) | 1, s.min(d));
        let margs = ctx.marginals(k);
        let (floor, _) = ctx.assemble(k, margs.clone(), 0, &empty, &reserve)?;
        let o = head.outputs();
        let estimate =
            floor.len() + (6 * (h * s + h + o * s + o * h)).div_ceil(8) + 4 * (4 + o) + 96;
        if !forced && estimate > budget {
            continue;
        }
        ctx.net(s, h)?;
        let entry = ctx.nets[&(s, h)].clone();
        let (bytes, _) = ctx.assemble(k, margs.clone(), s, &entry, &reserve)?;
        if forced || bytes.len() <= budget {
            ctx.boost(s, h);
            // The remaining bytes buy trees. The per-tree estimate ignores the
            // artifact's compression, so scale the room by what the trees
            // measurably cost and grow again; then trim if it overshoots.
            let base = bytes.len();
            let mut room = budget.saturating_sub(base);
            let mut grown = ctx.with_forest(s, h, &entry, room);
            for _ in 0..GROW_ROUNDS {
                let trees = grown.forest.trees();
                let used = ctx.assemble(k, margs.clone(), s, &grown, &reserve)?.0.len();
                if trees == 0 || used >= budget || used <= base {
                    break;
                }
                let scaled = (room as f64 * (budget - base) as f64 / (used - base) as f64) as usize;
                let next = ctx.with_forest(s, h, &entry, scaled);
                if next.forest.trees() <= trees {
                    break;
                }
                grown = next;
                room = scaled;
            }
            while grown.forest.trees() > 0
                && ctx.assemble(k, margs.clone(), s, &grown, &reserve)?.0.len() > budget
            {
                let keep = grown
                    .forest
                    .trees()
                    .saturating_sub(1 + grown.forest.trees() / 20);
                grown = ctx.retrim(s, &grown, keep);
            }
            fitting.push((k, s, h, grown));
            next = idx + 2;
            if fitting.len() == 3 {
                break;
            }
        }
    }
    // Prefer the leanest plan unless a richer one is meaningfully better.
    let mut pick: Option<usize> = None;
    for i in (0..fitting.len()).rev() {
        let better = match pick {
            None => true,
            Some(b) => {
                let best = fitting[b].3.val_loss;
                fitting[i].3.val_loss < best - 2e-3 * (null_val - best).max(0.0)
            }
        };
        if better {
            pick = Some(i);
        }
    }
    let Some(pick) = pick else {
        return Ok(None);
    };
    let (k, s, h, entry) = fitting.swap_remove(pick);
    let margs = ctx.marginals(k);
    let mut chosen = None;
    for (_, dep) in deps {
        let (bytes, program) = ctx.assemble(k, margs.clone(), s, &entry, dep)?;
        if forced || bytes.len() <= budget {
            chosen = Some((bytes, program));
            break;
        }
    }
    let (bytes, program) = match chosen {
        Some(c) => c,
        None => ctx.assemble(
            k,
            margs,
            s,
            &entry,
            &(Regimes::single(d), Dependence::Independent),
        )?,
    };
    Ok(Some(Candidate {
        k,
        s,
        h,
        entry,
        bytes,
        program,
    }))
}
