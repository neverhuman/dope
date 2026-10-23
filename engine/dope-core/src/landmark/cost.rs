use super::{LandmarkFrame, LandmarkOptions};
use crate::{DopeError, Result};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct WorkReport {
    pub budget: u64,
    pub estimated_units: u64,
    pub cells_run: usize,
    pub spectrum_run: usize,
    pub dropped: Vec<u8>,
}

pub(crate) fn cost(id: usize, n: usize, d: usize, classes: usize, opts: &LandmarkOptions) -> u64 {
    let n = n.min(if id <= 15 || id == 22 {
        opts.ridge_rows
    } else {
        opts.small_rows
    }) as u64;
    let d = d as u64;
    let o = classes.max(1) as u64;
    let p = match id {
        2 | 3 | 6 => d * 2,
        4 | 5 => d * 3,
        7 => d * 8,
        8 => d * 16,
        9 => d * 7,
        10 => d * 4,
        11 => d + 7,
        12 => d + 28,
        13 => 256,
        15 => 17,
        _ => d,
    }
    .min(511)
        + 1;
    let units = match id {
        1..=15 | 22 => n
            .saturating_mul(p * p + 2 * p * o)
            .saturating_mul(if id == 15 { d } else { 1 })
            .saturating_add(p * p * p * 3),
        16 | 17 => n.saturating_mul(d).saturating_mul(256).saturating_mul(3),
        18 | 23 => n.saturating_mul(d).saturating_mul(64 * 32 * 4),
        19 | 20 => n
            .saturating_mul(opts.knn_reference as u64)
            .saturating_mul(d.min(32)),
        21 => n
            .min(2048)
            .saturating_mul(d.min(32) + o)
            .saturating_mul(16 * 300 * 4),
        _ => 0,
    };
    units.saturating_mul(opts.folds as u64).saturating_mul(o)
}

pub(crate) fn plan(
    frame: &LandmarkFrame,
    opts: &LandmarkOptions,
) -> Result<([bool; 23], WorkReport)> {
    let d = frame
        .categories
        .iter()
        .zip(&frame.columns)
        .map(|(c, values)| {
            c.map_or(1, |k| k + 1) + usize::from(values.iter().any(|v| !v.is_finite()))
        })
        .sum::<usize>();
    if d > 512 {
        return Err(DopeError::Budget(
            "landmark representation exceeds 512 columns".into(),
        ));
    }
    let classes = super::ridge::outputs(frame.head);
    let costs: [u64; 23] = std::array::from_fn(|i| cost(i + 1, frame.rows(), d, classes, opts));
    if costs[0] > opts.work_budget {
        return Err(DopeError::Budget(
            "landmark budget cannot run linear baseline".into(),
        ));
    }
    let mut enabled = [true; 23];
    let mut units = costs.iter().fold(0u64, |a, b| a.saturating_add(*b));
    let mut dropped = Vec::new();
    for id in [
        23, 22, 21, 10, 4, 3, 6, 5, 2, 9, 11, 13, 7, 14, 12, 15, 20, 19, 18, 17, 16, 8,
    ] {
        if units <= opts.work_budget {
            break;
        }
        enabled[id - 1] = false;
        units = units.saturating_sub(costs[id - 1]);
        dropped.push(id as u8);
    }
    Ok((
        enabled,
        WorkReport {
            budget: opts.work_budget,
            estimated_units: units,
            cells_run: enabled.iter().filter(|&&v| v).count(),
            spectrum_run: 0,
            dropped,
        },
    ))
}
