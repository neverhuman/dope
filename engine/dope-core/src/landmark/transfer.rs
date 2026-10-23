//! A reference fitted once, scored repeatedly on untouched tune rows in one
//! frozen auditor representation. Per-row losses never implement Serialize.
use super::{cost, frame::Encoded, learners, ridge, CellState, Family, LandmarkOptions, PANEL};
use crate::{
    encode::{MlEncoder, Scaling},
    eval::{audit_predict, to_matrix, Matrix},
    rng::Rng,
    schema::ModelData,
    target::Head,
    DopeError, Program, Result, Samples,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize)]
pub struct TransferCell {
    pub id: String,
    pub family: Family,
    pub state: CellState,
    pub null_loss: f64,
    pub real_loss: f64,
    pub synthetic_loss: f64,
    pub retention: Option<f64>,
    pub stable: bool,
}
struct RowLosses {
    null: Vec<f64>,
    real: Vec<Vec<f64>>,
    synthetic: Vec<Vec<f64>>,
}
pub struct TransferSignature {
    pub cells: Vec<TransferCell>,
    pub mirror_retention: Option<f64>,
    pub panel_retention: Option<f64>,
    reference: [u8; 32],
    losses: RowLosses,
}
impl TransferSignature {
    pub fn private_report(&self) -> serde_json::Value {
        serde_json::json!({"panel":super::PANEL_V1,"privacy":"private","selection_only":true,
        "representation":"fixed-auditor-unit-v1","cells":self.cells,"mirror_retention":self.mirror_retention,"panel_retention":self.panel_retention})
    }
}

pub struct TransferReference {
    program: Program,
    query: Encoded,
    yq: Vec<f64>,
    real: Vec<Vec<f64>>,
    null: Vec<f64>,
    ids: Vec<(String, Family)>,
    reference: [u8; 32],
    opts: LandmarkOptions,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}
fn row_losses(head: Head, p: &[f64], y: &[f64]) -> Vec<f64> {
    let o = ridge::outputs(head);
    y.iter()
        .enumerate()
        .map(|(r, &t)| match head {
            Head::Regression => (p[r] - t).powi(2),
            _ => -p[r * o + t as usize].clamp(1e-7, 1.0 - 1e-7).ln(),
        })
        .collect()
}
fn encoded(program: &Program, x: Matrix) -> Encoded {
    let enc = MlEncoder::new(program, Scaling::Unit);
    let groups = enc.groups();
    let mut numeric = Vec::new();
    let mut missing = Vec::new();
    for j in 0..program.features() {
        let slots: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, &g)| (g == j).then_some(i))
            .collect();
        if matches!(
            program.schema.features[j].kind,
            crate::schema::Kind::Numeric(_)
        ) {
            numeric.push(slots[0]);
        }
        if program.missing[j] > 0.0 {
            missing.push(*slots.last().unwrap());
        }
    }
    Encoded {
        x,
        numeric,
        missing,
        groups,
    }
}

impl TransferReference {
    /// `program` must carry a schema and marginals fitted ONLY on fit rows.
    /// Fit and query share one data object so physical-role overlap is checked.
    /// Pass `broad=false` for the two roster-mirroring core learners only.
    pub fn new(
        program: &Program,
        data: &ModelData,
        fit_rows: &[usize],
        query_rows: &[usize],
        opts: &LandmarkOptions,
        broad: bool,
    ) -> Result<Self> {
        opts.validate()?;
        if fit_rows.len() < 20
            || query_rows.len() < 2
            || data.x.len() != program.features()
            || data.x.iter().any(|c| c.len() != data.y.len())
            || fit_rows
                .iter()
                .chain(query_rows)
                .any(|&r| r >= data.y.len() || !data.y[r].is_finite())
        {
            return Err(DopeError::Input("invalid transfer roles".into()));
        }
        let fit_set: BTreeSet<_> = fit_rows.iter().copied().collect();
        if fit_set.len() != fit_rows.len()
            || query_rows.iter().collect::<BTreeSet<_>>().len() != query_rows.len()
            || query_rows.iter().any(|r| fit_set.contains(r))
        {
            return Err(DopeError::Input(
                "transfer fit and query roles overlap".into(),
            ));
        }
        // Classification indices are validated before any native auditor can
        // index one-hot targets. Query-only classes retain the declared column.
        let head = program.head;
        let o = ridge::outputs(head);
        if head != Head::Regression
            && fit_rows
                .iter()
                .chain(query_rows)
                .any(|&r| data.y[r] < 0.0 || data.y[r].fract() != 0.0 || data.y[r] >= o as f64)
        {
            return Err(DopeError::Input("invalid transfer target codes".into()));
        }
        let enc = MlEncoder::new(program, Scaling::Unit);
        let cap = |rows: &[usize]| -> Vec<usize> {
            let mut r = rows.to_vec();
            r.sort_by_key(|&i| crate::rng::hash3(opts.seed, 0x5452414e, i as u64));
            r.truncate(opts.small_rows);
            r.sort_unstable();
            r
        };
        let (fit, query) = (cap(fit_rows), cap(query_rows));
        let (x, y) = to_matrix(&enc, data, &fit);
        let (q, yq) = to_matrix(&enc, data, &query);
        let null = row_losses(head, &ridge::null(head, &y, yq.len()), &yq);
        let mut real = Vec::new();
        let mut ids = Vec::new();
        for (name, id, family) in [
            ("linear", "L00a", Family::Additive),
            ("gbdt", "L00b", Family::Tree),
        ] {
            let p = learners::expanded(head, audit_predict(name, head, &x, &y, &q, opts.seed)?);
            real.push(row_losses(head, &p, &yq));
            ids.push((id.to_string(), family));
        }
        let (z, q) = (encoded(program, x), encoded(program, q));
        if broad {
            let mut spent = 0u64;
            for (i, recipe) in PANEL.iter().enumerate() {
                let price = cost::cost(i + 1, y.len(), z.x.cols, o, opts);
                if spent.saturating_add(price) > opts.work_budget {
                    continue;
                }
                if let Some((p, _)) = learners::run(
                    i + 1,
                    &z,
                    &y,
                    head,
                    &q,
                    opts,
                    super::job_seed(opts.seed, i + 1, 0),
                ) {
                    real.push(row_losses(head, &p, &yq));
                    ids.push((format!("L{:02}", i + 1), recipe.family));
                    spent += price;
                }
            }
        }
        let mut hash = Sha256::new();
        hash.update(serde_json::to_vec(&program.schema)?);
        for &r in fit_rows.iter().chain(query_rows) {
            hash.update((r as u64).to_le_bytes());
            hash.update(data.y[r].to_le_bytes());
            for c in &data.x {
                hash.update(c[r].to_le_bytes());
            }
        }
        Ok(Self {
            program: program.clone(),
            query: q,
            yq,
            real,
            null,
            ids,
            reference: hash.finalize().into(),
            opts: opts.clone(),
        })
    }

    /// Samples must already be encoded under the reference schema. The CLI
    /// remaps raw values and labels when scoring a different artifact schema.
    pub fn score(&self, samples: &Samples) -> Result<TransferSignature> {
        let data = ModelData {
            x: samples.x.clone(),
            y: samples.y.clone(),
        };
        if data.rows() < 20
            || data.x.len() != self.program.features()
            || data.x.iter().any(|c| c.len() != data.rows())
            || data.y.iter().any(|y| !y.is_finite())
        {
            return Err(DopeError::Input("invalid synthetic transfer frame".into()));
        }
        let head = self.program.head;
        let o = ridge::outputs(head);
        if head != Head::Regression
            && data
                .y
                .iter()
                .any(|y| *y < 0.0 || y.fract() != 0.0 || *y >= o as f64)
        {
            return Err(DopeError::Input("invalid synthetic target codes".into()));
        }
        let rows: Vec<usize> = (0..data.rows().min(self.opts.small_rows)).collect();
        let (x, y) = to_matrix(&MlEncoder::new(&self.program, Scaling::Unit), &data, &rows);
        let z = encoded(&self.program, x);
        let mut synth = Vec::new();
        for (id, _) in &self.ids {
            let p = match id.as_str() {
                "L00a" | "L00b" => learners::expanded(
                    head,
                    audit_predict(
                        if id == "L00a" { "linear" } else { "gbdt" },
                        head,
                        &z.x,
                        &y,
                        &self.query.x,
                        self.opts.seed,
                    )?,
                ),
                _ => {
                    let n: usize = id[1..]
                        .parse()
                        .map_err(|_| DopeError::Input("invalid transfer panel identity".into()))?;
                    learners::run(
                        n,
                        &z,
                        &y,
                        head,
                        &self.query,
                        &self.opts,
                        super::job_seed(self.opts.seed, n, 0),
                    )
                    .map(|v| v.0)
                    .unwrap_or_else(|| vec![f64::NAN; self.yq.len() * o])
                }
            };
            synth.push(row_losses(head, &p, &self.yq));
        }
        let losses = RowLosses {
            null: self.null.clone(),
            real: self.real.clone(),
            synthetic: synth,
        };
        let cells = aggregate_cells(&self.ids, &losses, None);
        let (mirror, panel) = aggregates(&cells);
        Ok(TransferSignature {
            cells,
            mirror_retention: mirror,
            panel_retention: panel,
            reference: self.reference,
            losses,
        })
    }

    pub fn compare(
        &self,
        a: &TransferSignature,
        b: &TransferSignature,
        draws: usize,
        seed: u64,
    ) -> Result<PairedComparison> {
        if a.reference != self.reference
            || b.reference != self.reference
            || a.cells.len() != b.cells.len()
            || draws == 0
            || draws > 10_000
        {
            return Err(DopeError::Input(
                "incompatible paired transfer comparison".into(),
            ));
        }
        let mut rng = Rng::new(seed);
        let mut mirror = Vec::new();
        let mut panel = Vec::new();
        for _ in 0..draws {
            let rows: Vec<usize> = (0..self.yq.len())
                .map(|_| rng.below(self.yq.len()))
                .collect();
            let aa = aggregates(&aggregate_cells(&self.ids, &a.losses, Some(&rows)));
            let bb = aggregates(&aggregate_cells(&self.ids, &b.losses, Some(&rows)));
            if let (Some(a), Some(b)) = (aa.0, bb.0) {
                mirror.push(b - a);
            }
            if let (Some(a), Some(b)) = (aa.1, bb.1) {
                panel.push(b - a);
            }
        }
        let interval = |mut v: Vec<f64>| -> Option<[f64; 2]> {
            if v.len() < draws * 9 / 10 {
                return None;
            }
            v.sort_by(f64::total_cmp);
            Some([
                crate::stats::quantile_sorted(&v, 0.025),
                crate::stats::quantile_sorted(&v, 0.975),
            ])
        };
        Ok(PairedComparison {
            draws,
            mirror_delta_95: interval(mirror),
            panel_delta_95: interval(panel),
        })
    }
}

fn aggregate_cells(
    ids: &[(String, Family)],
    losses: &RowLosses,
    rows: Option<&[usize]>,
) -> Vec<TransferCell> {
    let avg = |v: &[f64]| {
        rows.map_or_else(
            || mean(v),
            |r| r.iter().map(|&i| v[i]).sum::<f64>() / r.len() as f64,
        )
    };
    let null = avg(&losses.null);
    let max_gain = losses
        .real
        .iter()
        .map(|v| null - avg(v))
        .filter(|v| v.is_finite())
        .fold(0.0, f64::max);
    let core_max = losses
        .real
        .iter()
        .take(2)
        .map(|v| null - avg(v))
        .filter(|v| v.is_finite())
        .fold(0.0, f64::max);
    ids.iter()
        .enumerate()
        .map(|(i, (id, family))| {
            let real = avg(&losses.real[i]);
            let synthetic = avg(&losses.synthetic[i]);
            let denom = null - real;
            let threshold = if i < 2 { core_max } else { max_gain };
            let complete = denom >= (0.01 * null).max(0.25 * threshold) && denom > 1e-12;
            let influence: Vec<f64> = losses.real[i]
                .iter()
                .zip(&losses.null)
                .map(|(r, n)| r - (real / null.max(1e-12)) * n)
                .collect();
            let second: Vec<f64> = influence.iter().map(|v| v * v).collect();
            let count = rows.map_or(losses.null.len(), |r| r.len()).max(2);
            let se = ((avg(&second) - avg(&influence).powi(2)).max(0.0) / (count - 1) as f64)
                .sqrt()
                / null.max(1e-12);
            let gain = denom / null.max(1e-12);
            let stable = se <= 0.02f64.max(0.25 * gain.abs());
            let failed = ![null, real, synthetic].iter().all(|v| v.is_finite());
            TransferCell {
                id: id.clone(),
                family: *family,
                state: if failed {
                    CellState::Failed
                } else if complete {
                    CellState::Complete
                } else {
                    CellState::Uninformative
                },
                null_loss: null,
                real_loss: real,
                synthetic_loss: synthetic,
                retention: (!failed && complete).then_some((null - synthetic) / denom),
                stable,
            }
        })
        .collect()
}
fn aggregates(cells: &[TransferCell]) -> (Option<f64>, Option<f64>) {
    let mirror = cells
        .iter()
        .filter(|c| c.id == "L00a" || c.id == "L00b")
        .filter(|c| c.stable)
        .filter_map(|c| c.retention)
        .reduce(f64::min);
    let mut families: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for c in cells {
        if c.id == "L19" || !c.stable {
            continue;
        } // copying probe must never improve utility
        if let Some(r) = c.retention {
            let gain = c.null_loss - c.real_loss;
            let e = families.entry(format!("{:?}", c.family)).or_default();
            e.0 += gain * r.clamp(-1.0, 1.5);
            e.1 += gain;
        }
    }
    let panel = families
        .values()
        .filter(|(_, w)| *w > 0.0)
        .map(|(v, w)| v / w)
        .reduce(f64::min);
    (mirror, panel)
}
#[derive(Clone, Debug, Serialize)]
pub struct PairedComparison {
    pub draws: usize,
    pub mirror_delta_95: Option<[f64; 2]>,
    pub panel_delta_95: Option<[f64; 2]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn broad_diagnostics_never_change_mirror_and_knn1_never_enters_aggregate() {
        let ids = vec![
            ("L00a".into(), Family::Additive),
            ("L00b".into(), Family::Tree),
        ];
        let mut losses = RowLosses {
            null: vec![1.0; 40],
            real: vec![vec![0.8; 40], vec![0.7; 40]],
            synthetic: vec![vec![0.9; 40], vec![0.85; 40]],
        };
        let before = aggregates(&aggregate_cells(&ids, &losses, None));
        let mut more = ids.clone();
        more.push(("L19".into(), Family::Local));
        losses.real.push(vec![0.0; 40]);
        losses.synthetic.push(vec![0.0; 40]);
        let after = aggregates(&aggregate_cells(&more, &losses, None));
        assert_eq!(before, after);
        more.push(("L13".into(), Family::Kernel));
        losses.real.push(vec![0.0; 40]);
        losses.synthetic.push(vec![2.0; 40]);
        let worse = aggregates(&aggregate_cells(&more, &losses, None));
        assert_eq!(before.0, worse.0);
        assert!(worse.1.unwrap() <= before.1.unwrap());
    }
}
