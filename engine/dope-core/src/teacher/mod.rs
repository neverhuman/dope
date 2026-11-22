//! Whole-stack `K=5` nested cross-fitted teacher (`engine/dope-core/src/teacher/`).
//!
//! Fits a diverse retained native teacher panel (`Linear`, `GBDT-d4`, `GBDT-d6`, `16-NN`)
//! strictly on complement folds `C = BUILD \ k` under Rule A-1 duplicate grouping, stacks
//! members via inner OOF simplex/NNLS weights, calibrates probabilities (PAV / temp+bias),
//! fits honest OOF regression residual quantiles, and scores arbitrary `Q_X` probe pools with
//! member-agreement confidence weights (`T01`, `T03`, `T04`).

pub mod exchange;
pub mod noise;
pub mod panel;
pub mod stack;

use crate::error::Result;
use crate::target::Head;
use exchange::TeacherExchangeBundle;
use noise::TeacherResidualModel;
use panel::{duplicate_grouped_kfolds, GbdtMember, KnnMember, LinearMember};
use stack::{fit_simplex_weights, MulticlassCalibrator, PlattCalibrator};

#[derive(Clone, Debug)]
pub struct TeacherConfig {
    pub folds: usize,
    pub max_trees_d4: usize,
    pub max_trees_d6: usize,
    pub seed: u64,
}

impl Default for TeacherConfig {
    fn default() -> Self {
        Self {
            folds: 5,
            max_trees_d4: 120,
            max_trees_d6: 120,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug)]
struct FoldTeacher {
    linear: LinearMember,
    gbdt4: GbdtMember,
    gbdt6: GbdtMember,
    knn: KnnMember,
    weights: Vec<f64>,
    bin_cal: Option<PlattCalibrator>,
    mc_cal: Option<MulticlassCalibrator>,
}

impl FoldTeacher {
    fn predict_members(&self, x: &[Vec<f64>], rows: &[usize]) -> [Vec<f64>; 4] {
        [
            self.linear.predict_rows(x, rows),
            self.gbdt4.predict_rows(x, rows),
            self.gbdt6.predict_rows(x, rows),
            self.knn.predict_rows(x, rows),
        ]
    }

    fn combine_and_calibrate(&self, head: Head, members: &[Vec<f64>; 4], n: usize) -> Vec<f64> {
        let o = head.outputs();
        let mut stacked = vec![0.0; n * o];
        for (m, pred) in members.iter().enumerate() {
            let wm = self.weights[m];
            for (dst, &src) in stacked.iter_mut().zip(pred) {
                *dst += wm * src;
            }
        }
        match head {
            Head::Regression => stacked,
            Head::Binary => match &self.bin_cal {
                Some(cal) => stacked.into_iter().map(|p| cal.transform(p)).collect(),
                None => stacked
                    .into_iter()
                    .map(|p| p.clamp(1e-5, 1.0 - 1e-5))
                    .collect(),
            },
            Head::Multiclass(k) => match &self.mc_cal {
                Some(cal) => {
                    let mut out = Vec::with_capacity(n * k);
                    for r in 0..n {
                        out.extend(cal.transform_row(&stacked[r * k..(r + 1) * k]));
                    }
                    out
                }
                None => stacked,
            },
        }
    }
}

/// Retained `K=5` nested cross-fitted teacher ensemble callable on arbitrary `Q_X` probes.
#[derive(Clone, Debug)]
pub struct CrossFittedTeacher {
    pub head: Head,
    pub d: usize,
    pub oof_predictions: Vec<f64>,
    pub oof_agreement: Vec<f64>,
    pub residual_model: Option<TeacherResidualModel>,
    pub seen_teacher_families: Vec<String>,
    pub oof_beats_null: bool,
    pub external_probe_predictions: Option<Vec<f64>>,
    pub external_probe_agreement: Option<Vec<f64>>,
    fold_models: Vec<FoldTeacher>,
    y_scale: f64,
}

impl CrossFittedTeacher {
    /// Fit a `K`-fold nested cross-fitted teacher strictly on `BUILD` (`x`, `y`).
    pub fn fit(head: Head, x: &[Vec<f64>], y: &[f64], cfg: &TeacherConfig) -> Self {
        let n = y.len();
        let d = x.len();
        let o = head.outputs();
        let folds = duplicate_grouped_kfolds(x, y, cfg.folds, cfg.seed);
        let y_mean = y.iter().sum::<f64>() / n.max(1) as f64;
        let y_scale = (y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / n.max(1) as f64)
            .sqrt()
            .max(1e-4);

        let mut oof_predictions = vec![0.0; n * o];
        let mut oof_agreement = vec![1.0; n];
        let mut fold_models = Vec::with_capacity(folds.len());

        for (k_idx, val_rows) in folds.iter().enumerate() {
            if val_rows.is_empty() {
                continue;
            }
            let comp_rows: Vec<usize> = folds
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != k_idx)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();
            let fold_seed = cfg.seed ^ ((k_idx as u64 + 1).wrapping_mul(0x9e37_79b9));

            // Nested inner 3-fold OOF inside `comp_rows` for unbiased stacking + calibration (`T01`).
            let comp_x: Vec<Vec<f64>> = (0..d)
                .map(|j| comp_rows.iter().map(|&r| x[j][r]).collect())
                .collect();
            let comp_y: Vec<f64> = comp_rows.iter().map(|&r| y[r]).collect();
            let inner_folds = duplicate_grouped_kfolds(&comp_x, &comp_y, 3, fold_seed ^ 0x111);
            let m_comp = comp_rows.len();
            let mut inner_oof = vec![vec![0.0; m_comp * o]; 4];
            for (in_k, in_val_loc) in inner_folds.iter().enumerate() {
                if in_val_loc.is_empty() {
                    continue;
                }
                let in_tr: Vec<usize> = inner_folds
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != in_k)
                    .flat_map(|(_, f)| f.iter().map(|&loc| comp_rows[loc]))
                    .collect();
                let in_va: Vec<usize> = in_val_loc.iter().map(|&loc| comp_rows[loc]).collect();
                let s_in = fold_seed ^ ((in_k as u64 + 1) * 0x517c);
                let in_t4 = if m_comp < 300 {
                    cfg.max_trees_d4
                } else {
                    (cfg.max_trees_d4 / 2).max(24)
                };
                let in_t6 = if m_comp < 300 {
                    cfg.max_trees_d6
                } else {
                    (cfg.max_trees_d6 / 2).max(24)
                };
                let preds = [
                    LinearMember::fit(head, x, y, &in_tr).predict_rows(x, &in_va),
                    GbdtMember::fit(head, x, y, &in_tr, 4, in_t4, s_in).predict_rows(x, &in_va),
                    GbdtMember::fit(head, x, y, &in_tr, 6, in_t6, s_in ^ 7).predict_rows(x, &in_va),
                    KnnMember::fit(head, x, y, &in_tr, s_in ^ 13).predict_rows(x, &in_va),
                ];
                for m in 0..4 {
                    for (pos, &loc) in in_val_loc.iter().enumerate() {
                        for c in 0..o {
                            inner_oof[m][loc * o + c] = preds[m][pos * o + c];
                        }
                    }
                }
            }
            let mut weights = fit_simplex_weights(head, &inner_oof, &comp_y);
            let mut stacked_inner = vec![0.0; m_comp * o];
            for m in 0..4 {
                for i in 0..m_comp * o {
                    stacked_inner[i] += weights[m] * inner_oof[m][i];
                }
            }
            // Teacher quality gate: if stacked inner OOF is worse than best single member, fall back.
            let stack_loss = teacher_oof_loss(head, &stacked_inner, &comp_y);
            let mut best_m = None;
            let mut best_l = stack_loss;
            for (m, pred) in inner_oof.iter().enumerate() {
                let ml = teacher_oof_loss(head, pred, &comp_y);
                if ml + 1e-6 < best_l {
                    best_l = ml;
                    best_m = Some(m);
                }
            }
            if let Some(bm) = best_m {
                weights = vec![0.0; 4];
                weights[bm] = 1.0;
                stacked_inner = inner_oof[bm].clone();
            }
            let bin_cal = if head == Head::Binary {
                Some(PlattCalibrator::fit(&stacked_inner, &comp_y))
            } else {
                None
            };
            let mc_cal = match head {
                Head::Multiclass(k) => Some(MulticlassCalibrator::fit(&stacked_inner, &comp_y, k)),
                _ => None,
            };

            // Fit full-complement retained members on `comp_rows`.
            let ft = FoldTeacher {
                linear: LinearMember::fit(head, x, y, &comp_rows),
                gbdt4: GbdtMember::fit(head, x, y, &comp_rows, 4, cfg.max_trees_d4, fold_seed),
                gbdt6: GbdtMember::fit(
                    head,
                    x,
                    y,
                    &comp_rows,
                    6,
                    cfg.max_trees_d6,
                    fold_seed ^ 0x22,
                ),
                knn: KnnMember::fit(head, x, y, &comp_rows, fold_seed ^ 0x33),
                weights,
                bin_cal,
                mc_cal,
            };
            let val_members = ft.predict_members(x, val_rows);
            let val_pred = ft.combine_and_calibrate(head, &val_members, val_rows.len());
            let val_agr = member_agreement(head, &val_members, val_rows.len(), y_scale);
            for (pos, &r) in val_rows.iter().enumerate() {
                oof_agreement[r] = val_agr[pos];
                for c in 0..o {
                    oof_predictions[r * o + c] = val_pred[pos * o + c];
                }
            }
            fold_models.push(ft);
        }

        let residual_model = if head == Head::Regression {
            Some(TeacherResidualModel::fit_oof(y, &oof_predictions))
        } else {
            None
        };

        let null_pred = null_baseline_pred(head, y);
        let oof_beats_null = teacher_oof_loss(head, &oof_predictions, y) + 1e-6
            < teacher_oof_loss(head, &null_pred, y);

        Self {
            head,
            d,
            oof_predictions,
            oof_agreement,
            residual_model,
            seen_teacher_families: vec![
                "native_linear".into(),
                "native_gbdt_d4".into(),
                "native_gbdt_d6".into(),
                "native_knn16".into(),
            ],
            oof_beats_null,
            external_probe_predictions: None,
            external_probe_agreement: None,
            fold_models,
            y_scale,
        }
    }

    /// Construct a teacher directly from a validated external `TeacherExchangeBundle`
    /// (`--teacher external:DIR`), propagating `seen_teacher_families` for auditor independence.
    pub fn from_exchange(
        head: Head,
        d: usize,
        y_build: &[f64],
        bundle: &TeacherExchangeBundle,
    ) -> Result<Self> {
        bundle.validate(y_build.len(), head.outputs())?;
        let n = y_build.len();
        let y_mean = y_build.iter().sum::<f64>() / n.max(1) as f64;
        let y_scale = (y_build.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>()
            / n.max(1) as f64)
            .sqrt()
            .max(1e-4);
        let residual_model = if head == Head::Regression {
            Some(TeacherResidualModel::fit_oof(
                y_build,
                &bundle.oof_predictions,
            ))
        } else {
            None
        };
        let null_pred = null_baseline_pred(head, y_build);
        let oof_beats_null = teacher_oof_loss(head, &bundle.oof_predictions, y_build) + 1e-6
            < teacher_oof_loss(head, &null_pred, y_build);
        Ok(Self {
            head,
            d,
            oof_predictions: bundle.oof_predictions.clone(),
            oof_agreement: vec![1.0; n],
            residual_model,
            seen_teacher_families: bundle.seen_teacher_families.clone(),
            oof_beats_null,
            external_probe_predictions: bundle.probe_predictions.clone(),
            external_probe_agreement: bundle.probe_agreement.clone(),
            fold_models: Vec::new(),
            y_scale,
        })
    }

    /// Score arbitrary synthetic probes `x_probe` by averaging across all `K` retained fold
    /// teachers (or returning precomputed external exchange probe predictions).
    pub fn predict_probes(&self, x_probe: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
        let n = x_probe.first().map_or(0, Vec::len);
        let o = self.head.outputs();
        if let Some(ref ext) = self.external_probe_predictions {
            if ext.len() == n * o {
                let agr = self
                    .external_probe_agreement
                    .clone()
                    .filter(|a| a.len() == n)
                    .unwrap_or_else(|| vec![1.0; n]);
                return (ext.clone(), agr);
            }
        }
        if n == 0 || self.fold_models.is_empty() {
            return (vec![0.0; n * o], vec![1.0; n]);
        }
        let rows: Vec<usize> = (0..n).collect();
        let k_f = self.fold_models.len() as f64;
        let mut avg = vec![0.0; n * o];
        let mut agr = vec![0.0; n];
        for ft in &self.fold_models {
            let members = ft.predict_members(x_probe, &rows);
            let pred = ft.combine_and_calibrate(self.head, &members, n);
            let a = member_agreement(self.head, &members, n, self.y_scale);
            for i in 0..n * o {
                avg[i] += pred[i] / k_f;
            }
            for i in 0..n {
                agr[i] += a[i] / k_f;
            }
        }
        (avg, agr)
    }
}

fn null_baseline_pred(head: Head, y: &[f64]) -> Vec<f64> {
    let n = y.len().max(1);
    match head {
        Head::Regression => {
            let m = y.iter().sum::<f64>() / n as f64;
            vec![m; y.len()]
        }
        Head::Binary => {
            let p = (y.iter().sum::<f64>() / n as f64).clamp(1e-4, 1.0 - 1e-4);
            vec![p; y.len()]
        }
        Head::Multiclass(k) => {
            let mut cnt = vec![1.0; k];
            for &v in y {
                let c = (v.round() as usize).min(k - 1);
                cnt[c] += 1.0;
            }
            let tot: f64 = cnt.iter().sum();
            let probs: Vec<f64> = cnt.into_iter().map(|c| c / tot).collect();
            let mut out = Vec::with_capacity(y.len() * k);
            for _ in 0..y.len() {
                out.extend_from_slice(&probs);
            }
            out
        }
    }
}

fn teacher_oof_loss(head: Head, pred: &[f64], y: &[f64]) -> f64 {
    let n = y.len().max(1);
    match head {
        Head::Regression => {
            y.iter()
                .zip(pred)
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                / n as f64
        }
        Head::Binary => {
            -y.iter()
                .zip(pred)
                .map(|(&yi, &pi)| {
                    let p = pi.clamp(1e-6, 1.0 - 1e-6);
                    yi * p.ln() + (1.0 - yi) * (1.0 - p).ln()
                })
                .sum::<f64>()
                / n as f64
        }
        Head::Multiclass(k) => {
            -(0..y.len())
                .map(|i| {
                    let c = (y[i].round() as usize).min(k - 1);
                    pred[i * k + c].clamp(1e-6, 1.0).ln()
                })
                .sum::<f64>()
                / n as f64
        }
    }
}

/// Per-row teacher member agreement in `[0.2, 1.0]` based on cross-member dispersion.
fn member_agreement(head: Head, members: &[Vec<f64>; 4], n: usize, y_scale: f64) -> Vec<f64> {
    let o = head.outputs();
    let scale = match head {
        Head::Regression => y_scale.max(1e-4),
        Head::Binary | Head::Multiclass(_) => 0.35,
    };
    (0..n)
        .map(|r| {
            let mut disp = 0.0;
            for c in 0..o {
                let m = (0..4).map(|j| members[j][r * o + c]).sum::<f64>() * 0.25;
                let v = (0..4)
                    .map(|j| (members[j][r * o + c] - m).powi(2))
                    .sum::<f64>()
                    * 0.25;
                disp += v.sqrt();
            }
            let norm_disp = (disp / o as f64) / scale;
            (1.0 / (1.0 + norm_disp)).clamp(0.2, 1.0)
        })
        .collect()
}
