use super::{Family, LandmarkId, LandmarkOptions, WorkReport, PANEL_V1};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignatureKind {
    Real,
    Synthetic,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CellState {
    Complete,
    Uninformative,
    Unsupported,
    OverBudget,
    Failed,
    /// Additive: the caller did not request this landmark (`signature_requested`).
    /// Readers must treat it as missing, never as a measured zero.
    NotRequested,
}
#[derive(Clone, Debug, Serialize)]
pub struct LandmarkCell {
    pub id: LandmarkId,
    pub name: &'static str,
    pub family: Family,
    pub state: CellState,
    pub gain: Option<f64>,
    pub se: Option<f64>,
    pub stable: bool,
    pub lambda_indices: Vec<u8>,
    pub train_rows: Vec<usize>,
    pub query_rows: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct FeatureSpectrum {
    pub slot: usize,
    pub values: [Option<f64>; 9],
}

/// Intentionally does NOT implement Serialize or Debug. Choose a private view
/// explicitly, or an admitted global export. This prevents accidental sidecars.
#[derive(Clone)]
pub struct Signature {
    pub panel: String,
    pub kind: SignatureKind,
    pub task: &'static str,
    pub classes: usize,
    pub cells: Vec<LandmarkCell>,
    pub target_stats: [Option<f64>; 3],
    pub contrasts: Vec<Option<f64>>,
    pub spectrum: Vec<FeatureSpectrum>,
    pub work: WorkReport,
    pub options: LandmarkOptions,
    pub rows: usize,
    pub features: usize,
}
impl Signature {
    /// Explicit private report. Never attach this to a level export.
    pub fn private_report(&self) -> serde_json::Value {
        serde_json::json!({"panel":self.panel,"kind":self.kind,"task":self.task,"classes":self.classes,
            "cells":self.cells,"target_stats":self.target_stats,"contrasts":self.contrasts,
            "spectrum":self.spectrum,"work":self.work,"options":self.options,
            "rows":self.rows,"features":self.features,"privacy":if self.kind==SignatureKind::Real{"private"}else{"artifact_derived"}})
    }
    /// Caller explicitly admits disclosure of real-data global coordinates.
    /// Feature spectra need a separate policy; they are never included here.
    pub fn global_coordinates(&self) -> GlobalCoordinates {
        let mut v = vec![None; 96];
        let gains: Vec<(usize, f64)> = self
            .cells
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                (c.state == CellState::Complete && c.stable)
                    .then_some(c.gain)
                    .flatten()
                    .map(|g| (i, g))
            })
            .collect();
        for &(i, g) in &gains {
            v[i] = Some(g);
            v[23 + i] = self.cells[i].se;
            let less = gains.iter().filter(|(_, h)| *h < g).count() as f64;
            let equal = gains.iter().filter(|(_, h)| *h == g).count() as f64;
            v[46 + i] = Some((less + 0.5 * (equal - 1.0)) / (gains.len() - 1).max(1) as f64);
        }
        v[69..87].copy_from_slice(&self.contrasts);
        v[87..90].copy_from_slice(&self.target_stats);
        for i in 0..3 {
            v[90 + i] = Some(
                (i == if self.classes == 0 {
                    0
                } else if self.classes == 2 {
                    1
                } else {
                    2
                }) as u8 as f64,
            );
        }
        v[93] = Some((self.classes.max(1) as f64).log2() / 6.0);
        v[94] = Some((self.rows.max(1) as f64).log10() / 6.0);
        v[95] = Some((self.features.max(1) as f64).log2() / 12.0);
        GlobalCoordinates {
            panel: PANEL_V1.into(),
            options: self.options.clone(),
            kind: self.kind,
            mask: v.iter().map(Option::is_some).collect(),
            values: v.into_iter().map(|v| v.unwrap_or(0.0) as f32).collect(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalCoordinates {
    pub panel: String,
    pub options: LandmarkOptions,
    pub kind: SignatureKind,
    pub values: Vec<f32>,
    pub mask: Vec<bool>,
}

pub(crate) fn contrasts(cells: &[LandmarkCell], target: [Option<f64>; 3]) -> Vec<Option<f64>> {
    let g = |id: usize| {
        cells
            .get(id - 1)
            .filter(|c| c.stable && c.state == CellState::Complete)
            .and_then(|c| c.gain)
    };
    let max = |ids: &[usize]| -> Option<f64> { ids.iter().filter_map(|&i| g(i)).reduce(f64::max) };
    let s = max(&(1..=23).collect::<Vec<_>>());
    let Some(s) = s.filter(|s| *s >= 0.01) else {
        return vec![None; 18];
    };
    let a = max(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]);
    let sub = |a: Option<f64>, b: Option<f64>| a.zip(b).map(|(a, b)| (a - b) / s);
    let mut out = vec![
        Some(s),
        g(1).map(|v| v / s),
        a.map(|v| v / s),
        sub(max(&[18, 13, 21]), a),
        sub(g(12), g(1)),
        sub(g(8), g(7)),
        sub(g(9), max(&[1, 5])),
        sub(max(&[2, 3, 5]), g(1)),
        sub(max(&[6, 4, 10]), g(1)),
        g(19).zip(g(20)).map(|(a, b)| a / b.max(1e-12)),
        sub(g(13), g(18)),
        g(15).map(|v| v / s),
        sub(g(11), g(1)),
        g(14).map(|v| v / s),
        sub(g(1), g(22)),
        sub(g(18), g(23)),
        sub(g(21), max(&[18, 13])),
        target[2],
    ];
    for v in out.iter_mut().flatten() {
        *v = v.clamp(-1.0, 1.0);
    }
    out
}

impl GlobalCoordinates {
    /// Refuse another panel, recipe, width, nonfinite or malformed mask.
    pub fn decode(bytes: &[u8], expected: &LandmarkOptions) -> crate::Result<Self> {
        expected.validate()?;
        let v: Self = serde_json::from_slice(bytes)?;
        if v.panel != PANEL_V1
            || v.options != *expected
            || v.values.len() != 96
            || v.mask.len() != 96
            || v.values
                .iter()
                .zip(&v.mask)
                .any(|(x, m)| !x.is_finite() || (!*m && *x != 0.0))
        {
            return Err(crate::DopeError::Artifact(
                "incompatible landmark coordinates".into(),
            ));
        }
        Ok(v)
    }
}
