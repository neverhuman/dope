//! Authoritative fitted/encoded/decoded basis order, without hidden gate defaults.
use super::{Result, WireError};
use serde::{Deserialize, Serialize};
/// Every active basis gate is serialized. Noise lives explicitly in TargetModel.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateValues {
    pub linear: f64,
    pub asinh: f64,
    pub periodic: f64,
    pub hinge: f64,
    pub pair: f64,
    pub row_mean: f64,
}
/// Stable registry pairs: wire field, mathematical coordinate family.
pub const REGISTRY: [(&str, &str); 6] = [
    ("linear", "g*x[j]"),
    ("asinh", "g*asinh(x[j])"),
    ("periodic", "g*sin(2*pi*x[j])"),
    ("hinge", "g*max(x[j]-0.5,0)"),
    ("pair", "g*x[j]*x[j+1]"),
    ("row_mean", "g*mean(x)"),
];
impl GateValues {
    pub fn validate(&self) -> Result<()> {
        if [
            self.linear,
            self.asinh,
            self.periodic,
            self.hinge,
            self.pair,
            self.row_mean,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0 || *v > 4.0)
        {
            Err(WireError::new("invalid_gate"))
        } else {
            Ok(())
        }
    }
}
/// Four per-column blocks, adjacent pairs, then one row mean: exactly 5*p.
pub fn width(columns: usize) -> Result<usize> {
    if columns == 0 || columns > 256 {
        Err(WireError::new("basis_shape_limit"))
    } else {
        Ok(5 * columns)
    }
}
/// Fit and decoded sampling must call this same named registry with all gates.
pub fn features(row: &[f64], g: &GateValues) -> Result<Vec<f64>> {
    g.validate()?;
    let size = width(row.len())?;
    if row.iter().any(|v| !v.is_finite()) {
        return Err(WireError::new("nonfinite_basis_input"));
    }
    let mut out = Vec::with_capacity(size);
    out.extend(row.iter().map(|v| g.linear * v));
    out.extend(row.iter().map(|v| g.asinh * v.asinh()));
    out.extend(
        row.iter()
            .map(|v| g.periodic * (std::f64::consts::TAU * v).sin()),
    );
    out.extend(row.iter().map(|v| g.hinge * (v - 0.5).max(0.0)));
    out.extend(row.windows(2).map(|v| g.pair * v[0] * v[1]));
    out.push(g.row_mean * (row.iter().sum::<f64>() / row.len() as f64));
    if out.len() != size || out.iter().any(|v| !v.is_finite()) {
        Err(WireError::new("nonfinite_basis_output"))
    } else {
        Ok(out)
    }
}

/// Machine-readable registry for training/reference consumers.
pub const REGISTRY_JSON: &str = include_str!("basis_registry.json");
