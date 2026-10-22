//! Validate the complete decoded shape before allocating model arrays.
use super::Parts;
use crate::error::{DopeError, Result};
use crate::noise::RESIDUAL_KNOTS;
use serde_json::Value;

pub(super) const MAX_METADATA: usize = 1 << 20;
pub(super) const MAX_SECTIONS: usize = 64;
const MAX_FEATURES: usize = 4096;
const MAX_LEVELS: usize = 65536;
const MAX_MATERIALIZED: usize = 64 << 20;

fn bad() -> DopeError {
    DopeError::Artifact("invalid artifact shape or decode footprint".into())
}
fn require(ok: bool) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(bad())
    }
}
pub(super) fn integer(v: &Value) -> Result<usize> {
    v.as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(bad)
}
pub(super) fn integers(v: &Value) -> Result<Vec<usize>> {
    v.as_array().ok_or_else(bad)?.iter().map(integer).collect()
}
pub(super) fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(bad)
}
pub(super) fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(bad)
}
pub(super) fn packed(n: usize, bits: usize) -> Result<usize> {
    require((2..=8).contains(&bits))?;
    Ok(add(mul(n, bits)?, 7)? / 8)
}
struct Budget(usize);
impl Budget {
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.0 = add(self.0, bytes)?;
        require(self.0 <= MAX_MATERIALIZED)
    }
    // Covers simultaneously live f64 arrays, i32/u32 codes and unpacking buffers.
    fn elements(&mut self, n: usize) -> Result<()> {
        self.charge(mul(n, 32)?)
    }
}
fn section<'a>(p: &'a Parts, name: &str) -> &'a [u8] {
    p.sections.get(name).map(Vec::as_slice).unwrap_or(&[])
}
fn exact(p: &Parts, name: &str, n: usize) -> Result<()> {
    require(section(p, name).len() == n)
}
fn column(v: &Value) -> Result<Option<usize>> {
    let a = v.as_array().ok_or_else(bad)?;
    require(a.len() == 3 && a[0].is_string())?;
    match a[1].as_str() {
        Some("n") => {
            require(integer(&a[2])? <= 3)?;
            Ok(None)
        }
        Some("c") => {
            let levels = a[2].as_array().ok_or_else(bad)?;
            require(!levels.is_empty() && levels.len() <= MAX_LEVELS)?;
            require(levels.iter().all(Value::is_string))?;
            Ok(Some(levels.len()))
        }
        _ => Err(bad()),
    }
}

/// All checks precede schema cloning, floating-point expansion and net unpacking.
pub(super) fn validate(p: &Parts) -> Result<()> {
    let m = &p.meta;
    let cols = m["cols"].as_array().ok_or_else(bad)?;
    let d = cols.len();
    require(d <= MAX_FEATURES)?;
    let mut budget = Budget(0);
    budget.charge(mul(p.meta_bytes, 16)?)?;
    for bytes in p.sections.values() {
        budget.charge(mul(bytes.len(), 8)?)?;
    }
    budget.charge(mul(add(d, 1)?, 512)?)?;
    let mut numeric = 0;
    for c in cols.iter().chain(std::iter::once(&m["y"])) {
        if column(c)?.is_none() {
            numeric += 1;
        }
    }
    exact(p, "schema.affine", mul(numeric, 8)?)?;
    for name in ["schema.affine", "discrete.values", "net.params"] {
        require(section(p, name).len() % 4 == 0)?;
        require(
            section(p, name)
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite()),
        )?;
    }
    for name in ["marginal.range", "marginal.atoms", "noise"] {
        require(section(p, name).len() % 2 == 0)?;
        require(
            section(p, name)
                .chunks_exact(2)
                .all(|b| half::f16::from_le_bytes([b[0], b[1]]).is_finite()),
        )?;
    }
    let classes = column(&m["y"])?;
    let outputs = match (m["task"].as_str(), classes) {
        (Some("regression"), None) => 1,
        (Some("classification"), Some(k)) if k >= 2 => {
            if k == 2 {
                1
            } else {
                k
            }
        }
        _ => return Err(bad()),
    };
    if let Some(drop) = m.get("drop") {
        require(
            drop.as_array()
                .ok_or_else(bad)?
                .iter()
                .all(Value::is_string),
        )?;
    }
    if p.sections.contains_key("features.missing") {
        exact(p, "features.missing", d)?;
    }
    marginals(p, cols, &mut budget)?;
    dependence(p, d, &mut budget)?;
    if let Some(w) = p.sections.get("regime.weights") {
        require(!w.is_empty() && w.iter().any(|v| *v != 0))?;
        require(integer(&m["regimes"])? == w.len())?;
        let n = mul(w.len(), d)?;
        exact(p, "regime.means", n)?;
        exact(p, "regime.scales", n)?;
        budget.elements(add(mul(n, 2)?, w.len())?)?;
    } else {
        require(integer(&m["regimes"])? == 1)?;
        exact(p, "regime.means", 0)?;
        exact(p, "regime.scales", 0)?;
        budget.elements(add(mul(d, 2)?, 1)?)?;
    }
    let s = integers(&m["s"])?;
    require(s.len() <= d && s.iter().all(|f| *f < d))?;
    let mut seen = vec![false; d];
    for &f in &s {
        require(!seen[f])?;
        seen[f] = true;
    }
    let net = integers(&m["net"])?;
    require(net.len() == 7)?;
    let (inputs, hidden, out) = (net[0], net[1], net[2]);
    require(inputs == s.len() && out == outputs)?;
    let counts = [
        mul(hidden, inputs)?,
        hidden,
        mul(out, inputs)?,
        mul(out, hidden)?,
    ];
    let mut code_bytes = 0;
    for (n, bits) in counts.into_iter().zip(&net[3..]) {
        code_bytes = add(code_bytes, packed(n, *bits)?)?;
        budget.elements(n)?;
    }
    exact(p, "net.codes", code_bytes)?;
    exact(p, "net.params", mul(add(4, out)?, 4)?)?;
    budget.elements(out)?;
    forest(p, d, out, &mut budget)?;
    let t = integer(&m["t"])?;
    require(t <= inputs)?;
    let noise = if m["task"] == "regression" {
        add(t, add(5, RESIDUAL_KNOTS)?)?
    } else {
        require(t == 0)?;
        1
    };
    exact(p, "noise", mul(noise, 2)?)?;
    budget.elements(noise)
}

fn marginals(p: &Parts, cols: &[Value], budget: &mut Budget) -> Result<()> {
    let m = &p.meta;
    let knots = integer(&m["k"])?;
    require(knots >= 2)?;
    let disc = match m.get("disc") {
        None => None,
        Some(v) => Some(v.as_object().ok_or_else(bad)?),
    };
    if let Some(disc) = disc {
        for (key, value) in disc {
            let j = key.parse::<usize>().map_err(|_| bad())?;
            require(j < cols.len() && key == &j.to_string())?;
            let n = integer(value)?;
            require((1..=MAX_LEVELS).contains(&n))?;
        }
    }
    let (mut continuous, mut cumulative, mut values) = (0, 0, 0);
    for (j, c) in cols.iter().enumerate() {
        match disc.and_then(|d| d.get(&j.to_string())) {
            Some(n) => {
                let n = integer(n)?;
                if let Some(levels) = column(c)? {
                    require(n == levels)?;
                } else {
                    values = add(values, n)?;
                }
                cumulative = add(cumulative, n - 1)?;
                budget.elements(mul(n, 2)?)?;
            }
            None => {
                require(column(c)?.is_none())?;
                continuous += 1;
                budget.elements(knots)?;
            }
        }
    }
    exact(p, "marginal.range", mul(continuous, 4)?)?;
    exact(p, "marginal.knots", mul(continuous, knots - 2)?)?;
    exact(p, "discrete.cum", mul(cumulative, 2)?)?;
    exact(p, "discrete.values", mul(values, 4)?)?;
    let atoms = section(p, "marginal.atoms");
    require(atoms.len() % 8 == 0)?;
    for a in atoms.chunks_exact(8) {
        let j = half::f16::from_le_bytes([a[0], a[1]]).to_f64();
        require(j >= 0.0 && j.fract() == 0.0 && j < cols.len() as f64)?;
        require(disc.is_none_or(|d| !d.contains_key(&(j as usize).to_string())))?;
    }
    budget.elements(atoms.len() / 2)
}

fn dependence(p: &Parts, d: usize, budget: &mut Budget) -> Result<()> {
    let dep = p.meta["dep"].as_str().ok_or_else(bad)?;
    match dep {
        "independent" => Ok(()),
        "full" => {
            let cells = mul(d, d)?;
            exact(p, "copula.corr", mul(d, d.saturating_sub(1))? / 2)?;
            budget.elements(cells)
        }
        "tree" => {
            exact(p, "copula.parent", mul(d, 2)?)?;
            exact(p, "copula.rho", d)?;
            budget.elements(mul(d, 2)?)?;
            let parent: Vec<usize> = section(p, "copula.parent")
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as usize)
                .collect();
            require(parent.iter().all(|v| *v <= d))?;
            // Iterative coloring refuses cycles without recursive stack growth.
            let mut state = vec![0u8; d];
            for start in 0..d {
                let mut at = start;
                while state[at] == 0 {
                    state[at] = 1;
                    if parent[at] == 0 {
                        break;
                    }
                    at = parent[at] - 1;
                    require(state[at] != 1)?;
                }
                at = start;
                while state[at] == 1 {
                    state[at] = 2;
                    if parent[at] == 0 {
                        break;
                    }
                    at = parent[at] - 1;
                }
            }
            Ok(())
        }
        _ => {
            let rank = dep
                .strip_prefix("factor")
                .and_then(|r| r.parse::<usize>().ok())
                .ok_or_else(bad)?;
            require(rank > 0 && rank <= d)?;
            let n = mul(d, rank)?;
            exact(p, "copula.loadings", n)?;
            budget.elements(n)
        }
    }
}

fn forest(p: &Parts, d: usize, outputs: usize, budget: &mut Budget) -> Result<()> {
    let Some(meta) = p.meta.get("forest") else {
        return exact(p, "forest", 0);
    };
    let dims = integers(meta)?;
    require(dims.len() == 3)?;
    let (trees, depth, bits) = (dims[0], dims[1], dims[2]);
    require(trees > 0 && (1..=8).contains(&depth))?;
    let splits = mul(trees, depth)?;
    let leaves = mul(mul(trees, 1usize << depth)?, outputs)?;
    let prefix = add(mul(splits, if d <= 256 { 2 } else { 3 })?, mul(trees, 2)?)?;
    exact(p, "forest", add(prefix, packed(leaves, bits)?)?)?;
    budget.elements(add(add(mul(splits, 2)?, trees)?, leaves)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn full_matrix_refuses_missing_bytes_before_materialization_charge() {
        let parts = Parts {
            meta: json!({"dep": "full"}),
            sections: BTreeMap::from([("copula.corr".into(), vec![])]),
            header: 0,
            stored: 0,
            meta_bytes: 0,
        };
        let mut budget = Budget(0);
        assert!(dependence(&parts, MAX_FEATURES, &mut budget).is_err());
        assert_eq!(
            budget.0, 0,
            "section proof must precede matrix materialization"
        );
    }

    #[test]
    fn checked_sizes_and_materialization_budget_refuse_without_allocating() {
        assert!(mul(usize::MAX, 2).is_err());
        assert!(add(usize::MAX, 1).is_err());
        assert!(packed(usize::MAX, 8).is_err());
        let mut budget = Budget(0);
        assert!(budget.elements(MAX_MATERIALIZED / 32 + 1).is_err());
    }
}
