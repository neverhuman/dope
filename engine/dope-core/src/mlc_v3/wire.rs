//! Binary wire codec for `DML3` (`DOPE-MLC-v3` envelope + `CSC1` compact scaler sidecar + `LR-VLOT` multiclass) and `DML2`.

use super::inflate::{crc32_ieee, decompress_brotli, decompress_zlib};
use super::scaler::{bad, Csc1Sidecar, Cursor};
use crate::Result;
use half::f16;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub fn tier_byte_cap(level: &str) -> usize {
    match level {
        "low" | "small" => 2_048,
        "medium" => 4_096,
        "high" => 16_384,
        _ => 65_536,
    }
}

pub fn tier_coord_width(level: &str) -> usize {
    match level {
        "low" | "small" => 1_024,
        "medium" => 4_096,
        "high" => 8_192,
        _ => 10_240,
    }
}

pub const TIER_BYTE_CAPS: [(&str, usize); 4] = [
    ("small", 2_048),
    ("medium", 4_096),
    ("high", 16_384),
    ("ultra", 65_536),
];

pub const TIER_COORD_WIDTHS: [(&str, usize); 4] = [
    ("small", 1_024),
    ("medium", 4_096),
    ("high", 8_192),
    ("ultra", 10_240),
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObliviousTree {
    pub features: Vec<usize>,
    pub thresholds: Vec<f64>,
    pub leaf_values: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorLeafTree {
    pub features: Vec<usize>,
    pub thresholds: Vec<f64>,
    pub leaf_values: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetBundle {
    pub task_kind: String,
    pub gates: BTreeMap<String, f64>,
    pub pairs: Vec<(usize, usize)>,
    pub include_v3_primitives: bool,
    pub w_basis: Vec<f64>,
    pub res_net_mean: f64,
    pub neural_w: f64,
    pub trees: Vec<ObliviousTree>,
    pub residual_std: f64,
    pub noise_scale: f64,
    pub class_prior: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub k_classes: usize,
    pub class_priors: Vec<f64>,
    pub u_basis: Vec<Vec<f64>>,
    pub v_basis: Vec<Vec<f64>>,
    pub mc_trees: Vec<VectorLeafTree>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlcV3Encoding {
    pub level: String,
    pub task: String,
    pub code_width: usize,
    pub n_train: usize,
    pub p_use: usize,
    pub gates: BTreeMap<String, f64>,
    pub task_latent: Vec<f32>,
    pub dist_latent: Vec<f32>,
    pub knot_grid: Vec<f64>,
    pub col_knots: Vec<Vec<f64>>,
    pub is_binary_col: Vec<bool>,
    pub binary_probs: Vec<f64>,
    pub binary_low_high: Vec<[f64; 2]>,
    pub discrete_support: BTreeMap<usize, (Vec<f64>, Vec<f64>)>,
    pub chol_factor: Vec<Vec<f64>>,
    pub diag_res_std: Vec<f64>,
    pub regime_weights: Vec<f64>,
    pub regime_means: Vec<Vec<f64>>,
    pub regime_scales: Vec<Vec<f64>>,
    pub target_bundle: TargetBundle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dml3Envelope {
    pub wire_format: String,
    pub wire_bytes: usize,
    pub tier_cap_bytes: usize,
    pub within_tier_cap: bool,
    pub crc32: Option<u32>,
    pub compressed_flags: u8,
    pub encoding: MlcV3Encoding,
    pub scaler: Option<Csc1Sidecar>,
}

fn reshape_2d(flat: Vec<f64>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    flat.chunks_exact(cols)
        .take(rows)
        .map(|c| c.to_vec())
        .collect()
}

pub fn decode_dml2_core(payload: &[u8]) -> Result<MlcV3Encoding> {
    let body = if payload.starts_with(b"DML2") {
        &payload[4..]
    } else {
        payload
    };
    let mut c = Cursor::new(body);
    let (ver_byte, lvl_id, task_id, n_trees) = (c.u8()?, c.u8()?, c.u8()?, c.u8()? as usize);
    let ver = ver_byte & 0x0F;
    if ver != 1 && ver != 2 {
        return Err(bad("unsupported dml2 wire version"));
    }
    let (n_train, p_use) = (c.u32()? as usize, c.u32()? as usize);
    let (n_reg, rank_use, n_knots, n_pairs, n_w, n_disc) = (
        c.u16()? as usize,
        c.u16()? as usize,
        c.u16()? as usize,
        c.u16()? as usize,
        c.u16()? as usize,
        c.u16()? as usize,
    );
    let level = match lvl_id {
        0 => "medium",
        1 => "high",
        2 => "ultra",
        _ => return Err(bad("invalid dml2 level id")),
    }
    .to_string();
    let task = match task_id {
        0 => "regression",
        1 => "classification",
        _ => return Err(bad("invalid dml2 task id")),
    }
    .to_string();
    const DECIMALS: [i32; 7] = [6, 5, 6, 5, 6, 5, 5];
    let sc: Vec<f64> = if ver == 1 {
        c.f16_vec(11)?
    } else if (ver_byte & 0x10) == 0 {
        let mut v = Vec::with_capacity(11);
        for &dec in &DECIMALS {
            let q = c.i32()? as f64;
            v.push(q / 10f64.powi(dec));
        }
        for _ in 0..4 {
            v.push(c.f32()? as f64);
        }
        v
    } else {
        let mut v = Vec::with_capacity(11);
        for _ in 0..11 {
            v.push(c.f64()?);
        }
        v
    };
    let regime_weights = c.f16_vec(n_reg)?;
    let regime_means = reshape_2d(c.f16_vec(n_reg * p_use)?, n_reg, p_use);
    let regime_scales = reshape_2d(c.f16_vec(n_reg * p_use)?, n_reg, p_use);
    let chol_factor = reshape_2d(c.f16_vec(p_use * rank_use)?, p_use, rank_use);
    let diag_res_std = c.f16_vec(p_use)?;
    let dist_latent = c.f16_vec(128)?.into_iter().map(|v| v as f32).collect();
    let task_latent = c.f16_vec(128)?.into_iter().map(|v| v as f32).collect();
    let col_knots = reshape_2d(c.f16_vec(p_use * n_knots)?, p_use, n_knots);
    let binary_probs = c.f16_vec(p_use)?;
    let bin_lh_flat = c.f16_vec(p_use * 2)?;
    let binary_low_high = bin_lh_flat.chunks_exact(2).map(|b| [b[0], b[1]]).collect();
    let n_pack = p_use.div_ceil(8);
    let pack_bytes = c.take(n_pack)?;
    let is_binary_col = (0..p_use)
        .map(|j| ((pack_bytes[j / 8] >> (7 - (j % 8))) & 1) != 0)
        .collect();
    let mut discrete_support = BTreeMap::new();
    for _ in 0..n_disc {
        let (col_j, u_len) = (c.u16()? as usize, c.u16()? as usize);
        discrete_support.insert(col_j, (c.f16_vec(u_len)?, c.f16_vec(u_len)?));
    }
    let mut pairs = Vec::with_capacity(n_pairs);
    if n_pairs > 0 {
        let pb = c.take(n_pairs * 2)?;
        for chunk in pb.chunks_exact(2) {
            pairs.push((chunk[0] as usize, chunk[1] as usize));
        }
    }
    let w_basis = c.f16_vec(n_w)?;
    let mut trees = Vec::with_capacity(n_trees);
    for _ in 0..n_trees {
        let hdr = c.u8()?;
        let depth = (if ver == 1 { hdr } else { hdr & 0x1F }) as usize;
        let mode = if ver == 1 { 0 } else { hdr >> 5 };
        let feats = c.take(depth)?.iter().map(|&b| b as usize).collect();
        let thresholds = if ver == 1 {
            c.f16_vec(depth)?
        } else if mode == 0 {
            let mut t = Vec::with_capacity(depth);
            for _ in 0..depth {
                t.push(c.i16()? as f64 / 10_000.0);
            }
            t
        } else if mode == 1 {
            let mut t = Vec::with_capacity(depth);
            for _ in 0..depth {
                t.push(c.i32()? as f64 / 10_000.0);
            }
            t
        } else {
            let mut t = Vec::with_capacity(depth);
            for _ in 0..depth {
                t.push(c.f64()?);
            }
            t
        };
        let leaf_values = c.f16_vec(1usize << depth)?;
        trees.push(ObliviousTree {
            features: feats,
            thresholds,
            leaf_values,
        });
    }
    if c.off != body.len() {
        return Err(bad("dml2 wire payload has trailing bytes"));
    }
    let (k_lo, k_hi) = if level == "medium" {
        (0.01, 0.99)
    } else {
        (0.005, 0.995)
    };
    let knot_grid = (0..n_knots)
        .map(|i| {
            let v = if n_knots <= 1 {
                0.5
            } else {
                k_lo + (k_hi - k_lo) * (i as f64) / ((n_knots - 1) as f64)
            };
            f16::from_f64(v).to_f64()
        })
        .collect();
    let mut gates = BTreeMap::new();
    gates.insert("monotone_w".into(), sc[7]);
    gates.insert("hinge_w".into(), sc[8]);
    gates.insert("pair_w".into(), sc[9]);
    gates.insert("tree_w".into(), sc[10]);
    Ok(MlcV3Encoding {
        code_width: tier_coord_width(&level),
        level,
        task: task.clone(),
        n_train,
        p_use,
        gates: gates.clone(),
        task_latent,
        dist_latent,
        knot_grid,
        col_knots,
        is_binary_col,
        binary_probs,
        binary_low_high,
        discrete_support,
        chol_factor,
        diag_res_std,
        regime_weights,
        regime_means,
        regime_scales,
        target_bundle: TargetBundle {
            task_kind: task,
            gates,
            pairs,
            include_v3_primitives: false,
            w_basis,
            res_net_mean: sc[0],
            neural_w: sc[1],
            trees,
            residual_std: sc[2],
            noise_scale: sc[3],
            class_prior: sc[4],
            y_min: sc[5],
            y_max: sc[6],
            k_classes: 2,
            class_priors: Vec::new(),
            u_basis: Vec::new(),
            v_basis: Vec::new(),
            mc_trees: Vec::new(),
        },
    })
}

impl Dml3Envelope {
    pub fn from_bytes(wire: &[u8]) -> Result<Self> {
        if wire.len() < 10 {
            return Err(bad("mlc_v3 wire packet too short"));
        }
        if wire.len() > 262_144 {
            return Err(bad("mlc_v3 wire packet exceeds maximum envelope size"));
        }
        if !wire.starts_with(b"DML3") {
            let enc = decode_dml2_core(wire)?;
            let cap = tier_byte_cap(&enc.level);
            return Ok(Self {
                wire_format: "DML2".into(),
                wire_bytes: wire.len(),
                tier_cap_bytes: cap,
                within_tier_cap: wire.len() <= cap,
                crc32: None,
                compressed_flags: 0,
                encoding: enc,
                scaler: None,
            });
        }
        let ver = wire[4];
        let flags = wire[5];
        let expected_crc = u32::from_le_bytes(wire[6..10].try_into().unwrap());
        let body = &wire[10..];
        if ver != 3 {
            return Err(bad("unsupported dml3 envelope version"));
        }
        if crc32_ieee(body) != expected_crc {
            return Err(bad("dml3 wire crc32 mismatch"));
        }
        let payload = match flags & 3 {
            0 => body.to_vec(),
            1 => decompress_zlib(body, 4_194_304)?,
            2 => decompress_brotli(body, 4_194_304)?,
            _ => return Err(bad("unsupported dml3 compression flag")),
        };
        let mut c = Cursor::new(&payload);
        let (lvl_id, task_id, inc_v3, _res) = (c.u8()?, c.u8()?, c.u8()?, c.u8()?);
        let (len_core, len_sc, len_mc) = (c.u32()? as usize, c.u32()? as usize, c.u32()? as usize);
        let core_bytes = c.take(len_core)?;
        let sc_bytes = c.take(len_sc)?;
        let mc_bytes = c.take(len_mc)?;
        if c.off != payload.len() {
            return Err(bad("dml3 payload trailing bytes"));
        }
        let mut enc = decode_dml2_core(core_bytes)?;
        enc.level = match lvl_id {
            0 => "small",
            1 => "medium",
            2 => "high",
            3 => "ultra",
            _ => return Err(bad("invalid dml3 level id")),
        }
        .to_string();
        enc.code_width = tier_coord_width(&enc.level);
        if matches!(enc.level.as_str(), "small" | "medium") {
            let n_k = enc.knot_grid.len();
            enc.knot_grid = (0..n_k)
                .map(|i| {
                    f16::from_f64(if n_k <= 1 {
                        0.5
                    } else {
                        0.01 + 0.98 * (i as f64) / ((n_k - 1) as f64)
                    })
                    .to_f64()
                })
                .collect();
        }
        enc.task = match task_id {
            0 => "regression",
            1 => "classification",
            2 => "multiclass",
            _ => return Err(bad("invalid dml3 task id")),
        }
        .to_string();
        enc.target_bundle.include_v3_primitives = inc_v3 != 0;
        if task_id == 2 && !mc_bytes.is_empty() {
            let mut mc = Cursor::new(mc_bytes);
            let (k_cls, u_r, u_c, n_trees) = (
                mc.u16()? as usize,
                mc.u16()? as usize,
                mc.u16()? as usize,
                mc.u16()? as usize,
            );
            let priors = mc.f16_vec(k_cls)?;
            let u_basis = reshape_2d(mc.f16_vec(u_r * u_c)?, u_r, u_c);
            let v_basis = reshape_2d(mc.f16_vec(k_cls * u_c)?, k_cls, u_c);
            let mut mc_trees = Vec::with_capacity(n_trees);
            for _ in 0..n_trees {
                let d_t = mc.u8()? as usize;
                let feats = mc.take(d_t)?.iter().map(|&b| b as usize).collect();
                let thresholds = mc.f16_vec(d_t)?;
                let leaf_values =
                    reshape_2d(mc.f16_vec((1usize << d_t) * k_cls)?, 1usize << d_t, k_cls);
                mc_trees.push(VectorLeafTree {
                    features: feats,
                    thresholds,
                    leaf_values,
                });
            }
            if mc.off != mc_bytes.len() {
                return Err(bad("dml3 multiclass block trailing bytes"));
            }
            enc.target_bundle.task_kind = "multiclass".into();
            enc.target_bundle.k_classes = k_cls;
            enc.target_bundle.class_priors = priors;
            enc.target_bundle.u_basis = u_basis;
            enc.target_bundle.v_basis = v_basis;
            enc.target_bundle.mc_trees = mc_trees;
        }
        let scaler = if sc_bytes.is_empty() {
            None
        } else {
            let sc = Csc1Sidecar::from_bytes(sc_bytes)?;
            if sc.p_use != enc.p_use {
                return Err(bad("dml3 scaler p_use mismatch"));
            }
            Some(sc)
        };
        let cap = tier_byte_cap(&enc.level);
        Ok(Self {
            wire_format: "DML3".into(),
            wire_bytes: wire.len(),
            tier_cap_bytes: cap,
            within_tier_cap: wire.len() <= cap,
            crc32: Some(expected_crc),
            compressed_flags: flags,
            encoding: enc,
            scaler,
        })
    }
}
