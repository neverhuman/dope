//! Deterministic source-free `dope embed` (`PacketFeaturesV2` + `DopeEncoder629` bridge).

use crate::artifact::decode;
use crate::artifact::format2::{decode_format2, is_format2};
use crate::copula::Dependence;
use crate::error::Result;
use crate::marginal::Marginal;
use crate::sample::{sample_with_mode, EmissionMode, EmitOptions};
use crate::target::Head;
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const PACKET_FEATURES_V2_DIM: usize = 64;

/// Deterministic fixed-width 64-coordinate structural embedding extracted strictly from
/// a single `.dope` packet (Format 1 or Format 2) without opening any real dataset.
#[derive(Clone, Debug, Serialize)]
pub struct PacketFeaturesV2 {
    pub protocol: &'static str,
    pub format_version: u16,
    pub packet_bytes: usize,
    pub packet_sha256: String,
    pub features: Vec<f64>,
}

/// Compatibility bridge embedding into JopeDime's 4,096-row `DopeEncoder629` summary view.
#[derive(Clone, Debug, Serialize)]
pub struct Encoder629Bridge {
    pub packet_sha256: String,
    pub synthetic_rows: usize,
    pub feature_count: usize,
    pub col_means: Vec<f64>,
    pub col_stds: Vec<f64>,
    pub target_mean: f64,
    pub target_std: f64,
}

/// Extract `PacketFeaturesV2` deterministically from a Format 1 or Format 2 `.dope` packet.
pub fn embed_packet(bytes: &[u8]) -> Result<PacketFeaturesV2> {
    let sha = format!("{:x}", Sha256::digest(bytes));
    let mut vec = vec![0.0_f64; PACKET_FEATURES_V2_DIM];
    vec[0] = (bytes.len() as f64).max(1.0).ln();

    if is_format2(bytes) {
        let pkt = decode_format2(bytes)?;
        vec[1] = 2.0;
        vec[2] = pkt.sections.len() as f64;
        vec[3] = pkt.tiers.len() as f64;
        for (i, sec) in pkt.sections.iter().take(16).enumerate() {
            vec[4 + i] = (sec.data.len() as f64 + 1.0).ln();
            vec[20 + i] = sec.tag as f64;
        }
        for (i, tier) in pkt.tiers.iter().take(8).enumerate() {
            vec[36 + i] = (tier.tier_budget_bytes as f64 + 1.0).ln();
            vec[44 + i] = (tier.forest_trees_prefix as f64 + 1.0).ln();
        }
        return Ok(PacketFeaturesV2 {
            protocol: "dope-packet-features-v2",
            format_version: 2,
            packet_bytes: bytes.len(),
            packet_sha256: sha,
            features: vec,
        });
    }

    let p = decode(bytes)?;
    let d = p.features();
    let n_cont = p
        .marginals
        .iter()
        .filter(|m| matches!(m, Marginal::Continuous { .. }))
        .count();
    let n_disc = d.saturating_sub(n_cont);
    let mean_miss = if d > 0 {
        p.missing.iter().sum::<f64>() / d as f64
    } else {
        0.0
    };
    let max_miss = p.missing.iter().copied().fold(0.0_f64, f64::max);
    let n_atoms: usize = p
        .marginals
        .iter()
        .map(|m| match m {
            Marginal::Continuous { atoms, .. } => atoms.len(),
            _ => 0,
        })
        .sum();
    let dep_code = match &p.dependence {
        Dependence::Independent => 0.0,
        Dependence::Tree { .. } => 1.0,
        Dependence::Factor { rank, .. } => 2.0 + (*rank as f64) * 0.1,
        Dependence::Full { .. } => 4.0,
    };
    let (head_code, n_classes) = match p.head {
        Head::Regression => (0.0, 1.0),
        Head::Binary => (1.0, 2.0),
        Head::Multiclass(k) => (2.0, k as f64),
    };
    let params = p.parameters();

    vec[1] = 1.0;
    vec[2] = d as f64;
    vec[3] = (d as f64 + 1.0).ln();
    vec[4] = if d > 0 { n_cont as f64 / d as f64 } else { 0.0 };
    vec[5] = if d > 0 { n_disc as f64 / d as f64 } else { 0.0 };
    vec[6] = mean_miss;
    vec[7] = max_miss;
    vec[8] = n_atoms as f64;
    vec[9] = dep_code;
    vec[10] = p.regimes.k() as f64;
    vec[11] = head_code;
    vec[12] = n_classes;
    vec[13] = p.net.inputs as f64;
    vec[14] = p.net.hidden as f64;
    vec[15] = p.net.outputs as f64;
    vec[16] = p.forest.trees() as f64;
    vec[17] = p.forest.depth as f64;
    vec[18] = p.forest.bits as f64;
    vec[19] = (params.total as f64 + 1.0).ln();
    vec[20] = (params.forest as f64 + 1.0).ln();
    vec[21] = (params.network as f64 + 1.0).ln();
    vec[22] = (params.marginals as f64 + 1.0).ln();
    vec[23] = (params.dependence as f64 + 1.0).ln();
    vec[24] = p.temperature();

    // Spectral summary of copula mixing matrix energy (rows 25..40).
    let (mix, w) = p.dependence.mixing(d);
    let mut row_norms: Vec<f64> = (0..d)
        .map(|i| {
            let s: f64 = (0..w).map(|k| mix[i * w + k].powi(2)).sum();
            s.sqrt()
        })
        .collect();
    row_norms.sort_by(|a, b| b.total_cmp(a));
    for (i, &rn) in row_norms.iter().take(16).enumerate() {
        vec[25 + i] = rn;
    }

    // Forest scale profile (41..56).
    let mut scales = p.forest.scales.clone();
    scales.sort_by(|a, b| b.total_cmp(a));
    for (i, &sc) in scales.iter().take(16).enumerate() {
        vec[41 + i] = sc;
    }

    Ok(PacketFeaturesV2 {
        protocol: "dope-packet-features-v2",
        format_version: 1,
        packet_bytes: bytes.len(),
        packet_sha256: sha,
        features: vec,
    })
}

/// Deterministic 4,096-row source-free bridge summary for JopeDime `DopeEncoder629` compatibility.
pub fn embed_to_629_bridge(bytes: &[u8], seed: u64) -> Result<Encoder629Bridge> {
    let sha = format!("{:x}", Sha256::digest(bytes));
    let program = decode(bytes)?;
    let emitted = sample_with_mode(
        &program,
        &EmitOptions {
            mode: EmissionMode::Faithful,
            rows: 4096,
            seed,
            table_id: 629,
            ..Default::default()
        },
    )?;
    let d = program.features();
    let mut col_means = Vec::with_capacity(d);
    let mut col_stds = Vec::with_capacity(d);
    for j in 0..d {
        let fin: Vec<f64> = emitted.samples.x[j]
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        let m = if fin.is_empty() {
            0.0
        } else {
            fin.iter().sum::<f64>() / fin.len() as f64
        };
        let sd = if fin.len() < 2 {
            0.0
        } else {
            (fin.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / fin.len() as f64).sqrt()
        };
        col_means.push(m);
        col_stds.push(sd);
    }
    let ny = emitted.samples.y.len().max(1) as f64;
    let tm = emitted.samples.y.iter().sum::<f64>() / ny;
    let ts = (emitted
        .samples
        .y
        .iter()
        .map(|&v| (v - tm).powi(2))
        .sum::<f64>()
        / ny)
        .sqrt();
    Ok(Encoder629Bridge {
        packet_sha256: sha,
        synthetic_rows: emitted.samples.y.len(),
        feature_count: d,
        col_means,
        col_stds,
        target_mean: tm,
        target_std: ts,
    })
}
