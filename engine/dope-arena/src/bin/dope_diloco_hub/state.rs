//! SafeTensors parsing, serialization, and Nesterov outer-momentum state for `dope_diloco_hub`.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MAX_HEADER_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

pub type TensorMap = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

#[derive(Debug, Clone)]
pub struct HubConfig {
    pub state_dir: PathBuf,
    pub sync_group: String,
    pub outer_lr: f32,
    pub outer_momentum: f32,
    pub staleness_alpha: f32,
    pub save_every: u64,
    pub max_delta_rms: f32,
}

pub struct HubState {
    pub cfg: HubConfig,
    pub round: u64,
    pub global_sha: String,
    pub weights: Option<TensorMap>,
    pub momentum: TensorMap,
    pub cached_bytes: Vec<u8>,
    pub workers: HashSet<String>,
    pub total_syncs: u64,
    pub rejected_syncs: u64,
}

fn now_unix_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

pub fn parse_f32_safetensors(bytes: &[u8]) -> Result<TensorMap, String> {
    if bytes.len() < 8 || bytes.len() > MAX_BODY_BYTES {
        return Err(format!("invalid safetensors byte length: {}", bytes.len()));
    }
    let hlen = u64::from_le_bytes(bytes[..8].try_into().map_err(|_| "short header")?) as usize;
    if hlen == 0 || hlen > MAX_HEADER_BYTES || 8 + hlen > bytes.len() {
        return Err(format!("invalid safetensors header length: {hlen}"));
    }
    let hdr_val: Value =
        serde_json::from_slice(&bytes[8..8 + hlen]).map_err(|e| format!("header json: {e}"))?;
    let obj = hdr_val.as_object().ok_or("header is not json object")?;
    let body = &bytes[8 + hlen..];
    let mut out = TensorMap::new();
    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        let m = v
            .as_object()
            .ok_or_else(|| format!("tensor {k} meta not object"))?;
        if m.get("dtype").and_then(Value::as_str) != Some("F32") {
            return Err(format!("tensor {k} must be F32"));
        }
        let shape: Vec<usize> = m
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("tensor {k} missing shape"))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("bad dim in {k}"))
            })
            .collect::<Result<_, _>>()?;
        let offs = m
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("tensor {k} missing offsets"))?;
        if offs.len() != 2 {
            return Err(format!("tensor {k} offsets len != 2"));
        }
        let s0 = offs[0].as_u64().ok_or("bad offset start")? as usize;
        let s1 = offs[1].as_u64().ok_or("bad offset end")? as usize;
        let expect_elems: usize = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or("shape overflow")?;
        if s1 < s0 || s1 > body.len() || (s1 - s0) != expect_elems.saturating_mul(4) {
            return Err(format!("tensor {k} offset/shape mismatch"));
        }
        let mut data = Vec::with_capacity(expect_elems);
        for chunk in body[s0..s1].chunks_exact(4) {
            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !val.is_finite() {
                return Err(format!("non-finite value in tensor {k}"));
            }
            data.push(val);
        }
        out.insert(k.clone(), (shape, data));
    }
    if out.is_empty() {
        return Err("empty safetensors payload".into());
    }
    Ok(out)
}

pub fn encode_f32_safetensors(tensors: &TensorMap) -> Result<Vec<u8>, String> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    let mut total_body = 0usize;
    for (k, (shape, data)) in tensors {
        let nbytes = data.len() * 4;
        header.insert(
            k.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + nbytes]}),
        );
        offset += nbytes;
        total_body += nbytes;
    }
    let mut hb = serde_json::to_vec(&Value::Object(header))
        .map_err(|e| format!("cannot serialise the safetensors header: {e}"))?;
    while hb.len() % 8 != 0 {
        hb.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + hb.len() + total_body);
    out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
    out.extend_from_slice(&hb);
    for (_, data) in tensors.values() {
        for &v in data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(out)
}

fn atomic_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("part");
    fs::write(&tmp, data)?;
    fs::rename(&tmp, path)
}

impl HubState {
    pub fn load_or_new(cfg: HubConfig) -> Self {
        let _ = fs::create_dir_all(&cfg.state_dir);
        let latest_p = cfg.state_dir.join("latest_global.safetensors");
        let meta_p = cfg.state_dir.join("hub_meta.json");
        let (mut round, mut weights, mut cached_bytes, mut global_sha) =
            (0u64, None, Vec::new(), String::new());
        if let Ok(raw) = fs::read(&latest_p) {
            if let Ok(parsed) = parse_f32_safetensors(&raw) {
                global_sha = sha256_hex(&raw);
                cached_bytes = raw;
                weights = Some(parsed);
                round = fs::read_to_string(&meta_p)
                    .ok()
                    .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                    .and_then(|v| v.get("round").and_then(Value::as_u64))
                    .unwrap_or(1);
            }
        }
        Self {
            cfg,
            round,
            global_sha,
            weights,
            momentum: TensorMap::new(),
            cached_bytes,
            workers: HashSet::new(),
            total_syncs: 0,
            rejected_syncs: 0,
        }
    }

    pub fn apply_sync(
        &mut self,
        worker_id: &str,
        mode: &str,
        base_round: u64,
        payload: &[u8],
    ) -> Result<(u64, String, Vec<u8>), String> {
        let incoming = parse_f32_safetensors(payload)?;
        self.workers.insert(worker_id.to_string());
        if mode == "init" {
            if self.weights.is_some() {
                return Ok((
                    self.round,
                    self.global_sha.clone(),
                    self.cached_bytes.clone(),
                ));
            }
            let encoded = encode_f32_safetensors(&incoming)?;
            self.persist_global(incoming, encoded, worker_id, 0, 0, 0.0)?;
            return Ok((
                self.round,
                self.global_sha.clone(),
                self.cached_bytes.clone(),
            ));
        }
        let Some(ref mut g_weights) = self.weights else {
            let encoded = encode_f32_safetensors(&incoming)?;
            self.persist_global(incoming, encoded, worker_id, 0, 0, 0.0)?;
            return Ok((
                self.round,
                self.global_sha.clone(),
                self.cached_bytes.clone(),
            ));
        };
        if g_weights.len() != incoming.len() {
            self.rejected_syncs += 1;
            return Err("tensor key count mismatch".into());
        }
        for (k, (shape, data)) in &incoming {
            match g_weights.get(k) {
                Some((gs, gd)) if gs == shape && gd.len() == data.len() => {}
                _ => {
                    self.rejected_syncs += 1;
                    return Err(format!("shape/key mismatch on {k}"));
                }
            }
        }
        let staleness = self.round.saturating_sub(base_round);
        let scale = 1.0f32 / (1.0f32 + self.cfg.staleness_alpha * (staleness as f32));
        let (beta, eta, max_rms) = (
            self.cfg.outer_momentum,
            self.cfg.outer_lr,
            self.cfg.max_delta_rms,
        );
        let (mut sum_sq, mut total_elems, mut next_g) = (0.0f64, 0usize, TensorMap::new());
        for (k, (shape, g_data)) in g_weights.iter() {
            let inc_data = &incoming[k].1;
            let raw_deltas: Vec<f32> = if mode == "delta" {
                inc_data.clone()
            } else {
                g_data.iter().zip(inc_data).map(|(g, w)| g - w).collect()
            };
            let t_sq: f64 = raw_deltas
                .iter()
                .map(|&d| f64::from(d) * f64::from(d))
                .sum();
            let rms = (t_sq / (raw_deltas.len().max(1) as f64)).sqrt() as f32;
            let clamp = if max_rms > 0.0 && rms > max_rms {
                max_rms / rms
            } else {
                1.0
            };
            let eff_scale = scale * clamp;
            sum_sq += t_sq * f64::from(clamp * clamp);
            total_elems += raw_deltas.len();
            let mom = self
                .momentum
                .entry(k.clone())
                .or_insert_with(|| (shape.clone(), vec![0.0f32; g_data.len()]));
            let mut updated = Vec::with_capacity(g_data.len());
            for i in 0..g_data.len() {
                let d = raw_deltas[i] * eff_scale;
                let v = beta * mom.1[i] + d;
                mom.1[i] = v;
                let w_new = g_data[i] - eta * (d + beta * v);
                if !w_new.is_finite() {
                    self.rejected_syncs += 1;
                    return Err(format!("non-finite post-update weight in {k}"));
                }
                updated.push(w_new);
            }
            next_g.insert(k.clone(), (shape.clone(), updated));
        }
        let delta_rms = (sum_sq / (total_elems.max(1) as f64)).sqrt();
        let encoded = encode_f32_safetensors(&next_g)?;
        self.persist_global(next_g, encoded, worker_id, base_round, staleness, delta_rms)?;
        Ok((
            self.round,
            self.global_sha.clone(),
            self.cached_bytes.clone(),
        ))
    }

    fn persist_global(
        &mut self,
        next_g: TensorMap,
        encoded: Vec<u8>,
        worker_id: &str,
        base_round: u64,
        staleness: u64,
        delta_rms: f64,
    ) -> Result<(), String> {
        let sha = sha256_hex(&encoded);
        let next_round = self.round + 1;
        atomic_write(
            &self.cfg.state_dir.join("latest_global.safetensors"),
            &encoded,
        )
        .map_err(|e| format!("write latest: {e}"))?;
        if self.cfg.save_every > 0 && (next_round == 1 || next_round % self.cfg.save_every == 0) {
            let snap = self
                .cfg
                .state_dir
                .join(format!("global_round_{next_round:06}.safetensors"));
            atomic_write(&snap, &encoded).map_err(|e| format!("write snapshot: {e}"))?;
        }
        let meta = json!({
            "round": next_round, "global_sha256": sha, "sync_group": self.cfg.sync_group,
            "outer_lr": self.cfg.outer_lr, "outer_momentum": self.cfg.outer_momentum,
            "unique_workers": self.workers.len(), "total_syncs": self.total_syncs + 1,
            "bytes": encoded.len(), "updated_at": now_unix_s(),
        });
        let _ = atomic_write(
            &self.cfg.state_dir.join("hub_meta.json"),
            format!("{meta}\n").as_bytes(),
        );
        if let Ok(mut fh) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.cfg.state_dir.join("hub_rounds.jsonl"))
        {
            let line = json!({
                "round": next_round, "time": now_unix_s(), "worker_id": worker_id,
                "base_round": base_round, "staleness": staleness, "delta_rms": delta_rms,
                "global_sha": &sha[..16], "unique_workers": self.workers.len(),
            });
            let _ = writeln!(fh, "{line}");
        }
        self.round = next_round;
        self.global_sha = sha;
        self.weights = Some(next_g);
        self.cached_bytes = encoded;
        self.total_syncs += 1;
        Ok(())
    }

    pub fn status_json(&self) -> String {
        json!({
            "round": self.round,
            "global_sha": if self.global_sha.len() >= 16 { &self.global_sha[..16] } else { "" },
            "sync_group": self.cfg.sync_group,
            "outer_lr": self.cfg.outer_lr,
            "outer_momentum": self.cfg.outer_momentum,
            "unique_workers": self.workers.len(),
            "total_syncs": self.total_syncs,
            "rejected_syncs": self.rejected_syncs,
            "has_weights": self.weights.is_some(),
            "bytes": self.cached_bytes.len(),
        })
        .to_string()
    }
}
