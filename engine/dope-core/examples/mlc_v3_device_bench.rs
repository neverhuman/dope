//! Latency of the pinned MLC-v3 629D encoder on CPU and (with `--features cuda`) CUDA.
//!
//! `cargo run --release --example mlc_v3_device_bench [--features cuda] -- CHECKPOINT [REPS]`
//! prints one JSON object: cold costs (read, SHA-256, device init, load), warm B=1
//! latency for the host API and the resident tensor API (with and without the
//! host round trip), and batched per-table cost. Measurements only; no gate.
use candle_core::{Device, Tensor};
use dope_core::mlc_v3::MlcV3Model;
use sha2::{Digest, Sha256};
use std::time::Instant;

const PINNED: &str = "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c";
type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn inputs(b: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = dope_core::rng::Rng::new(20261027);
    let mut draw = |n: usize| (0..n).map(|_| (rng.uniform() * 2.0 - 1.0) as f32).collect();
    (draw(b * 256), draw(b * 64 * 17))
}

fn summary(mut us: Vec<f64>) -> serde_json::Value {
    us.sort_by(f64::total_cmp);
    let q = |p: f64| us[((us.len() - 1) as f64 * p).round() as usize];
    serde_json::json!({"n": us.len(), "p50_us": q(0.5), "p95_us": q(0.95), "min_us": us[0]})
}

fn time<F: FnMut() -> Res<()>>(reps: usize, mut f: F) -> Res<serde_json::Value> {
    for _ in 0..reps.min(20) {
        f()?;
    }
    let mut us = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f()?;
        us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    Ok(summary(us))
}

fn device_report(model: &MlcV3Model, device: &Device, reps: usize) -> Res<serde_json::Value> {
    let (d1, c1) = inputs(1);
    let ds = Tensor::from_slice(&d1, (1, 256), device)?;
    let cs = Tensor::from_slice(&c1, (1, 64, 17), device)?;
    let resident = time(reps, || {
        model.encode_features_629_tensor(&ds, &cs, "high")?;
        Ok(device.synchronize()?)
    })?;
    let round_trip = time(reps, || {
        let ds = Tensor::from_slice(&d1, (1, 256), device)?;
        let cs = Tensor::from_slice(&c1, (1, 64, 17), device)?;
        let out = model.encode_features_629_tensor(&ds, &cs, "high")?;
        out.to_vec2::<f32>()?;
        Ok(())
    })?;
    let mut batched = serde_json::Map::new();
    for b in [1usize, 8, 64, 256, 1024] {
        let (d, c) = inputs(b);
        let ds = Tensor::from_slice(&d, (b, 256), device)?;
        let cs = Tensor::from_slice(&c, (b, 64, 17), device)?;
        let r = time((reps / b).clamp(5, reps), || {
            model.encode_features_629_tensor(&ds, &cs, "high")?;
            Ok(device.synchronize()?)
        })?;
        let per_table = r["p50_us"].as_f64().unwrap_or(0.0) / b as f64;
        batched.insert(
            b.to_string(),
            serde_json::json!({"call": r, "p50_us_per_table": per_table}),
        );
    }
    Ok(
        serde_json::json!({"b1_resident": resident, "b1_host_round_trip": round_trip, "batched": batched}),
    )
}

fn main() -> Res<()> {
    let path = std::env::args().nth(1).ok_or("expected checkpoint path")?;
    let reps: usize = std::env::args().nth(2).map_or(Ok(400), |s| s.parse())?;
    let t = Instant::now();
    let raw = std::fs::read(&path)?;
    let read_us = t.elapsed().as_secs_f64() * 1e6;
    let t = Instant::now();
    let digest = format!("{:x}", Sha256::digest(&raw));
    let sha_us = t.elapsed().as_secs_f64() * 1e6;
    if digest != PINNED {
        return Err("checkpoint SHA-256 mismatch".into());
    }
    let mut report = serde_json::json!({
        "checkpoint_sha256": digest, "reps": reps,
        "cold": {"read_us": read_us, "sha256_us": sha_us},
    });
    let t = Instant::now();
    let cpu = MlcV3Model::from_safetensors_bytes_on_device(&raw, &Device::Cpu)?;
    report["cold"]["load_cpu_us"] = (t.elapsed().as_secs_f64() * 1e6).into();
    let (d1, c1) = inputs(1);
    let cols: Vec<Vec<f32>> = c1.chunks(17).map(<[f32]>::to_vec).collect();
    report["cpu"] = device_report(&cpu, &Device::Cpu, reps)?;
    report["cpu"]["b1_host_api"] = time(reps, || {
        cpu.encode_features_629(&d1, &cols, "high")?;
        Ok(())
    })?;
    if cfg!(feature = "cuda") {
        let t = Instant::now();
        let cuda = Device::new_cuda(0)?;
        cuda.synchronize()?;
        report["cold"]["cuda_init_us"] = (t.elapsed().as_secs_f64() * 1e6).into();
        let t = Instant::now();
        let gpu = MlcV3Model::from_safetensors_bytes_on_device(&raw, &cuda)?;
        cuda.synchronize()?;
        report["cold"]["load_cuda_us"] = (t.elapsed().as_secs_f64() * 1e6).into();
        let (ds, cs) = (
            Tensor::from_slice(&d1, (1, 256), &cuda)?,
            Tensor::from_slice(&c1, (1, 64, 17), &cuda)?,
        );
        let t = Instant::now();
        gpu.encode_features_629_tensor(&ds, &cs, "high")?;
        cuda.synchronize()?;
        report["cold"]["first_cuda_forward_us"] = (t.elapsed().as_secs_f64() * 1e6).into();
        report["cuda"] = device_report(&gpu, &cuda, reps)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
