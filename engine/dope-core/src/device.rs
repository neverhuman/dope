//! Device selection. One code path; the device is a runtime choice.

use crate::error::{DopeError, Result};
use candle_core::Device;

/// Parse `auto | cpu | cuda[:N] | metal[:N]`. `auto` prefers CUDA, then
/// Metal, then CPU, among the backends compiled into this build.
pub fn parse(spec: &str) -> Result<Device> {
    let (kind, ordinal) = match spec.split_once(':') {
        Some((k, n)) => (
            k,
            n.parse::<usize>()
                .map_err(|_| DopeError::Device("bad device ordinal".into()))?,
        ),
        None => (spec, 0),
    };
    match kind {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => Device::new_cuda(ordinal).map_err(|_| {
            DopeError::Device(
                "CUDA unavailable (build with --features cuda on a machine with a CUDA GPU)".into(),
            )
        }),
        "metal" => Device::new_metal(ordinal).map_err(|_| {
            DopeError::Device(
                "Metal unavailable (build with --features metal on Apple silicon)".into(),
            )
        }),
        "auto" => Ok(Device::cuda_if_available(ordinal)
            .ok()
            .filter(|d| d.is_cuda())
            .or_else(|| Device::new_metal(ordinal).ok())
            .unwrap_or(Device::Cpu)),
        _ => Err(DopeError::Device(
            "device must be auto, cpu, cuda[:N] or metal[:N]".into(),
        )),
    }
}

/// Human-readable list of backends compiled into this build.
pub fn backends() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    if candle_core::utils::cuda_is_available() {
        v.push("cuda");
    }
    if candle_core::utils::metal_is_available() {
        v.push("metal");
    }
    v
}

pub fn describe(device: &Device) -> String {
    match device {
        Device::Cpu => format!("cpu ({} threads)", rayon::current_num_threads()),
        Device::Cuda(_) => "cuda".into(),
        Device::Metal(_) => "metal".into(),
    }
}
