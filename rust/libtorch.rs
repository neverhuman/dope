use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use crate::error::{DopeError, Result};

/// libtorch owns a process-global random seed. Keep the seed and every
/// operation that consumes it in one critical section so parallel campaign
/// cells cannot perturb one another.
static SEEDED_LIBTORCH: Mutex<()> = Mutex::new(());

thread_local! {
    static LAST_GPU_PEAK_MEMORY_BYTES: Cell<Option<u64>> = const { Cell::new(None) };
}

unsafe extern "C" {
    fn cudaMemGetInfo(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
}

fn cuda_device_used_bytes() -> Result<u64> {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: CUDA writes exactly one size_t to each valid local pointer. The
    // status is checked before either value is used. This is called only after
    // tch confirms CUDA availability on the frozen device zero.
    let status = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if status != 0 || free > total {
        return Err(DopeError::Data(format!(
            "CUDA device-memory measurement failed with status {status}"
        )));
    }
    Ok(total.saturating_sub(free) as u64)
}

struct StopSampler<'a>(&'a AtomicBool);

impl Drop for StopSampler<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

pub(crate) fn clear_last_gpu_peak_memory_bytes() {
    LAST_GPU_PEAK_MEMORY_BYTES.set(None);
}

pub(crate) fn take_last_gpu_peak_memory_bytes() -> Option<u64> {
    LAST_GPU_PEAK_MEMORY_BYTES.take()
}

pub(crate) fn with_seeded_libtorch<T>(
    seed: u64,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":4096:8") {
        return Err(DopeError::Unsupported(
            "seeded libtorch requires CUBLAS_WORKSPACE_CONFIG=:4096:8".into(),
        ));
    }
    // A failed/panicking cell must not permanently disable later campaign
    // work. Recover the guard while preserving the original failure.
    let _guard = SEEDED_LIBTORCH
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // The pinned tch controls and CUDA workspace configuration establish the
    // supported deterministic training mode under this process-wide lock.
    tch::manual_seed(seed as i64);
    tch::Cuda::manual_seed_all(seed);
    tch::Cuda::cudnn_set_benchmark(false);
    if !tch::Cuda::is_available() {
        return operation();
    }
    let initial = cuda_device_used_bytes()?;
    let stop = AtomicBool::new(false);
    let maximum = AtomicU64::new(initial);
    std::thread::scope(|scope| {
        let monitor = scope.spawn(|| {
            while !stop.load(Ordering::Acquire) {
                if let Ok(used) = cuda_device_used_bytes() {
                    maximum.fetch_max(used, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        let guard = StopSampler(&stop);
        let result = operation();
        tch::Cuda::synchronize(0);
        maximum.fetch_max(cuda_device_used_bytes()?, Ordering::Relaxed);
        drop(guard);
        monitor
            .join()
            .map_err(|_| DopeError::Data("CUDA memory sampler panicked".into()))?;
        LAST_GPU_PEAK_MEMORY_BYTES.set(Some(maximum.load(Ordering::Relaxed)));
        result
    })
}

/// CPU research training shares the process-wide RNG lock with CUDA fits.
/// Torch manual_seed may query accelerator availability internally; this
/// helper makes no explicit CUDA API calls and all caller tensors use CPU.
#[cfg(feature = "gpu-research-training")]
pub(crate) fn with_seeded_cpu_libtorch<T>(
    seed: u64,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if std::env::var_os("CUDA_VISIBLE_DEVICES").as_deref() != Some(std::ffi::OsStr::new("")) {
        return Err(DopeError::Unsupported(
            "CPU research training requires empty CUDA_VISIBLE_DEVICES".into(),
        ));
    }
    let _guard = SEEDED_LIBTORCH
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    LAST_GPU_PEAK_MEMORY_BYTES.set(None);
    tch::manual_seed(seed as i64);
    operation()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu-research-training")]
    #[test]
    fn cpu_seed_replay_and_dispersion_use_cpu_tensors() {
        let draw = |seed| {
            with_seeded_cpu_libtorch(seed, || {
                let values = tch::Tensor::randn([32], (tch::Kind::Float, tch::Device::Cpu));
                assert_eq!(values.device(), tch::Device::Cpu);
                Vec::<f32>::try_from(values).map_err(|error| DopeError::Data(error.to_string()))
            })
            .unwrap()
        };
        let first = draw(23);
        assert_eq!(first, draw(23));
        assert_ne!(first, draw(37));
        assert_eq!(take_last_gpu_peak_memory_bytes(), None);
        let failed = std::panic::catch_unwind(|| {
            let _ = with_seeded_cpu_libtorch::<()>(23, || panic!("CPU fixture failure"));
        });
        assert!(failed.is_err());
        assert_eq!(with_seeded_cpu_libtorch(23, || Ok(11)).unwrap(), 11);
    }

    #[test]
    fn seeded_lock_recovers_after_a_panicking_cell() {
        let failed = std::panic::catch_unwind(|| {
            let _ = with_seeded_libtorch::<()>(7, || panic!("fixture failure"));
        });
        assert!(failed.is_err());
        assert_eq!(with_seeded_libtorch(7, || Ok(11)).unwrap(), 11);
    }

    #[test]
    fn seeded_operation_records_cuda_device_peak() {
        assert!(
            tch::Cuda::is_available(),
            "frozen CUDA runtime is unavailable"
        );
        clear_last_gpu_peak_memory_bytes();
        with_seeded_libtorch(11, || {
            let values = tch::Tensor::zeros([1024], (tch::Kind::Float, tch::Device::Cuda(0)));
            assert_eq!(values.size(), [1024]);
            Ok(())
        })
        .unwrap();
        assert!(take_last_gpu_peak_memory_bytes().is_some_and(|bytes| bytes >= 4096));
        assert_eq!(take_last_gpu_peak_memory_bytes(), None);
    }
}
