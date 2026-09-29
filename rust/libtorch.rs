use std::cell::Cell;
use std::sync::Mutex;

use crate::error::{DopeError, Result};

/// libtorch owns a process-global random seed. Keep the seed and every
/// operation that consumes it in one critical section so parallel campaign
/// cells cannot perturb one another.
static SEEDED_LIBTORCH: Mutex<()> = Mutex::new(());

thread_local! {
    static LAST_GPU_PEAK_MEMORY_BYTES: Cell<Option<u64>> = const { Cell::new(None) };
}

unsafe extern "C" {
    fn dope_enable_libtorch_determinism();
    fn dope_reset_cuda_peak_memory();
    fn dope_cuda_peak_memory_bytes() -> u64;
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
    // SAFETY: the bridge has no parameters, holds no Rust references, and
    // only enables process-global libtorch flags while the global lock is held.
    unsafe { dope_enable_libtorch_determinism() };
    tch::manual_seed(seed as i64);
    tch::Cuda::manual_seed_all(seed);
    tch::Cuda::cudnn_set_benchmark(false);
    let measure_cuda = tch::Cuda::is_available();
    if measure_cuda {
        // SAFETY: device zero is the frozen CUDA device and the seeded lock keeps
        // allocator peak resets from overlapping other campaign GPU operations.
        unsafe { dope_reset_cuda_peak_memory() };
    }
    let result = operation();
    if measure_cuda {
        // SAFETY: the query returns a value-owned allocator statistic and is made
        // under the same lock as the corresponding reset and operation.
        let peak = unsafe { dope_cuda_peak_memory_bytes() };
        LAST_GPU_PEAK_MEMORY_BYTES.set(Some(peak));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_lock_recovers_after_a_panicking_cell() {
        let failed = std::panic::catch_unwind(|| {
            let _ = with_seeded_libtorch::<()>(7, || panic!("fixture failure"));
        });
        assert!(failed.is_err());
        assert_eq!(with_seeded_libtorch(7, || Ok(11)).unwrap(), 11);
    }

    #[test]
    fn seeded_operation_records_cuda_allocator_peak() {
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
