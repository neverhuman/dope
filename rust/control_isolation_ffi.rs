//! Versioned boundary ABI for a dedicated fresh child only; never a supervisor.
//! Codes: 0 success, 1 boundary refusal, 2 unwind, 3 unsupported platform.
//! Release panic=abort terminates the worker; no completion receipt is possible.

#[unsafe(no_mangle)]
pub extern "C" fn dope_control_isolation_version() -> u32 {
    1
}

fn boundary(operation: fn() -> crate::Result<()>) -> i32 {
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = operation;
        return 3;
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        // Dedicated process only: suppress the default value/path-bearing hook.
        std::panic::set_hook(Box::new(|_| {}));
        match std::panic::catch_unwind(operation) {
            Ok(Ok(())) => 0,
            Ok(Err(_)) => 1,
            Err(_) => 2,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dope_control_confine_current_bundle_directory() -> i32 {
    boundary(|| crate::typed_data::isolation::confine_bundle_directory(std::path::Path::new(".")))
}

#[unsafe(no_mangle)]
pub extern "C" fn dope_control_seal_sampling_process() -> i32 {
    boundary(crate::typed_data::isolation::seal_sampling_process)
}
