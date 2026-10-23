//! Runtime diagnostics and explicit CPU worker configuration.
use dope_core::device;
use sha2::{Digest, Sha256};
use std::io::Read;

pub(crate) fn configure(threads: Option<u16>) -> Result<(), String> {
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(usize::from(threads))
            .build_global()
            .map_err(|_| "could not configure the CPU worker pool")?;
    }
    Ok(())
}

pub(crate) fn doctor(json: bool) -> Result<(), String> {
    let default = device::describe(&device::parse("auto").map_err(|e| e.to_string())?);
    if json {
        let path = std::env::current_exe().map_err(|e| e.to_string())?;
        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        let identity = serde_json::json!({
            "product": "dope-engine", "implementation": "native_rust",
            "version": env!("CARGO_PKG_VERSION"), "os": std::env::consts::OS,
            "architecture": std::env::consts::ARCH, "backends": device::backends(),
            "default_device": default, "threads": rayon::current_num_threads(),
            "python_required": false, "libtorch_required": false,
            "executable_sha256": format!("{:x}", hash.finalize()),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&identity).map_err(|e| e.to_string())?
        );
    } else {
        println!("dope {}", env!("CARGO_PKG_VERSION"));
        println!("  runtime    native Rust · Python not required");
        println!("  backends   {}", device::backends().join(", "));
        println!("  default    {default}");
        println!("  threads    {}", rayon::current_num_threads());
    }
    Ok(())
}
