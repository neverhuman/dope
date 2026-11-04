//! Bounded Linux command supervisor. Never invokes Cargo or scientific tooling.
#[cfg(target_os = "linux")]
#[path = "native_process_io.rs"]
mod io;
#[cfg(target_os = "linux")]
#[path = "native_process_supervisor.rs"]
mod supervisor;
#[cfg(target_os = "linux")]
#[path = "native_process_tree.rs"]
mod tree;

fn main() {
    #[cfg(target_os = "linux")]
    let result = io::read_request().and_then(supervisor::invoke);
    #[cfg(not(target_os = "linux"))]
    let result: Result<serde_json::Value, &'static str> = Err("native_process_unavailable");
    match result {
        Ok(value) => println!("{value}"),
        Err(code) => {
            // No argv, paths, captured output or platform error text crosses this boundary.
            println!("{}", serde_json::json!({"error":code}));
            std::process::exit(2);
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
#[path = "native_process_tests.rs"]
mod tests;

#[cfg(all(test, target_os = "linux"))]
#[path = "native_process_correction_tests.rs"]
mod correction_tests;
