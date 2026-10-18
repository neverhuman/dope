mod service;
mod wire;

fn main() {
    if let Err(code) = service::run(std::env::args().skip(1).collect()) {
        eprintln!("dope-service: {code}");
        std::process::exit(2);
    }
}
