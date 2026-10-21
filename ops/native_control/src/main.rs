#![cfg_attr(all(target_os = "linux", target_arch = "x86_64"), no_std)]
#![cfg_attr(all(target_os = "linux", target_arch = "x86_64"), no_main)]

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn main() {
    eprintln!("native_control_platform_unsupported");
    std::process::exit(3);
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod decode;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod hash;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod host_math;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod inflate;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod sample;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod sys;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
core::arch::global_asm!(
    ".global _start",
    ".type _start,@function",
    "_start:",
    "mov rdi, rsp",
    "and rsp, -16",
    "call native_start",
    "ud2"
);

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    sys::fail()
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use crate::{decode, hash, sample, sys};
    static mut ARTIFACT: [u8; decode::MAX] = [0; decode::MAX];
    static mut JSON: [u8; decode::MAX] = [0; decode::MAX];
    static mut VALUES: [f64; decode::VALUES] = [0.0; decode::VALUES];
    static mut OUTPUT: [u8; 8_000_000] = [0; 8_000_000];
    // BSS is bounded anonymous state, not an uncharged file/runtime dependency.
    #[used]
    #[link_section = ".rodata.dopectrl"]
    static ABI_MARKER: [u8; 17] = *b"DOPECTRL_ABI2_V1\0";
    fn u64_at(raw: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(raw[at..at + 8].try_into().unwrap())
    }
    fn u32_at(raw: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(raw[at..at + 4].try_into().unwrap())
    }
    pub unsafe fn run(stack: *const usize) -> ! {
        // argv0 only; no source locators, dynamic options, loaders or model paths.
        sys::check(unsafe { *stack } == 1);
        // Volatile read retains the ABI marker in the paid executable.
        sys::check(unsafe { ABI_MARKER.as_ptr().read_volatile() } == b'D');
        crate::host_math::supported();
        crate::host_math::initialize_tls();
        sys::confine();
        let mut request = [0u8; 104];
        sys::check(sys::read_file(b"request.bin\0", &mut request) == 104);
        sys::check(&request[..8] == b"DNC2REQ\0");
        let method = u32_at(&request, 8);
        let flags = u32_at(&request, 12);
        let rows = u64_at(&request, 16);
        let seed = u64_at(&request, 24);
        let artifact_size = u64_at(&request, 32);
        sys::check(
            (method == 1 || method == 2)
                && flags == 0
                && rows > 0
                && rows <= 1_000_000
                && artifact_size > 0
                && artifact_size <= decode::MAX as u64,
        );
        // The only artifact access occurs AFTER exact request type/count admission.
        let artifact = unsafe { &mut *core::ptr::addr_of_mut!(ARTIFACT) };
        let size = sys::read_file(b"control.tdl\0", artifact);
        sys::check(
            size as u64 == artifact_size && hash::sha256(&artifact[..size]) == request[40..72],
        );
        let denied = sys::seal_and_prove();
        // No filesystem/network/exec/read syscall can succeed from this point.
        let json = crate::inflate::inflate(&artifact[..size], unsafe {
            &mut *core::ptr::addr_of_mut!(JSON)
        });
        let values = unsafe { &mut *core::ptr::addr_of_mut!(VALUES) };
        let model = decode::parse(json, method, values);
        let output = unsafe { &mut *core::ptr::addr_of_mut!(OUTPUT) };
        let bytes = sample::draw(&model, values, rows as usize, seed, output);
        let mut receipt = [0u8; 144];
        receipt[..8].copy_from_slice(b"DNC2OUT\0");
        receipt[8..16].copy_from_slice(&rows.to_le_bytes());
        receipt[16..24].copy_from_slice(&(model.columns as u64).to_le_bytes());
        receipt[24..32].copy_from_slice(&seed.to_le_bytes());
        receipt[32..36].copy_from_slice(&method.to_le_bytes());
        receipt[36..40].copy_from_slice(&denied.to_le_bytes());
        receipt[40..48].copy_from_slice(&(bytes as u64).to_le_bytes());
        receipt[48..80].copy_from_slice(&request[40..72]);
        receipt[80..112].copy_from_slice(&request[72..104]);
        receipt[112..144].copy_from_slice(&hash::sha256(&output[..bytes]));
        sys::check(sys::write(1, &output[..bytes]));
        sys::check(sys::write(2, &receipt));
        sys::exit(0)
    }
}
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[no_mangle]
unsafe extern "C" fn native_start(stack: *const usize) -> ! {
    unsafe { native::run(stack) }
}

// LLVM may lower core slice/array copies to these freestanding intrinsics.
// Volatile byte operations prevent optimization back into recursive libc calls.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[no_mangle]
unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    for i in 0..n {
        unsafe {
            dst.add(i).write_volatile(src.add(i).read_volatile());
        }
    }
    dst
}
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[no_mangle]
unsafe extern "C" fn memset(dst: *mut u8, value: i32, n: usize) -> *mut u8 {
    for i in 0..n {
        unsafe {
            dst.add(i).write_volatile(value as u8);
        }
    }
    dst
}
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[no_mangle]
unsafe extern "C" fn memmove(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    if (dst as usize) < src as usize {
        unsafe {
            memcpy(dst, src, n);
        }
    } else {
        for i in (0..n).rev() {
            unsafe {
                dst.add(i).write_volatile(src.add(i).read_volatile());
            }
        }
    }
    dst
}
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[no_mangle]
unsafe extern "C" fn memcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    for i in 0..n {
        let x = unsafe { a.add(i).read_volatile() };
        let y = unsafe { b.add(i).read_volatile() };
        if x != y {
            return x as i32 - y as i32;
        }
    }
    0
}
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[no_mangle]
unsafe extern "C" fn bcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    unsafe { memcmp(a, b, n) }
}
