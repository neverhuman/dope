//! Build-pinned GNU libc math, fixed FMA variant on admitted x86_64 hosts.
//! This avoids silently substituting Rust libm rounding for the Python oracle.
use crate::sys::check;
extern "C" {
    fn __ieee754_log_fma(value: f64) -> f64;
    fn __sin_fma(value: f64) -> f64;
    fn __cos_fma(value: f64) -> f64;
    fn __erf(value: f64) -> f64;
}
pub fn supported() {
    let c = unsafe { core::arch::x86_64::__cpuid(1) }.ecx;
    check(c & ((1 << 12) | (1 << 27) | (1 << 28)) == ((1 << 12) | (1 << 27) | (1 << 28)));
    let state = unsafe { core::arch::x86_64::_xgetbv(0) };
    check(state & 6 == 6);
}
pub fn log(v: f64) -> f64 {
    unsafe { __ieee754_log_fma(v) }
}
pub fn sin(v: f64) -> f64 {
    unsafe { __sin_fma(v) }
}
pub fn cos(v: f64) -> f64 {
    unsafe { __cos_fma(v) }
}
pub fn erf(v: f64) -> f64 {
    unsafe { __erf(v) }
}
// Real one-thread TLS for the statically linked libc math errno/canary.
// The ELF TLS variable has four bytes at FS-4; no dynamic loader is used.
core::arch::global_asm!(
    ".section .tbss,\"awT\",@nobits",
    ".globl errno",
    ".type errno,@tls_object",
    ".align 4",
    "errno: .zero 4",
    ".size errno,4",
    ".previous"
);
#[repr(align(64))]
struct Tls([u8; 128]);
static mut TLS: Tls = Tls([0; 128]);
pub fn initialize_tls() {
    let raw = unsafe { core::ptr::addr_of_mut!(TLS.0).cast::<u8>() };
    let base = unsafe { raw.add(64) };
    // A real unpredictable stack canary, set before libc math or confinement.
    check(crate::sys::sc(318, unsafe { base.add(40) } as usize, 8, 0) == 8);
    unsafe {
        base.cast::<usize>().write(base as usize);
    }
    check(crate::sys::sc(158, 0x1002, base as usize, 0) == 0);
}
#[no_mangle]
extern "C" fn __stack_chk_fail() -> ! {
    crate::sys::fail()
}
