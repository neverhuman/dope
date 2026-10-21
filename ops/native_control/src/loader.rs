#![no_std]
#![no_main]
mod hash;
mod inflate;
mod sys;
use sys::check;
core::arch::global_asm!(
    ".global _start",
    ".type _start,@function",
    "_start:",
    "mov rdi,rsp",
    "and rsp,-16",
    "call unpack_start",
    "ud2"
);
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    sys::fail()
}
static ENCODED: &[u8] = include_bytes!(env!("DOPE_NATIVE_CONTROL_IMAGE"));
static mut UNPACKED: [u8; 65536] = [0; 65536];
#[used]
static MARKER: [u8; 17] = *b"DOPECTRL_ABI2_V1\0";
fn u16_at(raw: &[u8], n: usize) -> usize {
    u16::from_le_bytes(raw[n..n + 2].try_into().unwrap()) as usize
}
fn u32_at(raw: &[u8], n: usize) -> usize {
    u32::from_le_bytes(raw[n..n + 4].try_into().unwrap()) as usize
}
fn u64_at(raw: &[u8], n: usize) -> usize {
    u64::from_le_bytes(raw[n..n + 8].try_into().unwrap()) as usize
}
#[no_mangle]
unsafe extern "C" fn unpack_start(stack: *const usize) -> ! {
    check(unsafe { *stack } == 1);
    check(unsafe { MARKER.as_ptr().read_volatile() } == b'D');
    // All input is inside the parent-pinned paid runtime. No source/artifact
    // descriptor or path is touched during decompression or code mapping.
    let raw = inflate::inflate(ENCODED, unsafe { &mut *core::ptr::addr_of_mut!(UNPACKED) });
    check(
        raw.len() >= 64
            && &raw[..7] == b"\x7fELF\x02\x01\x01"
            && u16_at(raw, 16) == 2
            && u16_at(raw, 18) == 62,
    );
    let offset = u64_at(raw, 32);
    let count = u16_at(raw, 56);
    check(
        u16_at(raw, 54) == 56
            && count > 0
            && count <= 16
            && offset <= raw.len()
            && count * 56 <= raw.len() - offset,
    );
    let entry = u64_at(raw, 24);
    let mut entry_valid = false;
    let mut prior_end = 0;
    let mut total = 0;
    for i in 0..count {
        let at = offset + i * 56;
        let kind = u32_at(raw, at);
        let flags = u32_at(raw, at + 4);
        check(kind != 2 && kind != 3); // no dynamic/interpreter segment
        if kind != 1 {
            continue;
        }
        let file = u64_at(raw, at + 8);
        let address = u64_at(raw, at + 16);
        let size = u64_at(raw, at + 32);
        let memory = u64_at(raw, at + 40);
        check(flags & 3 != 3 && size <= memory && file <= raw.len() && size <= raw.len() - file);
        check(
            address >= 0x400000
                && address < 0x4000000
                && memory > 0
                && memory <= 64 << 20
                && address + memory <= 0x4000000,
        );
        let start = address & !4095;
        let length = (address - start + memory + 4095) & !4095;
        check(start >= prior_end);
        prior_end = start + length;
        total += length;
        check(total <= 64 << 20);
        let mapped = unsafe { sys::call(9, start, length, 3, 0x100022, usize::MAX, 0) };
        check(mapped == start as isize); // MAP_FIXED_NOREPLACE: never clobber another mapping
        unsafe {
            core::ptr::copy_nonoverlapping(
                raw[file..file + size].as_ptr(),
                address as *mut u8,
                size,
            );
        }
        let protection = ((flags & 4) >> 2) | (flags & 2) | ((flags & 1) << 2);
        check(sys::sc(10, start, length, protection) == 0);
        if flags & 1 != 0 && entry >= address && entry < address + size {
            entry_valid = true;
        }
    }
    check(entry_valid);
    // Original entry performs close_range, Landlock and TSYNC seccomp before
    // any artifact read. It later seals every read/open/network/exec syscall.
    unsafe {
        core::arch::asm!("mov rsp,{stack}", "jmp {entry}", stack=in(reg) stack, entry=in(reg) entry, options(noreturn));
    }
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
