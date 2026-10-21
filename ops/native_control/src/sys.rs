//! Linux x86_64 only. Fresh single-threaded process; no libc or dynamic loader.
use core::arch::asm;

pub unsafe fn call(n: usize, a: usize, b: usize, c: usize, d: usize, e: usize, f: usize) -> isize {
    let result: isize;
    unsafe {
        asm!("syscall", inlateout("rax") n => result, in("rdi") a, in("rsi") b,
            in("rdx") c, in("r10") d, in("r8") e, in("r9") f,
            lateout("rcx") _, lateout("r11") _, options(nostack));
    }
    result
}
pub fn sc(n: usize, a: usize, b: usize, c: usize) -> isize {
    unsafe { call(n, a, b, c, 0, 0, 0) }
}
pub fn exit(code: usize) -> ! {
    sc(231, code, 0, 0);
    loop {
        core::hint::spin_loop();
    }
}
pub fn fail() -> ! {
    let _ = write(2, b"native_control_refused\n");
    exit(1)
}
pub fn check(ok: bool) {
    if !ok {
        fail();
    }
}
pub fn write(fd: usize, mut bytes: &[u8]) -> bool {
    while !bytes.is_empty() {
        let n = sc(1, fd, bytes.as_ptr() as usize, bytes.len());
        if n == -4 {
            continue;
        }
        if n <= 0 || n as usize > bytes.len() {
            return false;
        }
        bytes = &bytes[n as usize..];
    }
    true
}
pub fn close_inputs() {
    for fd in [1, 2] {
        let flags = sc(72, fd, 3, 0); // F_GETFL; write-only outputs only
        check(flags >= 0 && flags & 3 == 1);
    }
    check(sc(436, 3, u32::MAX as usize, 0) == 0); // close_range
    sc(3, 0, 0, 0);
}
#[repr(C)]
struct Filter {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}
impl Filter {
    const fn new(code: u16, jt: u8, jf: u8, k: u32) -> Self {
        Self { code, jt, jf, k }
    }
}
#[repr(C)]
struct Program {
    len: u16,
    filter: *const Filter,
}

fn filter(loading: bool) {
    const DENY: u32 = 0x50001; // EPERM, including unavailable probes
    const ALLOW: u32 = 0x7fff0000;
    let mut ins = [const { Filter::new(0, 0, 0, 0) }; 80];
    let mut n = 0;
    macro_rules! push {
        ($x:expr) => {{
            ins[n] = $x;
            n += 1;
        }};
    }
    push!(Filter::new(0x20, 0, 0, 4));
    push!(Filter::new(0x15, 1, 0, 0xc000003e));
    push!(Filter::new(0x06, 0, 0, 0x80000000));
    push!(Filter::new(0x20, 0, 0, 0));
    push!(Filter::new(0x35, 0, 1, 0x40000000));
    push!(Filter::new(0x06, 0, 0, 0x80000000));
    // Write is restricted to the checked result descriptors even if new FDs
    // could somehow be acquired; loading also permits bounded read-only opens.
    push!(Filter::new(0x15, 0, 5, 1));
    push!(Filter::new(0x20, 0, 0, 16));
    push!(Filter::new(0x15, 2, 0, 1));
    push!(Filter::new(0x15, 1, 0, 2));
    push!(Filter::new(0x06, 0, 0, DENY));
    push!(Filter::new(0x06, 0, 0, ALLOW));
    push!(Filter::new(0x20, 0, 0, 0));
    if loading {
        // No O_PATH, write handles, writable creates, metadata traversal or
        // process creation. Landlock separately confines read opens to cwd.
        push!(Filter::new(0x15, 0, 4, 257));
        push!(Filter::new(0x20, 0, 0, 32));
        push!(Filter::new(0x45, 0, 1, 0x200000 | 3 | 0x40 | 0x200 | 0x400));
        push!(Filter::new(0x06, 0, 0, DENY));
        push!(Filter::new(0x06, 0, 0, ALLOW));
        push!(Filter::new(0x20, 0, 0, 0));
        for number in [0, 3, 5, 72, 317, 436] {
            push!(Filter::new(0x15, 0, 1, number));
            push!(Filter::new(0x06, 0, 0, ALLOW));
        }
    }
    // After seal only output and exit remain. No mmap, file reads, sockets,
    // exec, signals, threads, proc aliases, io_uring, or inherited-FD borrowing.
    for number in [60, 231] {
        push!(Filter::new(0x15, 0, 1, number));
        push!(Filter::new(0x06, 0, 0, ALLOW));
    }
    push!(Filter::new(0x06, 0, 0, DENY));
    let program = Program {
        len: n as u16,
        filter: ins.as_ptr(),
    };
    check(sc(317, 1, 1, &program as *const _ as usize) == 0); // TSYNC
}
pub fn confine() {
    close_inputs();
    let cwd = sc(
        257,
        (-100isize) as usize,
        b".\0".as_ptr() as usize,
        0x200000 | 0x10000 | 0x80000,
    );
    check(cwd >= 0);
    let rights: u64 = (1 << 15) - 1; // Landlock ABI3 including TRUNCATE
    let rules = sc(444, &rights as *const _ as usize, 8, 0);
    check(rules >= 0);
    #[repr(C, packed)]
    struct Rule {
        access: u64,
        fd: i32,
    }
    let rule = Rule {
        access: (1 << 2) | (1 << 3),
        fd: cwd as i32,
    };
    check(unsafe { call(445, rules as usize, 1, &rule as *const _ as usize, 0, 0, 0) } == 0);
    check(unsafe { call(157, 38, 1, 0, 0, 0, 0) } == 0); // NO_NEW_PRIVS
    check(sc(446, rules as usize, 0, 0) == 0);
    sc(3, rules as usize, 0, 0);
    sc(3, cwd as usize, 0, 0);
    filter(true);
    // A real outside-bundle read must be denied during pre-read. No source
    // locator is taken from input. Caller makes the new bundle under its root.
    let outside = sc(
        257,
        (-100isize) as usize,
        b"/proc/self/status\0".as_ptr() as usize,
        0,
    );
    check(outside == -13 || outside == -1); // Landlock EACCES or seccomp EPERM
}
pub fn read_file(name: &[u8], buffer: &mut [u8]) -> usize {
    let fd = sc(
        257,
        (-100isize) as usize,
        name.as_ptr() as usize,
        0x20000 | 0x80000 | 0x800,
    ); // nofollow,cloexec,nonblock
    check(fd >= 0);
    // x86_64 Linux stat: mode offset24, size48, mtime88, ctime104.
    let mut before = [0u64; 18];
    let mut after = [0u64; 18];
    check(sc(5, fd as usize, before.as_mut_ptr() as usize, 0) == 0);
    check((before[3] as u32) & 0xf000 == 0x8000 && before[6] <= buffer.len() as u64);
    let size = before[6] as usize;
    let mut pos = 0;
    while pos < size {
        let n = sc(
            0,
            fd as usize,
            buffer[pos..].as_mut_ptr() as usize,
            size - pos,
        );
        if n == -4 {
            continue;
        }
        check(n > 0 && n as usize <= size - pos);
        pos += n as usize;
    }
    let mut extra = 0u8;
    check(sc(0, fd as usize, &mut extra as *mut _ as usize, 1) == 0);
    check(sc(5, fd as usize, after.as_mut_ptr() as usize, 0) == 0);
    for i in [0, 1, 3, 6, 11, 12, 13, 14] {
        check(before[i] == after[i]);
    }
    check(sc(3, fd as usize, 0, 0) == 0);
    size
}
pub fn seal_and_prove() -> u32 {
    close_inputs();
    filter(false);
    let mut byte = 0u8;
    // These real syscalls all must return EPERM, not a prefilled success flag.
    let denied = [
        sc(
            257,
            (-100isize) as usize,
            b"control.tdl\0".as_ptr() as usize,
            0,
        ),
        sc(41, 2, 1, 0),
        sc(59, b"/bin/true\0".as_ptr() as usize, 0, 0),
        sc(0, 0, &mut byte as *mut _ as usize, 1),
        sc(0, 3, &mut byte as *mut _ as usize, 1),
    ];
    check(denied.iter().all(|v| *v == -1));
    0x3f // outside read plus five sealed probes actually denied
}
