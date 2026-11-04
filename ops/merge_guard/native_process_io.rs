use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;

pub(super) type Result<T> = std::result::Result<T, &'static str>;
pub(super) const CAPTURE_LIMIT: u64 = 32 << 20;
// Existing campaign_artifacts.file_hash/read_regular admission, separate from FSIZE.
const HASH_LIMIT: u64 = 8 << 20;
pub(super) struct Request {
    pub argv: Vec<String>,
    pub root: PathBuf,
    pub label: String,
    pub wall_ms: u64,
    pub ram_bytes: u64,
}

pub(super) fn read_request() -> Result<Request> {
    let mut raw = Vec::new();
    std::io::stdin()
        .take((1 << 20) + 1)
        .read_to_end(&mut raw)
        .map_err(|_| "native_process_unavailable")?;
    if raw.len() > 1 << 20 {
        return Err("native_process_unavailable");
    }
    parse_request(&raw)
}

pub(super) fn parse_request(raw: &[u8]) -> Result<Request> {
    let v: serde_json::Value =
        serde_json::from_slice(raw).map_err(|_| "native_process_unavailable")?;
    let obj = v.as_object().ok_or("native_process_unavailable")?;
    if obj.len() != 5
        || ["argv", "root", "label", "wall_ms", "ram_bytes"]
            .iter()
            .any(|key| !obj.contains_key(*key))
    {
        return Err("native_process_unavailable");
    }
    let argv: Vec<String> = obj["argv"]
        .as_array()
        .ok_or("native_process_unavailable")?
        .iter()
        .map(|x| {
            x.as_str()
                .filter(|x| !x.contains('\0'))
                .map(str::to_owned)
                .ok_or("native_process_unavailable")
        })
        .collect::<Result<_>>()?;
    let root = PathBuf::from(obj["root"].as_str().ok_or("native_process_unavailable")?);
    let label = obj["label"].as_str().ok_or("native_process_unavailable")?;
    let wall_ms = obj["wall_ms"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("native_process_unavailable")?;
    let ram_bytes = obj["ram_bytes"]
        .as_u64()
        .filter(|n| *n > 0 && *n < i64::MAX as u64)
        .ok_or("native_process_unavailable")?;
    if argv.is_empty()
        || argv.len() > 4096
        || argv[0].is_empty()
        || !root.is_absolute()
        || label.is_empty()
        || label.len() > 128
        || !label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("native_process_unavailable");
    }
    Ok(Request {
        argv,
        root,
        label: label.to_owned(),
        wall_ms,
        ram_bytes,
    })
}

pub(super) struct Captures {
    pub directory: File,
    pub out: File,
    pub err: File,
    root: PathBuf,
    label: String,
}
impl Captures {
    pub fn create(r: &Request) -> Result<Self> {
        let path = std::ffi::CString::new(r.root.as_os_str().as_encoded_bytes())
            .map_err(|_| "native_output_unavailable")?;
        // SAFETY: CString is NUL-terminated; open returns an owned descriptor or error.
        let fd = unsafe {
            libc::open(
                path.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err("native_output_unavailable");
        }
        // SAFETY: successful open returned a new fd, transferred to exactly one File.
        let directory = unsafe { File::from_raw_fd(fd) };
        let meta = directory
            .metadata()
            .map_err(|_| "native_output_unavailable")?;
        // SAFETY: geteuid only reads this process identity.
        if meta.uid() != unsafe { libc::geteuid() } || meta.mode() & 0o077 != 0 {
            return Err("native_output_unavailable");
        }
        let open = |suffix: &str| -> Result<File> {
            let name = std::ffi::CString::new(format!("{}.{suffix}", r.label))
                .map_err(|_| "native_output_unavailable")?;
            // SAFETY: directory fd remains live; name is a single sanitized basename;
            // O_EXCL/NOFOLLOW prevents overwrite or final-symlink capture.
            let fd = unsafe {
                libc::openat(
                    directory.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDWR
                        | libc::O_CREAT
                        | libc::O_EXCL
                        | libc::O_NOFOLLOW
                        | libc::O_CLOEXEC,
                    0o600,
                )
            };
            if fd < 0 {
                return Err("native_output_unavailable");
            }
            // SAFETY: successful openat returned a new fd, owned exactly once.
            let f = unsafe { File::from_raw_fd(fd) };
            f.sync_all().map_err(|_| "native_output_unavailable")?;
            Ok(f)
        };
        let out = open("stdout")?;
        let err = open("stderr")?;
        directory
            .sync_all()
            .map_err(|_| "native_output_unavailable")?;
        Ok(Self {
            directory,
            out,
            err,
            root: r.root.clone(),
            label: r.label.clone(),
        })
    }
    fn validate_paths(&self) -> Result<()> {
        let path = std::ffi::CString::new(self.root.as_os_str().as_encoded_bytes())
            .map_err(|_| "native_file_unavailable")?;
        // SAFETY: the CString is live; flags refuse a final symlink/non-directory,
        // and nonblocking admission cannot wait on an attacker-supplied FIFO.
        let fd = unsafe {
            libc::open(
                path.as_ptr(),
                libc::O_RDONLY
                    | libc::O_DIRECTORY
                    | libc::O_NOFOLLOW
                    | libc::O_NONBLOCK
                    | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err("native_file_unavailable");
        }
        // SAFETY: successful open returned a new fd transferred to exactly one File.
        let current = unsafe { File::from_raw_fd(fd) };
        let directory = self
            .directory
            .metadata()
            .map_err(|_| "native_file_unavailable")?;
        let named = current.metadata().map_err(|_| "native_file_unavailable")?;
        if directory.dev() != named.dev()
            || directory.ino() != named.ino()
            || named.uid() != directory.uid()
            || named.mode() & 0o077 != 0
        {
            return Err("native_file_unavailable");
        }
        for (suffix, held) in [("stdout", &self.out), ("stderr", &self.err)] {
            let name = std::ffi::CString::new(format!("{}.{suffix}", self.label))
                .map_err(|_| "native_file_unavailable")?;
            // SAFETY: current is an admitted live directory fd, basename is sanitized,
            // and NOFOLLOW/NONBLOCK refuse unsafe final components without blocking.
            let fd = unsafe {
                libc::openat(
                    current.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
                )
            };
            if fd < 0 {
                return Err("native_file_unavailable");
            }
            // SAFETY: successful openat returned a newly owned fd, closed once by File.
            let capture = unsafe { File::from_raw_fd(fd) };
            let named = capture.metadata().map_err(|_| "native_file_unavailable")?;
            let original = held.metadata().map_err(|_| "native_file_unavailable")?;
            if !named.is_file()
                || named.nlink() != 1
                || named.len() > HASH_LIMIT
                || named.dev() != original.dev()
                || named.ino() != original.ino()
            {
                return Err("native_file_unavailable");
            }
        }
        Ok(())
    }
    pub fn hashes(&mut self) -> Result<(String, String)> {
        self.validate_paths()?;
        fn one(f: &mut File) -> Result<String> {
            f.sync_all().map_err(|_| "native_process_unavailable")?;
            let meta = f.metadata().map_err(|_| "native_process_unavailable")?;
            if !meta.is_file() || meta.nlink() != 1 || meta.len() > HASH_LIMIT {
                return Err("native_file_unavailable");
            }
            f.seek(SeekFrom::Start(0))
                .map_err(|_| "native_process_unavailable")?;
            let mut hash = Sha256::new();
            let mut bytes = 0u64;
            let mut buf = [0u8; 65536];
            loop {
                let n = f.read(&mut buf).map_err(|_| "native_process_unavailable")?;
                if n == 0 {
                    break;
                }
                bytes += n as u64;
                if bytes > HASH_LIMIT {
                    return Err("native_file_limit");
                }
                hash.update(&buf[..n]);
            }
            Ok(format!("{:x}", hash.finalize()))
        }
        let a = one(&mut self.out)?;
        let b = one(&mut self.err)?;
        self.directory
            .sync_all()
            .map_err(|_| "native_process_unavailable")?;
        self.validate_paths()?;
        Ok((a, b))
    }
}
