use serde::{Deserialize, Serialize};

use crate::corpus::{
    DatasetRecord, SplitManifest, ValidationSubmanifest, validate_split_manifest,
    validate_validation_submanifest,
};
use crate::error::{DopeError, Result};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AccessRole {
    Generator,
    SurfaceTrainer,
    GuardedEvaluator,
    FinalReleaseEvaluator,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccessBroker {
    pub manifest_checksum: String,
    pub role: AccessRole,
    pub final_release_open: bool,
}

impl AccessBroker {
    pub fn new(
        manifest: &SplitManifest,
        role: AccessRole,
        final_release_open: bool,
    ) -> Result<Self> {
        validate_split_manifest(manifest)?;
        if final_release_open && role != AccessRole::FinalReleaseEvaluator {
            return Err(DopeError::Data(
                "only the final release evaluator may open sealed lineages".into(),
            ));
        }
        Ok(Self {
            manifest_checksum: manifest.checksum.clone(),
            role,
            final_release_open,
        })
    }

    pub fn authorize(&self, record: &DatasetRecord, partition: &str) -> Result<()> {
        let sealed = record.split == "test";
        let allowed = match self.role {
            AccessRole::Generator | AccessRole::SurfaceTrainer => !sealed && partition == "train",
            AccessRole::GuardedEvaluator => !sealed && matches!(partition, "train" | "test"),
            AccessRole::FinalReleaseEvaluator => {
                (!sealed || self.final_release_open) && matches!(partition, "train" | "test")
            }
        };
        if allowed {
            Ok(())
        } else {
            Err(DopeError::Data(format!(
                "access denied for role {:?}, split {}, partition {partition}",
                self.role, record.split
            )))
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationAccessBroker {
    pub manifest_checksum: String,
    pub role: AccessRole,
    pub validation_cert_open: bool,
}

impl ValidationAccessBroker {
    pub fn new(
        manifest: &ValidationSubmanifest,
        role: AccessRole,
        validation_cert_open: bool,
    ) -> Result<Self> {
        validate_validation_submanifest(manifest)?;
        if validation_cert_open && role != AccessRole::FinalReleaseEvaluator {
            return Err(DopeError::Data(
                "only the guarded final evaluator may open validation-cert".into(),
            ));
        }
        Ok(Self {
            manifest_checksum: manifest.checksum.clone(),
            role,
            validation_cert_open,
        })
    }

    pub fn authorize_lineage(
        &self,
        manifest: &ValidationSubmanifest,
        lineage_group_id: &str,
    ) -> Result<()> {
        validate_validation_submanifest(manifest)?;
        if manifest.checksum != self.manifest_checksum {
            return Err(DopeError::Data(
                "validation access manifest changed after authorization".into(),
            ));
        }
        let partition = manifest
            .assignments
            .iter()
            .find(|assignment| assignment.lineage_group_id == lineage_group_id)
            .map(|assignment| assignment.partition.as_str())
            .ok_or_else(|| DopeError::Data("lineage is absent from validation manifest".into()))?;
        let allowed = match self.role {
            AccessRole::Generator | AccessRole::SurfaceTrainer => partition == "validation-select",
            AccessRole::GuardedEvaluator => partition == "validation-select",
            AccessRole::FinalReleaseEvaluator => {
                partition == "validation-select"
                    || (partition == "validation-cert" && self.validation_cert_open)
            }
        };
        if allowed {
            Ok(())
        } else {
            Err(DopeError::Data(format!(
                "access denied for role {:?} to {partition}",
                self.role
            )))
        }
    }
}

/// Strict descriptor-relative reads used by the custody boundary. Names and
/// symlinks cannot turn an authorized development file into a protected input.
pub(crate) fn read_regular(
    path: &std::path::Path,
    owner: Option<u32>,
    private: bool,
) -> Result<Vec<u8>> {
    use rustix::fs::{Mode, OFlags, open, openat};
    use std::io::Read;
    use std::os::unix::fs::MetadataExt;
    use std::path::Component;
    let denied = || DopeError::Data("custody_file_denied".into());
    let mut directory = open(
        if path.is_absolute() { "/" } else { "." },
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|_| denied())?;
    let parts = path
        .components()
        .filter(|c| !matches!(c, Component::RootDir | Component::CurDir))
        .collect::<Vec<_>>();
    let (last, parents) = parts.split_last().ok_or_else(denied)?;
    for part in parents {
        let Component::Normal(name) = part else {
            return Err(denied());
        };
        directory = openat(
            &directory,
            *name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|_| denied())?;
    }
    let Component::Normal(name) = last else {
        return Err(denied());
    };
    let descriptor = openat(
        &directory,
        *name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|_| denied())?;
    let file = std::fs::File::from(descriptor);
    let metadata = file.metadata().map_err(|_| denied())?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.len() > 16 * 1024 * 1024
        || owner.is_some_and(|uid| uid != metadata.uid())
        || (private && metadata.mode() & 0o077 != 0)
    {
        return Err(denied());
    }
    let mut bytes = Vec::new();
    file.take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| denied())?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(denied());
    }
    Ok(bytes)
}

pub(crate) fn effective_uid() -> u32 {
    // SAFETY: geteuid has no pointer arguments or memory effects.
    unsafe { libc::geteuid() }
}
#[cfg(target_os = "linux")]
pub(crate) fn peer_uid(stream: &std::os::unix::net::UnixStream) -> Result<u32> {
    use std::os::fd::AsRawFd;
    let mut credentials = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: credentials and length are valid writable buffers of the declared size.
    let status = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut credentials as *mut libc::ucred).cast(),
            &mut length,
        )
    };
    if status != 0 || length as usize != std::mem::size_of::<libc::ucred>() {
        return Err(DopeError::Data("custody_peer_denied".into()));
    }
    Ok(credentials.uid)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn peer_uid(_stream: &std::os::unix::net::UnixStream) -> Result<u32> {
    Err(DopeError::Unsupported(
        "custody_peer_credentials_unavailable".into(),
    ))
}

/// Dedicated Linux evaluator processes cannot open outbound sockets, connect to
/// an optimizer, or execute a replacement program. TSYNC covers existing threads.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn deny_evaluator_network() -> Result<()> {
    let instruction = |code, jt, jf, k| libc::sock_filter { code, jt, jf, k };
    let mut filter = vec![
        instruction(0x20, 0, 0, 4),          // seccomp_data.arch
        instruction(0x15, 1, 0, 0xc000003e), // AUDIT_ARCH_X86_64
        instruction(0x06, 0, 0, libc::SECCOMP_RET_KILL_PROCESS),
        instruction(0x20, 0, 0, 0),          // seccomp_data.nr
        instruction(0x35, 0, 1, 0x40000000), // reject x32 ABI syscall aliases
        instruction(0x06, 0, 0, libc::SECCOMP_RET_KILL_PROCESS),
    ];
    for syscall in [
        libc::SYS_socket,
        libc::SYS_connect,
        libc::SYS_execve,
        libc::SYS_execveat,
        libc::SYS_sendto,
        libc::SYS_sendmsg,
        libc::SYS_sendmmsg,
        libc::SYS_io_uring_setup,
        libc::SYS_io_uring_enter,
        libc::SYS_io_uring_register,
    ] {
        filter.push(instruction(0x15, 0, 1, syscall as u32));
        filter.push(instruction(
            0x06,
            0,
            0,
            libc::SECCOMP_RET_ERRNO | libc::EPERM as u32,
        ));
    }
    filter.push(instruction(0x06, 0, 0, libc::SECCOMP_RET_ALLOW));
    let program = libc::sock_fprog {
        len: filter.len() as u16,
        filter: filter.as_mut_ptr(),
    };
    // SAFETY: the kernel copies this complete BPF array during the syscall.
    let status = unsafe {
        if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
            -1
        } else {
            libc::syscall(
                libc::SYS_seccomp,
                libc::SECCOMP_SET_MODE_FILTER,
                libc::SECCOMP_FILTER_FLAG_TSYNC,
                &program,
            ) as i32
        }
    };
    if status != 0 {
        return Err(DopeError::Unsupported(
            "custody_network_isolation_unavailable".into(),
        ));
    }
    Ok(())
}
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub(crate) fn deny_evaluator_network() -> Result<()> {
    Err(DopeError::Unsupported(
        "custody_network_isolation_unavailable".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{ValidationAssignment, ValidationSubmanifest};

    fn manifest() -> ValidationSubmanifest {
        let mut manifest = ValidationSubmanifest {
            format: "dope-validation-submanifest".into(),
            version: 1,
            seed: 1729,
            source_manifest_checksum: "source".into(),
            selection_numerator: 3,
            selection_denominator: 5,
            validation_cert_sealed: true,
            validation_select_lineage_groups: 1,
            validation_cert_lineage_groups: 1,
            assignments: vec![
                ValidationAssignment {
                    dataset_id: "a".into(),
                    lineage_group_id: "select".into(),
                    partition: "validation-select".into(),
                },
                ValidationAssignment {
                    dataset_id: "b".into(),
                    lineage_group_id: "cert".into(),
                    partition: "validation-cert".into(),
                },
            ],
            checksum: String::new(),
        };
        // This mirrors the private canonical checksum rule used by corpus.
        manifest.checksum = blake3::hash(&serde_json::to_vec(&manifest).unwrap())
            .to_hex()
            .to_string();
        manifest
    }

    #[test]
    fn train_roles_cannot_open_validation_cert() {
        let manifest = manifest();
        let trainer =
            ValidationAccessBroker::new(&manifest, AccessRole::SurfaceTrainer, false).unwrap();
        assert!(trainer.authorize_lineage(&manifest, "select").is_ok());
        assert!(trainer.authorize_lineage(&manifest, "cert").is_err());
        assert!(ValidationAccessBroker::new(&manifest, AccessRole::SurfaceTrainer, true).is_err());
        let evaluator =
            ValidationAccessBroker::new(&manifest, AccessRole::FinalReleaseEvaluator, true)
                .unwrap();
        assert!(evaluator.authorize_lineage(&manifest, "cert").is_ok());
    }
}
