//! Stat packet v0: a zero-raw-row encoding of a table in at most 2 KiB that
//! needs no training, in the typed wire format `dope.single-encoding/v1`.
//!
//! The packet holds explicit, quantised sufficient statistics of the FIT rows
//! in unit space (see [`scale`]): class priors, per-column marginals (nested
//! smoothed knots or atom tables), a nested low-rank dependence model and, from
//! the target stage on, target coefficients. Synthetic rows are sampled from
//! the packet alone by a row-local analytic decoder ([`decode`]).
//!
//! Privacy by construction: no column minimum or maximum, no level carried by
//! fewer than 10 rows and no name or string is ever written. Errors never
//! contain data values.
//!
//! Normative text: `docs/plans/v2-gap-closure/single-encoding-v1.md`.

pub mod binary16;
pub mod codec;
pub mod decode;
pub mod dependence;
pub mod fit;
pub mod marginals;
pub mod matio;
pub mod packet;
pub mod scale;
pub mod sections;

#[cfg(test)]
mod testkit;
#[cfg(test)]
mod tests_codec;
#[cfg(test)]
mod tests_dependence;
#[cfg(test)]
mod tests_marginals;
#[cfg(test)]
mod tests_privacy;
#[cfg(test)]
mod tests_scale;
#[cfg(test)]
mod tests_wire;

/// Every failure of the module. Messages are static text plus section numbers.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StatError {
    #[error("single-encoding: bad magic")]
    BadMagic,
    #[error("single-encoding: unsupported version")]
    Version,
    #[error("single-encoding: unsupported kind")]
    Kind,
    #[error("single-encoding: decoder identity does not match this reader")]
    Decoder,
    #[error("single-encoding: file is truncated")]
    Truncated,
    #[error("single-encoding: trailing bytes after the checksum")]
    Trailing,
    #[error("single-encoding: checksum mismatch")]
    Crc,
    #[error("single-encoding: invalid header: {0}")]
    Header(&'static str),
    #[error("single-encoding: invalid section S{0}: {1}")]
    Section(u8, &'static str),
    #[error("stat packet: {0}")]
    Invalid(&'static str),
    #[error("stat packet: file error ({0:?})")]
    Io(std::io::ErrorKind),
}
