//! Worker processes for the auditor-heavy subcommands.
//!
//! The native auditors run their inner loops on rayon's process-wide pool,
//! which this crate cannot size per thread. Worker threads inside one process
//! therefore queue on that pool, and a wider pool pays a hand-off per tree
//! node. With `--threads N` the subcommand instead starts `N` copies of this
//! executable, each owning every `N`-th cell and computing on one thread; the
//! parent supervises them from scoped threads and merges their outputs by
//! key, so the result is the same bytes for any `N`.

use super::par::par_map;
use dope_arena::{ArenaError, ArenaResult};
use std::ffi::OsString;
use std::process::{Command, Stdio};
use std::str::FromStr;

/// Environment variable read by rayon and by the tensor backend for their thread counts.
pub const INNER_THREADS_ENV: &str = "RAYON_NUM_THREADS";

/// Shard `index` of `count`: the cells at schedule positions `index, index + count, ...`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shard {
    pub index: usize,
    pub count: usize,
}

impl Shard {
    pub fn owns(self, position: usize) -> bool {
        position % self.count == self.index
    }
}

impl FromStr for Shard {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let parsed = text.split_once('/').and_then(|(index, count)| {
            Some(Shard {
                index: index.parse().ok()?,
                count: count.parse().ok()?,
            })
        });
        parsed
            .filter(|shard| shard.index < shard.count)
            .ok_or_else(|| "expected <index>/<count> with index < count".to_string())
    }
}

/// Positions of `0..len` owned by `shard` (all of them when there is none).
pub fn owned(shard: Option<Shard>, len: usize) -> Vec<usize> {
    (0..len)
        .filter(|&position| shard.is_none_or(|s| s.owns(position)))
        .collect()
}

/// Workers to start for `threads` requested threads and `cells` cells.
pub fn worker_count(threads: usize, cells: usize) -> usize {
    threads.min(cells).max(1)
}

/// Keep every computation of this process on the calling thread.
pub fn single_threaded_inner_pools() {
    std::env::set_var(INNER_THREADS_ENV, "1");
}

/// Run `count` copies of this executable, worker `i` with `args(i)` plus
/// `--threads 1 --shard i/count`, and fail if any of them fails.
pub fn run_workers<F>(count: usize, args: F) -> ArenaResult<()>
where
    F: Fn(usize) -> Vec<OsString> + Sync,
{
    let exe = std::env::current_exe()?;
    let outcomes = par_map(count, count, |index| {
        let output = Command::new(&exe)
            .args(args(index))
            .args(["--threads", "1", "--shard"])
            .arg(format!("{index}/{count}"))
            .env(INNER_THREADS_ENV, "1")
            .stdin(Stdio::null())
            .output()?;
        if output.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.lines().last().unwrap_or("no message").trim();
        Err(ArenaError::Contract(format!(
            "worker {index} of {count} failed: {detail}"
        )))
    });
    outcomes.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_partition_the_schedule() {
        let shard: Shard = "2/5".parse().unwrap();
        assert_eq!(shard, Shard { index: 2, count: 5 });
        assert_eq!(owned(Some(shard), 13), [2, 7, 12]);
        assert_eq!(owned(None, 3), [0, 1, 2]);
        let mut seen = [0; 23];
        for index in 0..4 {
            for position in owned(Some(Shard { index, count: 4 }), 23) {
                seen[position] += 1;
            }
        }
        assert!(seen.iter().all(|&count| count == 1));
        for bad in ["5/5", "1", "a/3", "1/0", "-1/3", "1/3/5"] {
            assert!(bad.parse::<Shard>().is_err(), "{bad}");
        }
        assert_eq!(worker_count(12, 5), 5);
        assert_eq!(worker_count(0, 5), 1);
        assert_eq!(worker_count(4, 0), 1);
    }
}
