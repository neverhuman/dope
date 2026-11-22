//! Order-preserving parallel map on scoped threads (standard library only).

use std::sync::atomic::{AtomicUsize, Ordering};

/// Evaluate `work(0..count)` on up to `threads` workers and return the results
/// in index order, so the output never depends on scheduling.
pub fn par_map<R, F>(threads: usize, count: usize, work: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
{
    let workers = threads.max(1).min(count.max(1));
    if workers == 1 {
        return (0..count).map(work).collect();
    }
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, R)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= count {
                            break;
                        }
                        done.push((index, work(index)));
                    }
                    done
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| match handle.join() {
                Ok(done) => done,
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    });
    let mut slots: Vec<Option<R>> = (0..count).map(|_| None).collect();
    for (index, value) in parts.into_iter().flatten() {
        slots[index] = Some(value);
    }
    slots.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::par_map;

    #[test]
    fn results_keep_index_order_for_any_worker_count() {
        let serial = par_map(1, 257, |i| i * i + 1);
        for threads in [2, 4, 9, 400] {
            assert_eq!(par_map(threads, 257, |i| i * i + 1), serial);
        }
        assert_eq!(serial.len(), 257);
        assert_eq!(serial[16], 257);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(par_map(4, 0, |i| i).is_empty());
    }
}
