//! Embedding tests: the 64-D embedding is the prefix of the 128-D one, it does
//! not depend on the column order of the table, and two disjoint halves of
//! one table are closer than any two unrelated tables.

use super::codec::unpack;
use super::embed::{cosine, embed, embed_summary, rotation, summary, EMBED_DIM, EMBED_PREFIX};
use super::fit::{fit_packet, Size, Target};
use super::glm::sigmoid;
use super::packet::{Level, Packet, Task, MAIN_BINS, MAX_RANK};
use super::testkit::{n01, table, u01};

/// One synthetic table: raw features, target and task.
struct Case {
    name: &'static str,
    raw: Vec<f64>,
    y: Vec<f64>,
    task: Task,
    k: u8,
    p: usize,
}

const ROWS: usize = 1200;

fn cases() -> Vec<Case> {
    let n = ROWS;
    let chain = |seed: u64, p: usize, rho: f64| {
        let mut raw = vec![0.0; n * p];
        for i in 0..n {
            raw[i * p] = n01(seed, 0, i);
            for j in 1..p {
                raw[i * p + j] =
                    rho * raw[i * p + j - 1] + (1.0 - rho * rho).sqrt() * n01(seed, j, i);
            }
        }
        raw
    };
    let mut out = Vec::new();
    // Correlated Gaussian chain, linear regression.
    let raw = chain(201, 14, 0.7);
    let y = raw
        .chunks(14)
        .enumerate()
        .map(|(i, r)| r[0] - r[5] + 0.5 * n01(202, 0, i))
        .collect();
    out.push(Case {
        name: "regression/chain",
        raw,
        y,
        task: Task::Regression,
        k: 0,
        p: 14,
    });
    // Independent skewed and discrete columns, weak regression.
    let raw = table(n, 14, |i, j| match j % 3 {
        0 => (0.8 * n01(203, j, i)).exp(),
        1 => (4.0 * u01(203, j, i)).floor(),
        _ => n01(203, j, i),
    });
    let y = raw
        .chunks(14)
        .enumerate()
        .map(|(i, r)| 0.3 * r[2] + n01(204, 0, i))
        .collect();
    out.push(Case {
        name: "regression/skewed",
        raw,
        y,
        task: Task::Regression,
        k: 0,
        p: 14,
    });
    // Binary, 20 % positives, one factor.
    let raw = table(n, 10, |i, j| 0.7 * n01(205, 50, i) + n01(205, j, i));
    let y = raw
        .chunks(10)
        .enumerate()
        .map(|(i, r)| (u01(206, 0, i) < sigmoid(1.5 * r[0] - r[1] - 1.8)) as u8 as f64)
        .collect();
    out.push(Case {
        name: "binary/factor",
        raw,
        y,
        task: Task::Binary,
        k: 2,
        p: 10,
    });
    // Four classes with separated means.
    let class = |i: usize| (4.0 * u01(207, 60, i)) as usize;
    let raw = table(n, 16, |i, j| {
        1.5 * ((j % 4 == class(i)) as u8 as f64) + n01(207, j, i)
    });
    let y = (0..n).map(|i| class(i) as f64).collect();
    out.push(Case {
        name: "multiclass/separated",
        raw,
        y,
        task: Task::Multiclass,
        k: 4,
        p: 16,
    });
    // Balanced binary on independent heavy-tailed columns.
    let raw = table(n, 10, |i, j| {
        n01(208, j, i) * (1.0 + 2.0 * (u01(208, 30 + j, i) < 0.1) as u8 as f64)
    });
    let y = raw
        .chunks(10)
        .enumerate()
        .map(|(i, r)| (u01(209, 0, i) < sigmoid(0.8 * r[3] + 0.8 * r[4])) as u8 as f64)
        .collect();
    out.push(Case {
        name: "binary/heavy-tailed",
        raw,
        y,
        task: Task::Binary,
        k: 2,
        p: 10,
    });
    out
}

/// Level-E packet of rows `from..to` of a case, with its columns in `order`.
fn fit_rows(case: &Case, from: usize, to: usize, order: &[usize]) -> Packet {
    let p = case.p;
    let raw: Vec<f64> = (from..to)
        .flat_map(|i| order.iter().map(move |&j| case.raw[i * p + j]))
        .collect();
    let y = &case.y[from..to];
    let ids: Vec<u16> = y.iter().map(|&v| v as u16).collect();
    let target = if case.task == Task::Regression {
        Target::Regression(y)
    } else {
        Target::Classes(&ids)
    };
    let fitted = fit_packet(
        &raw,
        target,
        case.task,
        case.k,
        Size::level(Level::E),
        to - from,
    )
    .unwrap();
    unpack(&fitted.bytes).unwrap()
}

#[test]
fn the_rotations_are_orthonormal_and_the_prefix_is_the_short_embedding() {
    for block in 0..2 {
        let r = rotation(block);
        for a in 0..EMBED_PREFIX {
            for b in 0..EMBED_PREFIX {
                let dot: f64 = (0..EMBED_PREFIX)
                    .map(|c| r[a * EMBED_PREFIX + c] * r[b * EMBED_PREFIX + c])
                    .sum();
                assert!((dot - (a == b) as u8 as f64).abs() < 1e-12);
            }
        }
    }
    assert_ne!(rotation(0), rotation(1));
    let case = &cases()[0];
    let identity: Vec<usize> = (0..case.p).collect();
    let packet = fit_rows(case, 0, ROWS, &identity);
    let full = embed(&packet);
    assert_eq!(full, embed(&packet));
    // The secondary half of the summary does not reach the first 64 codes,
    // and the primary half does not reach the last 64.
    let source = summary(&packet);
    assert!(source[..EMBED_PREFIX].iter().any(|&v| v != 0.0));
    assert!(source[EMBED_PREFIX..].iter().any(|&v| v != 0.0));
    let (mut primary, mut secondary) = (source, source);
    primary[EMBED_PREFIX..].fill(0.0);
    secondary[..EMBED_PREFIX].fill(0.0);
    assert_eq!(
        embed_summary(&primary)[..EMBED_PREFIX],
        full[..EMBED_PREFIX]
    );
    assert_eq!(
        embed_summary(&secondary)[EMBED_PREFIX..],
        full[EMBED_PREFIX..]
    );
    assert!(embed_summary(&primary)[EMBED_PREFIX..]
        .iter()
        .all(|&c| c == 0));
    // A lower level is another packet and has another embedding.
    assert_ne!(embed(&packet.at_level(Level::A).unwrap())[..], full[..]);
    assert!(
        full.iter().filter(|&&c| c.abs() == 127).count() <= 2,
        "codes saturate"
    );
}

/// The same packet with its columns reordered: every per-column field moves.
fn permute_packet(packet: &Packet, order: &[usize]) -> Packet {
    let (p, o) = (packet.cols(), packet.outputs());
    let mut out = packet.clone();
    let place: Vec<usize> = (0..p)
        .map(|j| order.iter().position(|&s| s == j).unwrap())
        .collect();
    for (new, &old) in order.iter().enumerate() {
        out.columns[new] = packet.columns[old].clone();
        out.target_mask[new] = packet.target_mask[old];
        out.main_mask[new] = packet.main_mask[old];
        out.loading[new * MAX_RANK..(new + 1) * MAX_RANK]
            .copy_from_slice(&packet.loading[old * MAX_RANK..(old + 1) * MAX_RANK]);
        let width = o * MAIN_BINS;
        out.main_effect[new * width..(new + 1) * width]
            .copy_from_slice(&packet.main_effect[old * width..(old + 1) * width]);
        for k in 0..o {
            out.target_weight[k * p + new] = packet.target_weight[k * p + old];
        }
    }
    let last = out.pairs.len() - 1;
    for term in &mut out.pairs[..last] {
        if term.coef.iter().any(|&c| c != 0) {
            term.a = place[term.a as usize] as u16;
            term.b = place[term.b as usize] as u16;
        }
    }
    out.validate().unwrap();
    out
}

#[test]
fn the_embedding_ignores_the_column_order() {
    let mut report = Vec::new();
    for case in cases() {
        let p = case.p;
        let identity: Vec<usize> = (0..p).collect();
        let order: Vec<usize> = (0..p).map(|j| (5 * j + 3) % p).collect();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        // 5 is coprime with every width used here only if the map is a permutation.
        let order = if sorted == identity {
            order
        } else {
            (0..p).rev().collect()
        };
        let packet = fit_rows(&case, 0, ROWS, &identity);
        // Exactly invariant for the same statistics in another column order.
        assert_eq!(
            embed(&permute_packet(&packet, &order)),
            embed(&packet),
            "{}",
            case.name
        );
        // Refitting the reordered table changes a few codes by rounding only.
        let refit = fit_rows(&case, 0, ROWS, &order);
        let (a, b) = (embed(&packet), embed(&refit));
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max()
            .unwrap();
        let cos = cosine(&a[..EMBED_PREFIX], &b[..EMBED_PREFIX]);
        report.push(format!(
            "{}: cosine64 {cos:.5}, largest code change {worst}",
            case.name
        ));
        assert!(cos > 0.995 && worst <= 8, "{}: {cos} {worst}", case.name);
    }
    println!("column permutation, refit:\n  {}", report.join("\n  "));
}

#[test]
fn two_halves_of_a_table_are_closer_than_unrelated_tables() {
    let all = cases();
    let halves: Vec<(Packet, Packet)> = all
        .iter()
        .map(|case| {
            let identity: Vec<usize> = (0..case.p).collect();
            (
                fit_rows(case, 0, ROWS / 2, &identity),
                fit_rows(case, ROWS / 2, ROWS, &identity),
            )
        })
        .collect();
    for dim in [EMBED_PREFIX, EMBED_DIM] {
        let code = |p: &Packet| embed(p)[..dim].to_vec();
        let same: Vec<f64> = halves
            .iter()
            .map(|(a, b)| cosine(&code(a), &code(b)))
            .collect();
        let mut other = Vec::new();
        for i in 0..halves.len() {
            for j in 0..halves.len() {
                if i != j {
                    other.push((cosine(&code(&halves[i].0), &code(&halves[j].1)), i, j));
                }
            }
        }
        let low = same.iter().cloned().fold(f64::INFINITY, f64::min);
        let (high, i, j) =
            other.iter().cloned().fold(
                (f64::NEG_INFINITY, 0, 0),
                |m, v| if v.0 > m.0 { v } else { m },
            );
        let mean_other = other.iter().map(|v| v.0).sum::<f64>() / other.len() as f64;
        let names: Vec<String> = all
            .iter()
            .zip(&same)
            .map(|(c, s)| format!("{} {s:.4}", c.name))
            .collect();
        println!(
            "embedding {dim}-D, halves of one table: {}\n  lowest {low:.4}; unrelated tables: mean {mean_other:.4}, highest {high:.4} ({} vs {})",
            names.join(", "),
            all[i].name,
            all[j].name
        );
        assert!(low > high, "{dim}-D: halves {low} vs unrelated {high}");
    }
}
