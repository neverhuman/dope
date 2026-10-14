//! Checked Gregorian RFC3339 instants, with Python datetime microsecond precision.
use super::{Refused, Result, require};
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid() -> Refused {
    Refused("invalid_timestamp")
}
fn digits(s: &str) -> Result<i64> {
    require(
        !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit()),
        "invalid_timestamp",
    )?;
    s.parse().map_err(|_| invalid())
}
fn days(y: i64, m: i64, d: i64) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yo = y - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    era * 146097 + yo * 365 + yo / 4 - yo / 100 + (153 * mp + 2) / 5 + d - 1 - 719468
}
pub fn parse(value: &str) -> Result<i64> {
    require(value.is_ascii() && value.len() >= 20, "invalid_timestamp")?;
    require(
        &value[4..5] == "-"
            && &value[7..8] == "-"
            && &value[10..11] == "T"
            && &value[13..14] == ":"
            && &value[16..17] == ":",
        "invalid_timestamp",
    )?;
    let y = digits(&value[..4])?;
    let m = digits(&value[5..7])?;
    let d = digits(&value[8..10])?;
    let h = digits(&value[11..13])?;
    let minute = digits(&value[14..16])?;
    let second = digits(&value[17..19])?;
    require(
        (1..=9999).contains(&y)
            && (1..=12).contains(&m)
            && (0..24).contains(&h)
            && (0..60).contains(&minute)
            && (0..60).contains(&second),
        "invalid_timestamp",
    )?;
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let max = match m {
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    require((1..=max).contains(&d), "invalid_timestamp")?;
    let suffix = &value[19..];
    let (fraction, zone) = if let Some(s) = suffix.strip_prefix('.') {
        let n = s.bytes().take_while(|c| c.is_ascii_digit()).count();
        require(n > 0, "invalid_timestamp")?;
        (&s[..n], &s[n..])
    } else {
        ("", suffix)
    };
    let micro = if fraction.is_empty() {
        0
    } else {
        digits(&fraction[..fraction.len().min(6)])? * 10_i64.pow(6 - fraction.len().min(6) as u32)
    };
    let offset = if zone == "Z" {
        0
    } else {
        require(
            zone.len() == 6
                && (zone.starts_with('+') || zone.starts_with('-'))
                && &zone[3..4] == ":",
            "invalid_timestamp",
        )?;
        let h = digits(&zone[1..3])?;
        let m = digits(&zone[4..6])?;
        require(h < 24 && m < 60, "invalid_timestamp")?;
        (h * 3600 + m * 60) * if zone.starts_with('-') { -1 } else { 1 }
    };
    Ok((days(y, m, d) * 86400 + h * 3600 + minute * 60 + second - offset) * 1_000_000 + micro)
}
pub fn now() -> Result<String> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid())?;
    let seconds = i64::try_from(elapsed.as_secs()).map_err(|_| invalid())?;
    let z = seconds / 86400 + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yo = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yo + era * 400;
    let doy = doe - (365 * yo + yo / 4 - yo / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let y = y + i64::from(m <= 2);
    Ok(format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:06}+00:00",
        seconds % 86400 / 3600,
        seconds % 3600 / 60,
        seconds % 60,
        elapsed.subsec_micros()
    ))
}
