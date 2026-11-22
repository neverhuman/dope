//! Fixed reason codes of the external-command arm (`dope_kpi emit-cli`).
//!
//! A reason is the only explanation an `emit.jsonl` row carries: a short
//! constant chosen from this list. It never contains a cell value, a column
//! name, a path or anything the child process printed
//! (`docs/plans/v2-gap-closure/ext-arms.md` documents each code).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The schedule marks the cell `unsupported`; no command is run.
    ScheduleUnsupported,
    /// The run stopped before the cell was finished; `--resume` runs it again.
    RunInterrupted,
    /// The fit command could not be started.
    FitSpawn,
    /// The fit command exited with a non-zero status or was killed by a signal.
    FitExit,
    /// The fit command was still running at the timeout and was killed.
    FitTimeout,
    /// The fit command succeeded but `{packet}` is not a regular file.
    PacketMissing,
    /// The sample command could not be started.
    SampleSpawn,
    /// The sample command exited with a non-zero status or was killed by a signal.
    SampleExit,
    /// The sample command was still running at the timeout and was killed.
    SampleTimeout,
    /// The sample command succeeded but `{out}` is not a readable regular file.
    OutputMissing,
    /// `{out}` is larger than any valid output of the requested shape.
    OutputTooLarge,
    /// `io = "mat"`: `{out}` is not a well-formed `DOPEMAT1` container.
    MatInvalid,
    /// CSV header with a missing, extra or repeated column; `DOPEMAT1` with the wrong `x_cols`.
    HeaderMismatch,
    /// Malformed CSV, a record of the wrong width or a non-empty field that is not a number.
    ParseError,
    /// The output does not hold exactly `{rows}` rows.
    RowCountMismatch,
    /// A target is empty, NaN or infinite.
    NonFiniteTarget,
    /// A classification target is not an integer in `0..K-1`.
    LabelOutOfRange,
}

impl Reason {
    pub const ALL: [Reason; 17] = [
        Reason::ScheduleUnsupported,
        Reason::RunInterrupted,
        Reason::FitSpawn,
        Reason::FitExit,
        Reason::FitTimeout,
        Reason::PacketMissing,
        Reason::SampleSpawn,
        Reason::SampleExit,
        Reason::SampleTimeout,
        Reason::OutputMissing,
        Reason::OutputTooLarge,
        Reason::MatInvalid,
        Reason::HeaderMismatch,
        Reason::ParseError,
        Reason::RowCountMismatch,
        Reason::NonFiniteTarget,
        Reason::LabelOutOfRange,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Reason::ScheduleUnsupported => "schedule_unsupported",
            Reason::RunInterrupted => "run_interrupted",
            Reason::FitSpawn => "fit_spawn",
            Reason::FitExit => "fit_exit",
            Reason::FitTimeout => "fit_timeout",
            Reason::PacketMissing => "packet_missing",
            Reason::SampleSpawn => "sample_spawn",
            Reason::SampleExit => "sample_exit",
            Reason::SampleTimeout => "sample_timeout",
            Reason::OutputMissing => "output_missing",
            Reason::OutputTooLarge => "output_too_large",
            Reason::MatInvalid => "mat_invalid",
            Reason::HeaderMismatch => "header_mismatch",
            Reason::ParseError => "parse_error",
            Reason::RowCountMismatch => "row_count_mismatch",
            Reason::NonFiniteTarget => "non_finite_target",
            Reason::LabelOutOfRange => "label_out_of_range",
        }
    }

    pub fn from_code(code: &str) -> Option<Reason> {
        Reason::ALL.into_iter().find(|reason| reason.code() == code)
    }

    /// `emit.jsonl` status of a row that carries this reason.
    pub fn status(self) -> &'static str {
        match self {
            Reason::ScheduleUnsupported => "unsupported",
            _ => "failed",
        }
    }

    /// Whether the fit command had produced a packet when the row failed, so
    /// the row still reports the measured packet and fit time.
    pub fn after_fit(self) -> bool {
        !matches!(
            self,
            Reason::ScheduleUnsupported
                | Reason::RunInterrupted
                | Reason::FitSpawn
                | Reason::FitExit
                | Reason::FitTimeout
                | Reason::PacketMissing
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Reason;
    use std::collections::BTreeSet;

    #[test]
    fn codes_are_distinct_short_and_round_trip() {
        let codes: BTreeSet<&str> = Reason::ALL.iter().map(|reason| reason.code()).collect();
        assert_eq!(codes.len(), Reason::ALL.len());
        for reason in Reason::ALL {
            let code = reason.code();
            assert!(code.len() <= 24, "{code}");
            assert!(code.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'));
            assert_eq!(Reason::from_code(code), Some(reason));
        }
        assert_eq!(Reason::from_code(""), None);
        assert_eq!(Reason::from_code("fit_exit "), None);
    }

    #[test]
    fn only_the_schedule_code_is_unsupported_and_fit_codes_precede_the_packet() {
        for reason in Reason::ALL {
            let unsupported = reason == Reason::ScheduleUnsupported;
            assert_eq!(reason.status() == "unsupported", unsupported);
            assert_eq!(reason.status() == "failed", !unsupported);
        }
        let before: Vec<&str> = Reason::ALL
            .iter()
            .filter(|reason| !reason.after_fit())
            .map(|reason| reason.code())
            .collect();
        assert_eq!(
            before,
            [
                "schedule_unsupported",
                "run_interrupted",
                "fit_spawn",
                "fit_exit",
                "fit_timeout",
                "packet_missing"
            ]
        );
    }
}
