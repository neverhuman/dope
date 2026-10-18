use dope_kernel::DopeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Code {
    Invalid = 2,
    Unsupported = 3,
    Rights = 4,
    Budget = 5,
    Incomplete = 6,
    Science = 7,
    Privacy = 8,
    Internal = 70,
}

#[derive(Debug)]
pub struct Failure {
    pub code: Code,
    pub reason: &'static str,
}

pub type Result<T> = std::result::Result<T, Failure>;

impl Failure {
    pub fn new(code: Code, reason: &'static str) -> Self {
        Self { code, reason }
    }
    pub fn usage(reason: &'static str) -> Self {
        Self::new(Code::Invalid, reason)
    }
    pub fn budget() -> Self {
        Self::new(Code::Budget, "resource_budget_exceeded")
    }
    pub fn unsupported() -> Self {
        Self::new(Code::Unsupported, "capability_unavailable_in_this_build")
    }
}

impl From<DopeError> for Failure {
    fn from(error: DopeError) -> Self {
        // Only exact known machine codes cross this boundary. Never echo Display.
        match error {
            DopeError::Unsupported(_) => Self::unsupported(),
            DopeError::Io { .. } | DopeError::Sql(_) => {
                Self::new(Code::Internal, "local_operation_failed")
            }
            DopeError::Data(ref reason) | DopeError::Codec(ref reason) => match reason.as_str() {
                "duplicate_json_key" => Self::usage("duplicate_json_key"),
                "raw_complete_byte_cap_exceeded" | "mixed_complete_byte_cap_exceeded" => {
                    Self::budget()
                }
                "deadline expired before baseline encoding" => Self::budget(),
                reason if reason.starts_with("no encoded candidate fits the release policy;") => {
                    Self::budget()
                }
                _ => Self::usage("input_or_artifact_invalid"),
            },
            DopeError::Csv(_) | DopeError::Json(_) => Self::usage("input_or_artifact_invalid"),
        }
    }
}
