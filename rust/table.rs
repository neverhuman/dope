//! Header-aware table workflow.
//!
//! `fit_table` reads a headered table, infers each column's type, and encodes
//! it onto the unit interval with an invertible plan: numeric columns are
//! min-max scaled (heavily skewed ones on a log scale first, so 8-bit grid
//! codes keep resolution where the data are), categories become one code per
//! level ordered by the level's smoothed target statistic, dates become day
//! counts, and missing cells stay missing. The plan and a packed DPK3 kernel
//! form one artifact; `sample_table` decodes samples back to the source
//! header, categories and units.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DopeError, Result};
use crate::model::{Kernel, Task};

include!("table/plan.rs");
include!("table/encoding.rs");
include!("table/container.rs");
include!("table/workflow.rs");

#[cfg(test)]
mod tests;
