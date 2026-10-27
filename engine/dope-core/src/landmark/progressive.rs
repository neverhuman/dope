//! Cached progressive native landmarks: zero → +macro → +larger → +fine.
//!
//! Fold assignment, the full-panel work plan and the shared ridge/small fold
//! preparations are computed once and reused; every group evaluates only its
//! own not-yet-computed cells. The accumulated report reuses [`Signature`]
//! (no Serialize; use `private_report()`), and once all 23 cells have been
//! computed it is byte-identical to `signature()` on the same (subsampled)
//! frame and options: the full-panel plan decides over-budget cells exactly as
//! the full path does, and the feature spectrum is computed when the panel
//! completes (intermediate reports carry no spectrum).

use super::{
    bundle::{RowSelection, REPORT_RECIPE_OBSERVED},
    cost, folds,
    signature::{self, Fold},
    CellState, LandmarkBundle, LandmarkCell, LandmarkFrame, LandmarkGroup, LandmarkOptions,
    LandmarkScope, Signature, SubsampleSpec,
};
use crate::Result;
use rayon::prelude::*;

pub struct ProgressiveLandmarks {
    frame: LandmarkFrame,
    opts: LandmarkOptions,
    scope: LandmarkScope,
    selection: Option<RowSelection>,
    assigned: Vec<usize>,
    enabled: [bool; 23],
    costs: [u64; 23],
    plan: cost::WorkReport,
    ridge: Option<Vec<Fold>>,
    small: Option<Vec<Fold>>,
    computed: [bool; 23],
    spectrum_computed: bool,
    report: Signature,
    bundle: LandmarkBundle,
}

impl ProgressiveLandmarks {
    /// `frame` is the FULL training partition; `subsample` (if any) is applied
    /// here, keyed by the frame's physical row ids. Cheap: no fold is prepared
    /// until the first group. The scope binds the pre-subsample frame content
    /// (numeric frame-only shorthand; use [`Self::new_with_source_digest`] or
    /// [`Self::new_with_scope`] when binding a raw/prepared source digest).
    pub fn new(
        frame: &LandmarkFrame,
        opts: &LandmarkOptions,
        subsample: Option<SubsampleSpec>,
    ) -> Result<Self> {
        opts.validate()?;
        frame.validate()?;
        let scope = LandmarkScope::for_frame(frame, opts, subsample);
        Self::new_with_scope(frame, opts, scope)
    }

    /// Like [`Self::new`], but binds an explicit pre-subsample training source
    /// digest (such as `PreparedTable::source_rows_digest()` or a canonical
    /// training-cell digest including categorical/target vocabulary) into the
    /// progressive [`LandmarkScope`].
    pub fn new_with_source_digest(
        frame: &LandmarkFrame,
        opts: &LandmarkOptions,
        subsample: Option<SubsampleSpec>,
        source_rows_digest: impl Into<String>,
    ) -> Result<Self> {
        let scope = LandmarkScope::for_source_digest(opts, subsample, source_rows_digest);
        Self::new_with_scope(frame, opts, scope)
    }

    /// Construct a progressive driver bound to an explicit [`LandmarkScope`].
    pub fn new_with_scope(
        frame: &LandmarkFrame,
        opts: &LandmarkOptions,
        scope: LandmarkScope,
    ) -> Result<Self> {
        opts.validate()?;
        frame.validate()?;
        if scope.source_rows_digest.is_empty()
            || scope.source_rows_digest == super::bundle::ZERO_SOURCE_DIGEST
        {
            return Err(crate::DopeError::Input(
                "landmark scope source_rows_digest required".into(),
            ));
        }
        if scope.panel != opts.panel || scope.options_digest != super::bundle::options_digest(opts)
        {
            return Err(crate::DopeError::Input(
                "landmark scope does not match options".into(),
            ));
        }
        let subsample = scope.subsample;
        let (frame, selection) = match subsample {
            Some(spec) => {
                let (f, s) = frame.subsample(&spec)?;
                (f, Some(s))
            }
            None => (frame.clone(), None),
        };
        let assigned = folds::assign_folds(&frame, opts.folds, opts.seed)?;
        let (enabled, costs, plan) = cost::plan_requested_with_costs(&frame, opts, &[true; 23])?;
        let cells = (0..23)
            .map(|i| signature::idle_cell(i, CellState::NotRequested))
            .collect();
        let mut work = plan.clone();
        work.cells_run = 0;
        work.estimated_units = 0;
        let report = signature::assemble(&frame, opts, cells, Vec::new(), work);
        Ok(Self {
            frame,
            opts: opts.clone(),
            scope,
            selection,
            assigned,
            enabled,
            costs,
            plan,
            ridge: None,
            small: None,
            computed: [false; 23],
            spectrum_computed: false,
            report,
            bundle: LandmarkBundle::zero(),
        })
    }

    /// Compute one progressive group (no-op for cells already computed).
    pub fn next_group(&mut self, group: LandmarkGroup) -> Result<(LandmarkBundle, &Signature)> {
        let ids: Vec<u8> = group.ids();
        self.run_ids(&ids)
    }

    /// Compute arbitrary landmark ids (1..=23), reusing everything cached.
    pub fn run_ids(&mut self, ids: &[u8]) -> Result<(LandmarkBundle, &Signature)> {
        let requested = signature::requested_mask(Some(ids))?;
        let pending: Vec<usize> = (0..23)
            .filter(|&i| requested[i] && !self.computed[i])
            .collect();
        if pending.is_empty() {
            return Ok((self.bundle.clone(), &self.report));
        }
        if pending.iter().any(|&i| self.enabled[i] && i < 15) && self.ridge.is_none() {
            self.ridge = Some(signature::ridge_folds(
                &self.frame,
                &self.assigned,
                &self.opts,
            )?);
        }
        if pending
            .iter()
            .any(|&i| self.enabled[i] && (15..21).contains(&i))
            && self.small.is_none()
        {
            self.small = Some(signature::small_folds(
                &self.frame,
                &self.assigned,
                &self.opts,
            )?);
        }
        let ridge: &[Fold] = self.ridge.as_deref().unwrap_or(&[]);
        let small: &[Fold] = self.small.as_deref().unwrap_or(&[]);
        let (frame, assigned, opts, enabled) =
            (&self.frame, &self.assigned, &self.opts, &self.enabled);
        let fresh: Vec<(usize, LandmarkCell)> = pending
            .par_iter()
            .map(|&i| {
                let cell = if enabled[i] {
                    signature::run_cell(i, frame, assigned, opts, ridge, small)
                } else {
                    signature::idle_cell(i, CellState::OverBudget)
                };
                (i, cell)
            })
            .collect();
        let mut cells = std::mem::take(&mut self.report.cells);
        for (i, cell) in fresh {
            cells[i] = cell;
            self.computed[i] = true;
        }
        let mut work = self.plan.clone();
        work.cells_run = (0..23)
            .filter(|&i| self.computed[i] && self.enabled[i])
            .count();
        work.estimated_units = (0..23)
            .filter(|&i| self.computed[i] && self.enabled[i])
            .map(|i| self.costs[i])
            .fold(0u64, u64::saturating_add);
        let spectrum = if self.is_complete() && !self.spectrum_computed {
            if self.ridge.is_none() {
                self.ridge = Some(signature::ridge_folds(
                    &self.frame,
                    &self.assigned,
                    &self.opts,
                )?);
            }
            self.spectrum_computed = true;
            signature::spectrum(
                &self.frame,
                &self.opts,
                self.ridge.as_deref().unwrap_or(&[]),
                &mut work,
            )
        } else {
            std::mem::take(&mut self.report.spectrum)
        };
        self.report = signature::assemble(&self.frame, &self.opts, cells, spectrum, work);
        let incoming = LandmarkBundle::from_report_with_scope(
            &self.report.private_report(),
            self.scope.bundle_scope(),
            REPORT_RECIPE_OBSERVED,
        )?;
        self.bundle = if self.bundle.is_zero() {
            incoming
        } else {
            self.bundle.merge(&incoming)?
        };
        Ok((self.bundle.clone(), &self.report))
    }

    /// True once every panel cell has been computed (or ruled over budget).
    pub fn is_complete(&self) -> bool {
        self.computed.iter().all(|c| *c)
    }
    /// Accumulated report (unrequested cells are `NotRequested`).
    pub fn report(&self) -> &Signature {
        &self.report
    }
    /// Accumulated bundle (zero before the first group).
    pub fn bundle(&self) -> &LandmarkBundle {
        &self.bundle
    }
    pub fn scope(&self) -> &LandmarkScope {
        &self.scope
    }
    /// Selected physical rows when a subsample was requested.
    pub fn selection(&self) -> Option<&RowSelection> {
        self.selection.as_ref()
    }
    /// The (subsampled) frame actually measured.
    pub fn frame(&self) -> &LandmarkFrame {
        &self.frame
    }
}
