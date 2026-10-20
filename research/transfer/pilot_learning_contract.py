"""Immutable development-only learner inputs; no admission authority."""
from dataclasses import asdict, dataclass
import json
import math

from research.transfer.adapters.base import FeatureBatch, TrainingBatch
from research.transfer.evidence import digest
from .pilot_native_seed import NativeSeed

LEARNERS = ("kumo_tabular", "tabicl", "mitra", "gradient_boosting", "neural")
SCALES = ("primary_n", "sensitivity_4n")
CONTEXTS = ("standard", "stratified_anchor")
ARMS = ("real", "null", "synthetic")


def require(ok, code="pilot_learning_invalid"):
    if not ok:
        raise ValueError(code)


def pin(value):
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value))


def count(value, minimum=0):
    require(type(value) is int and minimum <= value <= (1 << 53) - 1)
    return value


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value))
    return float(value)


def identity(value):
    # Existing evidence digest uses lists, not Python tuples; preserve numeric types.
    return digest(json.loads(json.dumps(value, allow_nan=False)))


@dataclass(frozen=True, repr=False)
class Pool:
    batch: TrainingBatch
    source_role: str
    source_sha256: str
    artifact_sha256: str
    requested_rows: int
    produced_rows: int
    draw_seed: int

    def __post_init__(self):
        require(type(self.batch) is TrainingBatch)
        require(self.source_role in ("fit_real", "fit_null", "synthetic_draw"))
        require(self.batch.role == ("synthetic_train" if self.source_role == "synthetic_draw" else "development_train"))
        pin(self.source_sha256)
        pin(self.artifact_sha256)
        count(self.requested_rows, 1)
        require(count(self.produced_rows, 1) == len(self.batch.targets))
        count(self.draw_seed)
        for row in self.batch.features.rows:
            for value in row:
                finite(value)
        for value in self.batch.targets:
            finite(value)

    @property
    def sha256(self):
        return identity(asdict(self))


@dataclass(frozen=True)
class Context:
    name: str
    rows: int
    seed: int
    policy_sha256: str
    selections: tuple  # (scale, arm, immutable ordered row IDs), no label lookup here

    def __post_init__(self):
        require(self.name in CONTEXTS)
        count(self.rows, 1)
        count(self.seed)
        pin(self.policy_sha256)
        require(type(self.selections) is tuple and len(self.selections) == 6)
        require({(s, a) for s, a, _ in self.selections} == {(s, a) for s in SCALES for a in ARMS})
        for _, _, ids in self.selections:
            require(type(ids) is tuple and len(ids) == self.rows and len(set(ids)) == len(ids))
            for row_id in ids:
                pin(row_id)


@dataclass(frozen=True, repr=False)
class LearningPlan:
    task_id: str
    comparator: str
    candidate: str
    artifact_sha256: str
    input_sha256: str
    source_sha256: str
    environment_sha256: str
    policy_sha256: str
    owner_recipe_decision_sha256: str
    seed_derivation_sha256: str
    primary_n: int
    fit_seed: int
    predict_seed: int
    pools: tuple  # (scale, arm, Pool); real/null are actual fit-only pools
    contexts: tuple
    query: FeatureBatch
    model_retention: str
    null_reference_sha256: str
    native_seed: NativeSeed

    def __post_init__(self):
        pin(self.null_reference_sha256)
        require(type(self.native_seed) is NativeSeed)
        require(self.seed_derivation_sha256 == self.native_seed.sha256, "native_seed_binding_mismatch")
        require(self.fit_seed == self.predict_seed == self.native_seed.seed32, "native_seed_mismatch")
        require(self.model_retention in ("stream_drivers_then_release", "retain_for_separate_drivers"))
        for name in ("task_id", "artifact_sha256", "input_sha256", "source_sha256",
                     "environment_sha256", "policy_sha256", "owner_recipe_decision_sha256",
                     "seed_derivation_sha256"):
            pin(getattr(self, name))
        require(self.comparator in ("dope", "gaussian_copula", "compressed_real_rows"))
        require(type(self.candidate) is str and bool(self.candidate))
        count(self.primary_n, 1)
        # Owner must resolve native uint64 streams into explicit sklearn-compatible seeds.
        require(count(self.fit_seed) < 1 << 32 and count(self.predict_seed) < 1 << 32)
        require(type(self.query) is FeatureBatch and type(self.pools) is tuple and len(self.pools) == 6)
        require({(s, a) for s, a, _ in self.pools} == {(s, a) for s in SCALES for a in ARMS})
        require(type(self.contexts) is tuple and all(type(c) is Context for c in self.contexts)
                and tuple(c.name for c in self.contexts) == CONTEXTS)
        width = len(self.query.rows[0])
        for row in self.query.rows:
            for value in row:
                finite(value)
        for scale, arm, pool in self.pools:
            require(type(pool) is Pool and pool.source_role == {
                "real": "fit_real", "null": "fit_null", "synthetic": "synthetic_draw"}[arm])
            batch = pool.batch
            require(batch.features.schema_sha256 == self.query.schema_sha256)
            require(all(len(row) == width for row in batch.features.rows))
            require(not set(batch.features.row_ids) & set(self.query.row_ids))
            if arm == "synthetic":
                expected = self.primary_n * (4 if scale == "sensitivity_4n" else 1)
                require(pool.requested_rows == pool.produced_rows == expected)
                require(pool.artifact_sha256 == self.artifact_sha256)
            for context in self.contexts:
                ids = next(ids for s, a, ids in context.selections if (s, a) == (scale, arm))
                require(set(ids) <= set(batch.features.row_ids), "matched_context_unavailable")

        for scale in SCALES:
            real = next(p for s, a, p in self.pools if (s, a) == (scale, "real"))
            null = next(p for s, a, p in self.pools if (s, a) == (scale, "null"))
            require(real.batch == null.batch, "null_requires_original_real_training")
            require((real.source_sha256, real.artifact_sha256, real.requested_rows,
                     real.produced_rows, real.draw_seed) ==
                    (null.source_sha256, null.artifact_sha256, null.requested_rows,
                     null.produced_rows, null.draw_seed), "null_provenance_mismatch")
            for context in self.contexts:
                real_ids = next(ids for s, a, ids in context.selections if (s, a) == (scale, "real"))
                null_ids = next(ids for s, a, ids in context.selections if (s, a) == (scale, "null"))
                require(real_ids == null_ids, "null_context_order_mismatch")

    @property
    def sha256(self):
        return identity(asdict(self))

    def selected(self, scale, arm, context):
        pool = next(p for s, a, p in self.pools if (s, a) == (scale, arm))
        ids = next(ids for s, a, ids in context.selections if (s, a) == (scale, arm))
        positions = {row_id: i for i, row_id in enumerate(pool.batch.features.row_ids)}
        indices = [positions[row_id] for row_id in ids]
        batch = TrainingBatch(FeatureBatch(ids, tuple(pool.batch.features.rows[i] for i in indices),
                                          pool.batch.features.schema_sha256),
                              tuple(pool.batch.targets[i] for i in indices), pool.batch.role)
        return pool, batch


@dataclass(frozen=True, repr=False)
class QueryLabels:
    row_ids: tuple
    values: tuple
    source_sha256: str

    def __post_init__(self):
        pin(self.source_sha256)
        require(type(self.row_ids) is tuple and type(self.values) is tuple and
                len(self.row_ids) == len(self.values) > 0 and len(set(self.row_ids)) == len(self.row_ids))
        for row_id in self.row_ids:
            pin(row_id)
        for value in self.values:
            finite(value)

    @property
    def sha256(self):
        return identity(asdict(self))


@dataclass(frozen=True)
class DriverPlan:
    groups: tuple  # (opaque group ID, tuple of feature indices)
    repeats: int
    seed: int
    protocol_sha256: str

    def __post_init__(self):
        pin(self.protocol_sha256)
        count(self.repeats, 1)
        count(self.seed)
        require(type(self.groups) is tuple and bool(self.groups))
        require(len({name for name, _ in self.groups}) == len(self.groups))
        for name, indices in self.groups:
            pin(name)
            require(type(indices) is tuple and bool(indices) and len(set(indices)) == len(indices))
            for index in indices:
                count(index)

    @property
    def sha256(self):
        return identity(asdict(self))
