"""Source-family lineage, rights, and split exposure.

The producer repository keeps a development inventory. Final source rows and
audit labels stay in external custody. Unknown rights or unknown exposure leave
the confirmation protocol unfrozen. Public provenance never carries row
identifiers, source headers, or low-entropy hashes.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from research.transfer.lineage_manifest import (
    LineageError,
    admit_source_manifest,
    verify_source_projection,
)

REGISTRY_FORMAT = "dope-source-family-registry-v1"
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "registries" / "sources.json"
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PUBLIC_KEYS = frozenset({
    "row_id",
    "row_ids",
    "header",
    "headers",
    "column_header",
    "column_headers",
    "columns",
    "field_names",
    "column_names",
    "sensitive_id",
    "sensitive_ids",
    "cell_values",
    "values",
    "sample_extrema",
    "extrema",
    "minimum",
    "maximum",
    "min",
    "max",
    "private_path",
    "source_path",
    "filesystem_path",
})
HASH_KEYS = frozenset({"sha256", "hash", "digest", "content_hash"})
MIN_PREIMAGE_BYTES = 16
MIN_HASH_ENTROPY_BITS = 3.0
PRIVATE_PATH = re.compile(r"(?:^|[\s\"'])(?:/(?:home|Users|root|tmp|var|etc|mnt|opt)/|file:|[A-Za-z]:[\\/])")


class Relation(str, Enum):
    ORIGINAL = "original"
    VARIANT = "variant"
    ALTERNATE_TARGET = "alternate_target"
    ALIAS = "alias"
    SIMULATED_DERIVATIVE = "simulated_derivative"
    REPEATED_ENTITY_EXTRACT = "repeated_entity_extract"


class Role(str, Enum):
    FIT = "fit"
    SELECTION = "selection"
    AUDIT = "audit"


class Suite(str, Enum):
    DEVELOPMENT = "development"
    HISTORICAL = "historical"
    CONFIRMATION = "confirmation"
    STRESS = "stress"


class Permission(str, Enum):
    GRANTED = "granted"
    DENIED = "denied"
    UNKNOWN = "unknown"


class ExposureState(str, Enum):
    CLEAN = "clean"
    EXPOSED = "exposed"
    UNKNOWN = "unknown"


PRODUCER_SUITES = frozenset({Suite.DEVELOPMENT, Suite.HISTORICAL})


def _enum(kind: type[Enum], value: object, label: str) -> Enum:
    if not isinstance(value, str):
        raise LineageError(f"{label} is not registered")
    try:
        return kind(value)
    except ValueError:
        raise LineageError(f"{label} is not registered") from None


def _entropy_per_char(text: str) -> float:
    counts: dict[str, int] = {}
    for character in text:
        counts[character] = counts.get(character, 0) + 1
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


@dataclass(frozen=True)
class RightsRecord:
    acquisition: Permission
    redistribution: Permission
    training: Permission

    def resolved(self) -> bool:
        return Permission.UNKNOWN not in (
            self.acquisition,
            self.redistribution,
            self.training,
        )

    @classmethod
    def from_json(cls, value: object) -> RightsRecord:
        if not isinstance(value, dict):
            raise LineageError("rights record is not registered")
        return cls(
            acquisition=_enum(Permission, value.get("acquisition"), "acquisition"),
            redistribution=_enum(Permission, value.get("redistribution"), "redistribution"),
            training=_enum(Permission, value.get("training"), "training"),
        )


@dataclass(frozen=True)
class ExposureStatus:
    prior_design: ExposureState
    model_contamination: ExposureState

    def status(self) -> ExposureState:
        axes = (self.prior_design, self.model_contamination)
        if ExposureState.UNKNOWN in axes:
            return ExposureState.UNKNOWN
        if ExposureState.EXPOSED in axes:
            return ExposureState.EXPOSED
        return ExposureState.CLEAN

    def is_clean(self) -> bool:
        return self.status() is ExposureState.CLEAN

    @classmethod
    def from_json(cls, value: object) -> ExposureStatus:
        if not isinstance(value, dict):
            raise LineageError("exposure record is not registered")
        return cls(
            prior_design=_enum(ExposureState, value.get("prior_design"), "prior_design"),
            model_contamination=_enum(
                ExposureState, value.get("model_contamination"), "model_contamination"
            ),
        )


@dataclass(frozen=True)
class SplitManifest:
    split_id: str
    kind: str
    rationale: str
    meta_partition: str
    fit_view: str
    selection_view: str
    audit_view: str
    labels_in_repository: bool

    def __post_init__(self) -> None:
        if self.labels_in_repository:
            raise LineageError("audit labels stay outside the producer repository")
        if self.audit_view != "external":
            raise LineageError("audit view custody must stay external")
        if self.fit_view != "producer" or self.selection_view != "producer":
            raise LineageError("fit and selection views belong to the producer")
        if self.kind not in {"entity", "time", "entity_time"}:
            raise LineageError("split kind is not registered")
        if not self.split_id or not self.rationale or not self.meta_partition:
            raise LineageError("split id, rationale, and meta partition are required")

    @classmethod
    def from_json(cls, value: object) -> SplitManifest:
        if not isinstance(value, dict):
            raise LineageError("split manifest is not registered")
        try:
            return cls(
                split_id=value["split_id"],
                kind=value["kind"],
                rationale=value["rationale"],
                meta_partition=value["meta_partition"],
                fit_view=value["fit_view"],
                selection_view=value["selection_view"],
                audit_view=value["audit_view"],
                labels_in_repository=value["labels_in_repository"],
            )
        except KeyError:
            raise LineageError("split manifest is missing a required field") from None


@dataclass(frozen=True)
class SourceFamily:
    family_id: str
    parent_lineage_id: str
    relation: Relation
    suite: Suite
    roles: tuple[Role, ...]
    rights: RightsRecord
    exposure: ExposureStatus
    split: SplitManifest
    alias_of: str | None = None

    def __post_init__(self) -> None:
        if not self.family_id or not self.parent_lineage_id:
            raise LineageError("family and parent ids are required")
        if not self.roles:
            raise LineageError("at least one role is required")
        if self.relation is Relation.ORIGINAL:
            if self.parent_lineage_id != self.family_id:
                raise LineageError("an original is its own parent lineage")
            if self.alias_of is not None:
                raise LineageError("an original is not an alias")
        elif self.parent_lineage_id == self.family_id:
            raise LineageError("a derived table needs a distinct parent lineage")
        if self.relation is Relation.ALIAS:
            if not self.alias_of:
                raise LineageError("an alias names the record it repeats")
        elif self.alias_of is not None:
            raise LineageError("only an alias names another record")

    @classmethod
    def from_json(cls, value: object) -> SourceFamily:
        if not isinstance(value, dict):
            raise LineageError("source family is not registered")
        roles_value = value.get("roles")
        if not isinstance(roles_value, list):
            raise LineageError("roles are not registered")
        roles = tuple(_enum(Role, item, "role") for item in roles_value)
        alias_of = value.get("alias_of")
        if alias_of is not None and not isinstance(alias_of, str):
            raise LineageError("alias target is not registered")
        try:
            return cls(
                family_id=value["family_id"],
                parent_lineage_id=value["parent_lineage_id"],
                relation=_enum(Relation, value.get("relation"), "relation"),
                suite=_enum(Suite, value.get("suite"), "suite"),
                roles=roles,
                rights=RightsRecord.from_json(value.get("rights")),
                exposure=ExposureStatus.from_json(value.get("exposure")),
                split=SplitManifest.from_json(value.get("split")),
                alias_of=alias_of,
            )
        except KeyError:
            raise LineageError("source family is missing a required field") from None


def protocol_frozen(family: SourceFamily) -> bool:
    """Unknown rights or unknown exposure keep this family's protocol unfrozen."""
    return family.rights.resolved() and family.exposure.status() is not ExposureState.UNKNOWN


def confirmation_allowed(family: SourceFamily) -> bool:
    """Confirmation needs a frozen record, clean exposure, and explicit grants."""
    if not protocol_frozen(family):
        return False
    if not family.exposure.is_clean():
        return False
    return (
        family.rights.acquisition is Permission.GRANTED
        and family.rights.redistribution is Permission.GRANTED
        and family.rights.training is Permission.GRANTED
    )


def independent_source_count(families: list[SourceFamily]) -> int:
    """Aliases and other derivatives share a parent and do not add a source."""
    return len({family.parent_lineage_id for family in families})


def _add_order(family: SourceFamily) -> tuple[int, str]:
    if family.relation is Relation.ORIGINAL:
        rank = 0
    elif family.relation is Relation.ALIAS:
        rank = 2
    else:
        rank = 1
    return rank, family.family_id


def _alias_order(aliases: list[SourceFamily], known: dict[str, SourceFamily]) -> list[SourceFamily]:
    """Order aliases so each one follows the record it repeats, whatever the id order.

    A chain may run through other aliases. It must end at a registered non-alias
    record; a missing end or a cycle is refused. Each hop is checked for the same
    parent lineage when the alias is added.
    """
    pending = {family.family_id: family for family in aliases}
    ordered: list[SourceFamily] = []
    placed: set[str] = set()
    for start in sorted(pending):
        chain: list[str] = []
        seen: set[str] = set()
        current = start
        while current in pending and current not in placed:
            if current in seen:
                raise LineageError("alias chain is cyclic")
            seen.add(current)
            chain.append(current)
            current = pending[current].alias_of or ""
        if current not in placed and current not in known:
            raise LineageError("alias target is missing")
        for family_id in reversed(chain):
            ordered.append(pending[family_id])
            placed.add(family_id)
    return ordered


class Registry:
    def __init__(self, families: list[SourceFamily], *, final_label_custody: str):
        if final_label_custody != "external":
            raise LineageError("final source and label custody stays external")
        self.final_label_custody = final_label_custody
        self._by_id: dict[str, SourceFamily] = {}
        identifiers = [family.family_id for family in families]
        if len(set(identifiers)) != len(identifiers):
            raise LineageError("duplicate family id")
        aliases = [family for family in families if family.relation is Relation.ALIAS]
        for family in sorted(families, key=_add_order):
            if family.relation is not Relation.ALIAS:
                self.add(family)
        for family in _alias_order(aliases, self._by_id):
            self.add(family)

    def add(self, family: SourceFamily) -> None:
        if family.family_id in self._by_id:
            raise LineageError("duplicate family id")
        if family.relation is not Relation.ORIGINAL:
            parent = self._by_id.get(family.parent_lineage_id)
            if parent is None or parent.relation is not Relation.ORIGINAL:
                raise LineageError("derived table parent lineage is missing")
            if (
                family.split.meta_partition != parent.split.meta_partition
                or family.suite is not parent.suite
            ):
                raise LineageError("derived tables cannot cross meta partitions")
        if family.relation is Relation.ALIAS:
            target = self._by_id.get(family.alias_of or "")
            if target is None:
                raise LineageError("alias target is missing")
            if target.parent_lineage_id != family.parent_lineage_id:
                raise LineageError("alias target is outside the parent lineage")
        self._by_id[family.family_id] = family

    def families(self) -> list[SourceFamily]:
        return sorted(self._by_id.values(), key=lambda item: item.family_id)

    def independent_source_count(self) -> int:
        return independent_source_count(self.families())

    def development_inventory(self) -> list[SourceFamily]:
        """Fit and selection views for historical or development suites.

        Audit-only records and confirmation or stress suites stay out. Final
        labels are not in this repository.
        """
        kept = [
            family
            for family in self.families()
            if family.suite in PRODUCER_SUITES
            and any(role in {Role.FIT, Role.SELECTION} for role in family.roles)
        ]
        return kept

    def protocol_frozen(self) -> bool:
        if not self._by_id:
            return False
        return all(protocol_frozen(family) for family in self._by_id.values())

    def confirmation_allowed(self) -> bool:
        if not self._by_id:
            return False
        return all(confirmation_allowed(family) for family in self._by_id.values())


def _reject_public(value: object) -> None:
    if isinstance(value, dict):
        preimage = value.get("preimage_bytes")
        if "preimage_bytes" in value and (
            isinstance(preimage, bool) or not isinstance(preimage, int)
        ):
            raise LineageError("public provenance discloses a low-entropy hash")
        for key, item in value.items():
            if not isinstance(key, str):
                raise LineageError("public provenance has an invalid field")
            lowered = key.lower().replace("-", "_")
            if lowered in FORBIDDEN_PUBLIC_KEYS:
                raise LineageError("public provenance discloses a forbidden field")
            if lowered in HASH_KEYS or lowered.endswith("_sha256"):
                weak = (
                    not isinstance(item, str)
                    or SHA256_HEX.fullmatch(item) is None
                    or _entropy_per_char(item) < MIN_HASH_ENTROPY_BITS
                    or (isinstance(preimage, int) and preimage < MIN_PREIMAGE_BYTES)
                )
                if weak:
                    raise LineageError("public provenance discloses a low-entropy hash")
            _reject_public(item)
    elif isinstance(value, list):
        for item in value:
            _reject_public(item)
    elif isinstance(value, str) and PRIVATE_PATH.search(value):
        raise LineageError("public provenance discloses a private path")


def assert_public_provenance(document: dict) -> dict:
    """Return the document when it is safe to publish. Refuse source leakage."""
    if not isinstance(document, dict):
        raise LineageError("public provenance must be an object")
    _reject_public(document)
    return document


def load_registry(path: Path | None = None) -> Registry:
    document_path = Path(path or DEFAULT_REGISTRY)
    try:
        document = json.loads(document_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        raise LineageError("source registry could not be read") from None
    if not isinstance(document, dict):
        raise LineageError("source registry is not an object")
    if document.get("format") != REGISTRY_FORMAT:
        raise LineageError("source registry format is not registered")
    if document.get("custody") != "development_only":
        raise LineageError("source registry must be a development-only inventory")
    rows = document.get("families")
    if not isinstance(rows, list) or not rows:
        raise LineageError("source registry has no families")
    families = [SourceFamily.from_json(row) for row in rows]
    if any(family.suite not in PRODUCER_SUITES for family in families):
        raise LineageError("development inventory cannot hold confirmation or stress sources")
    assert_public_provenance(document)
    return Registry(families, final_label_custody=str(document.get("final_label_custody")))
