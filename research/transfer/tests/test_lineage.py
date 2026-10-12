"""Acceptance tests for source-family lineage, rights, and public provenance."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import jsonschema

from research.transfer.lineage import (
    DEFAULT_REGISTRY,
    ExposureState,
    ExposureStatus,
    LineageError,
    Permission,
    Registry,
    Relation,
    RightsRecord,
    Role,
    SourceFamily,
    SplitManifest,
    Suite,
    assert_public_provenance,
    confirmation_allowed,
    load_registry,
    protocol_frozen,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "schemas" / "source-family-v1.schema.json"
SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def split(partition: str = "part-alpha") -> SplitManifest:
    return SplitManifest(
        split_id=partition,
        kind="entity_time",
        rationale="Entity and time grain for this parent. Audit labels stay external.",
        meta_partition=partition,
        fit_view="producer",
        selection_view="producer",
        audit_view="external",
        labels_in_repository=False,
    )


def family(**overrides: object) -> SourceFamily:
    payload: dict[str, object] = {
        "family_id": "dev-lineage-alpha",
        "parent_lineage_id": "dev-lineage-alpha",
        "relation": Relation.ORIGINAL,
        "suite": Suite.DEVELOPMENT,
        "roles": (Role.FIT, Role.SELECTION),
        "rights": RightsRecord(Permission.UNKNOWN, Permission.UNKNOWN, Permission.UNKNOWN),
        "exposure": ExposureStatus(ExposureState.UNKNOWN, ExposureState.UNKNOWN),
        "split": split(),
        "alias_of": None,
    }
    payload.update(overrides)
    return SourceFamily(**payload)  # type: ignore[arg-type]


class LineageTests(unittest.TestCase):
    def test_aliases_do_not_increase_independent_source_count(self) -> None:
        parent = family()
        alias = family(
            family_id="dev-lineage-alpha-alias",
            relation=Relation.ALIAS,
            alias_of="dev-lineage-alpha",
        )
        variant = family(
            family_id="dev-lineage-alpha-variant",
            relation=Relation.VARIANT,
        )
        registry = Registry([parent, alias, variant], final_label_custody="external")
        self.assertEqual(registry.independent_source_count(), 1)
        registry.add(family(family_id="dev-lineage-beta", parent_lineage_id="dev-lineage-beta", split=split("part-beta")))
        self.assertEqual(registry.independent_source_count(), 2)
        self.assertEqual(
            registry.independent_source_count(),
            sum(item.relation is Relation.ORIGINAL for item in registry.families()),
        )

    def test_derived_tables_cannot_cross_meta_partitions(self) -> None:
        registry = Registry([family()], final_label_custody="external")
        crossed = family(
            family_id="dev-lineage-alpha-variant",
            relation=Relation.VARIANT,
            split=split("part-other"),
        )
        with self.assertRaisesRegex(LineageError, "cannot cross meta partitions"):
            registry.add(crossed)
        renamed = family(
            family_id="dev-lineage-alpha-sim",
            relation=Relation.SIMULATED_DERIVATIVE,
            suite=Suite.CONFIRMATION,
        )
        with self.assertRaisesRegex(LineageError, "cannot cross meta partitions"):
            registry.add(renamed)

    def test_unknown_exposure_is_never_clean(self) -> None:
        unknown = ExposureStatus(ExposureState.UNKNOWN, ExposureState.CLEAN)
        self.assertIs(unknown.status(), ExposureState.UNKNOWN)
        self.assertFalse(unknown.is_clean())
        contaminated = ExposureStatus(ExposureState.CLEAN, ExposureState.EXPOSED)
        self.assertIs(contaminated.status(), ExposureState.EXPOSED)
        self.assertFalse(contaminated.is_clean())
        self.assertTrue(ExposureStatus(ExposureState.CLEAN, ExposureState.CLEAN).is_clean())

    def test_unknown_rights_leave_the_protocol_unfrozen(self) -> None:
        record = family()
        self.assertFalse(record.rights.resolved())
        self.assertFalse(protocol_frozen(record))
        self.assertFalse(confirmation_allowed(record))
        granted = family(
            rights=RightsRecord(Permission.GRANTED, Permission.GRANTED, Permission.UNKNOWN),
            exposure=ExposureStatus(ExposureState.CLEAN, ExposureState.CLEAN),
        )
        self.assertFalse(protocol_frozen(granted))
        self.assertFalse(confirmation_allowed(granted))
        ready = family(
            rights=RightsRecord(Permission.GRANTED, Permission.GRANTED, Permission.GRANTED),
            exposure=ExposureStatus(ExposureState.CLEAN, ExposureState.CLEAN),
        )
        self.assertTrue(protocol_frozen(ready))
        self.assertTrue(confirmation_allowed(ready))

    def test_public_provenance_hides_row_ids_headers_and_weak_hashes(self) -> None:
        with self.assertRaisesRegex(LineageError, "forbidden field"):
            assert_public_provenance({"row_id": "entity-kept-out"})
        with self.assertRaisesRegex(LineageError, "forbidden field"):
            assert_public_provenance({"note": "scope", "headers": ["kept-out"]})
        with self.assertRaisesRegex(LineageError, "low-entropy hash"):
            assert_public_provenance({"sha256": "abcd"})
        with self.assertRaisesRegex(LineageError, "low-entropy hash"):
            assert_public_provenance({"sha256": "a" * 64})
        with self.assertRaisesRegex(LineageError, "low-entropy hash"):
            assert_public_provenance({"sha256": SHA256_EMPTY, "preimage_bytes": 4})
        safe = {"family_id": "dev-lineage-alpha", "sha256": SHA256_EMPTY}
        self.assertEqual(assert_public_provenance(safe)["family_id"], "dev-lineage-alpha")

    def test_loaded_registry_is_development_only_and_unfrozen(self) -> None:
        document = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        jsonschema.validate(document, json.loads(SCHEMA.read_text(encoding="utf-8")))
        assert_public_provenance(document)
        registry = load_registry()
        self.assertFalse(registry.protocol_frozen())
        self.assertFalse(registry.confirmation_allowed())
        self.assertEqual(registry.final_label_custody, "external")
        self.assertEqual(registry.independent_source_count(), 2)
        relations = {item.relation for item in registry.families()}
        self.assertEqual(
            relations,
            {
                Relation.ORIGINAL,
                Relation.VARIANT,
                Relation.ALTERNATE_TARGET,
                Relation.ALIAS,
                Relation.SIMULATED_DERIVATIVE,
                Relation.REPEATED_ENTITY_EXTRACT,
            },
        )
        inventory_ids = [item.family_id for item in registry.development_inventory()]
        self.assertIn("dev-lineage-alpha-alias", inventory_ids)
        self.assertNotIn("dev-lineage-alpha-repeat", inventory_ids)
        self.assertTrue(all(item.suite is Suite.DEVELOPMENT for item in registry.families()))
        self.assertTrue(all(not item.exposure.is_clean() for item in registry.families()))

    def test_confirmation_sources_and_local_labels_stay_out(self) -> None:
        parent = family()
        confirmation = family(
            family_id="confirm-lineage-gamma",
            parent_lineage_id="confirm-lineage-gamma",
            suite=Suite.CONFIRMATION,
            split=split("part-gamma"),
        )
        registry = Registry([parent, confirmation], final_label_custody="external")
        self.assertEqual(
            [item.family_id for item in registry.development_inventory()],
            ["dev-lineage-alpha"],
        )
        with self.assertRaisesRegex(LineageError, "outside the producer repository"):
            SplitManifest(
                split_id="part-alpha",
                kind="entity_time",
                rationale="labels do not belong here",
                meta_partition="part-alpha",
                fit_view="producer",
                selection_view="producer",
                audit_view="external",
                labels_in_repository=True,
            )
        with self.assertRaisesRegex(LineageError, "custody stays external"):
            Registry([parent], final_label_custody="local")
        mixed = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        mixed["families"].append(
            {
                "family_id": "confirm-lineage-gamma",
                "parent_lineage_id": "confirm-lineage-gamma",
                "relation": "original",
                "suite": "confirmation",
                "roles": ["fit"],
                "rights": {
                    "acquisition": "unknown",
                    "redistribution": "unknown",
                    "training": "unknown",
                },
                "exposure": {"prior_design": "unknown", "model_contamination": "unknown"},
                "split": {
                    "split_id": "part-gamma",
                    "kind": "entity_time",
                    "rationale": "Fresh confirmation custody is not a development inventory row.",
                    "meta_partition": "part-gamma",
                    "fit_view": "producer",
                    "selection_view": "producer",
                    "audit_view": "external",
                    "labels_in_repository": False,
                },
                "alias_of": None,
            }
        )
        path = ROOT / "target" / "vnext" / "mixed-registry.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mixed), encoding="utf-8")
        with self.assertRaisesRegex(LineageError, "development inventory"):
            load_registry(path)


if __name__ == "__main__":
    unittest.main()
