"""Acceptance tests for source-family lineage, rights, and public provenance."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from itertools import permutations
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
    admit_source_manifest,
    assert_public_provenance,
    confirmation_allowed,
    load_registry,
    protocol_frozen,
    verify_source_projection,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "schemas" / "source-family-v1.schema.json"
SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
KEY = bytes(range(32))
MANIFEST_SCHEMA = ROOT / "schemas/source-manifest-v1.schema.json"


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


def private_manifest() -> dict:
    return {
        "format": "dope-source-manifest-v1", "version": 1,
        "custody": "development_only", "custody_hold": "D18_HOLD",
        "sources": [{
            "source_id": "private-source-canary", "family_id": "dev-lineage-alpha",
            "source_family": "dev-lineage-alpha", "cohort": "train", "use": "redistribute",
            "acquisition": {"uri": "file:///home/private/canary.csv", "method": "owner_delivery"},
            "version": "private-version-canary", "content_sha256": SHA256_EMPTY,
            "rights": {"license": "fixture-license-canary", "acquisition": "granted", "training": "granted",
                       "redistribution": "permitted_redistribution", "redistribution_approved": True,
                       "review_sha256": SHA256_EMPTY},
            "target": {"column": "private-target-canary", "definition": "private-outcome-canary"},
            "prediction_time": {"definition": "private-time-canary", "timestamp": "2026-01-01T00:00:00Z"},
            "hierarchy": {"group": "private-group-canary", "site": None,
                          "entity": "private-entity-canary", "time": "private-clock-canary"},
            "features": [{"column": "private-feature-canary", "feature_timestamp": "2025-12-31T23:00:00Z"}],
            "split": {"order": "entity_then_time_then_fit", "operations": ["entity", "time", "fit_transforms"],
                      "fit_transforms_cohort": "train"},
            "pretraining": [{"checkpoint": "private-kumo-canary", "checkpoint_sha256": SHA256_EMPTY,
                             "corpus_uri": None, "status": "unknown", "review_sha256": None,
                             "caveat": "private-corpus-canary"}],
        }],
    }


def admitted_registry() -> Registry:
    grants = RightsRecord(Permission.GRANTED, Permission.GRANTED, Permission.GRANTED)
    return Registry([
        family(rights=grants),
        family(family_id="dev-lineage-alpha-alias", relation=Relation.ALIAS,
               alias_of="dev-lineage-alpha", rights=grants),
        family(family_id="dev-lineage-beta", parent_lineage_id="dev-lineage-beta",
               split=split("part-beta"), rights=grants),
    ], final_label_custody="external")


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

    def test_alias_chains_resolve_in_any_id_order(self) -> None:
        parent = family()
        for first, second in (("dev-lineage-alpha-a1", "dev-lineage-alpha-a2"),
                              ("dev-lineage-alpha-a2", "dev-lineage-alpha-a1")):
            root = family(family_id=first, relation=Relation.ALIAS, alias_of="dev-lineage-alpha")
            chained = family(family_id=second, relation=Relation.ALIAS, alias_of=first)
            forward = Registry([parent, root, chained], final_label_custody="external")
            backward = Registry([chained, root, parent], final_label_custody="external")
            self.assertEqual(forward.independent_source_count(), 1)
            self.assertEqual(
                [item.family_id for item in forward.families()],
                ["dev-lineage-alpha", "dev-lineage-alpha-a1", "dev-lineage-alpha-a2"],
            )
            self.assertEqual(forward.families(), backward.families())

    def test_alias_cycles_missing_and_cross_parent_targets_are_refused(self) -> None:
        parent = family()
        other = family(family_id="dev-lineage-beta", parent_lineage_id="dev-lineage-beta", split=split("part-beta"))
        cases = (
            ("alias chain is cyclic", [
                family(family_id="dev-lineage-alpha-c1", relation=Relation.ALIAS, alias_of="dev-lineage-alpha-c2"),
                family(family_id="dev-lineage-alpha-c2", relation=Relation.ALIAS, alias_of="dev-lineage-alpha-c1"),
            ]),
            ("alias chain is cyclic", [
                family(family_id="dev-lineage-alpha-self", relation=Relation.ALIAS, alias_of="dev-lineage-alpha-self"),
            ]),
            ("alias target is missing", [
                family(family_id="dev-lineage-alpha-lost", relation=Relation.ALIAS, alias_of="dev-lineage-absent"),
            ]),
            ("alias target is outside the parent lineage", [
                family(family_id="dev-lineage-alpha-cross", relation=Relation.ALIAS, alias_of="dev-lineage-beta"),
            ]),
            ("alias target is outside the parent lineage", [
                family(family_id="dev-lineage-alpha-hop", relation=Relation.ALIAS, alias_of="dev-lineage-beta-alias"),
                family(family_id="dev-lineage-beta-alias", parent_lineage_id="dev-lineage-beta", split=split("part-beta"),
                       relation=Relation.ALIAS, alias_of="dev-lineage-beta"),
            ]),
            ("duplicate family id", [
                family(family_id="dev-lineage-alpha-twin", relation=Relation.ALIAS, alias_of="dev-lineage-alpha"),
                family(family_id="dev-lineage-alpha-twin", relation=Relation.ALIAS, alias_of="dev-lineage-alpha"),
            ]),
        )
        for message, aliases in cases:
            with self.assertRaises(LineageError) as refused:
                Registry([parent, other, *aliases], final_label_custody="external")
            self.assertEqual(str(refused.exception), message)
        registry = Registry([parent], final_label_custody="external")
        with self.assertRaises(LineageError) as single:
            registry.add(family(family_id="dev-lineage-alpha-late", relation=Relation.ALIAS, alias_of="dev-lineage-gone"))
        self.assertEqual(str(single.exception), "alias target is missing")

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

    def test_private_manifest_schema_is_closed_and_all_source_fields_required(self) -> None:
        schema = json.loads(MANIFEST_SCHEMA.read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
        validator.validate(private_manifest())
        for field in schema["$defs"]["source"]["required"]:
            document = private_manifest()
            del document["sources"][0][field]
            with self.subTest(field=field), self.assertRaisesRegex(LineageError, "source_manifest_schema_invalid"):
                admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        document = private_manifest()
        document["sources"][0]["target"]["raw_values"] = ["private-cell-canary"]
        with self.assertRaisesRegex(LineageError, "^source_manifest_schema_invalid$"):
            admit_source_manifest(document, admitted_registry(), signing_key=KEY)

    def test_split_order_and_train_only_transform_fit_are_mandatory(self) -> None:
        for order in permutations(["entity", "time", "fit_transforms"]):
            if order == ("entity", "time", "fit_transforms"):
                continue
            document = private_manifest()
            document["sources"][0]["split"]["operations"] = list(order)
            with self.assertRaisesRegex(LineageError, "source_manifest_schema_invalid"):
                admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        for field, value in (("order", "time_then_entity"), ("fit_transforms_cohort", "eval")):
            document = private_manifest()
            document["sources"][0]["split"][field] = value
            with self.assertRaises(LineageError):
                admit_source_manifest(document, admitted_registry(), signing_key=KEY)

    def test_alias_family_and_identical_content_cannot_cross_cohorts(self) -> None:
        for identifier in ("dev-lineage-alpha", "dev-lineage-alpha-alias", "dev-lineage-beta"):
            document = private_manifest()
            other = deepcopy(document["sources"][0])
            other.update(source_id="private-other", cohort="eval", family_id=identifier)
            if identifier == "dev-lineage-beta":
                other["source_family"] = identifier  # Same content, different registered parent.
            else:
                other["content_sha256"] = "b" * 64  # Different content, same canonical parent.
            document["sources"].append(other)
            with self.assertRaisesRegex(LineageError, "^source_family_cohort_leakage$"):
                admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        document["sources"][1]["content_sha256"] = "b" * 64
        self.assertEqual(len(admit_source_manifest(document, admitted_registry(), signing_key=KEY)["sources"]), 2)
        document["sources"][1]["source_family"] = "dev-lineage-alpha"
        with self.assertRaisesRegex(LineageError, "source_family_unregistered"):
            admit_source_manifest(document, admitted_registry(), signing_key=KEY)

    def test_feature_availability_compares_instants_and_excludes_target(self) -> None:
        for timestamp, allowed in (("2026-01-01T01:00:00+01:00", True),
                                   ("2026-01-01T00:00:01Z", False), ("2026-01-01", False)):
            document = private_manifest()
            document["sources"][0]["features"][0]["feature_timestamp"] = timestamp
            if allowed:
                admit_source_manifest(document, admitted_registry(), signing_key=KEY)
            else:
                with self.assertRaises(LineageError):
                    admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        document["sources"][0]["features"][0]["column"] = "private-target-canary"
        with self.assertRaisesRegex(LineageError, "source_feature_leakage"):
            admit_source_manifest(document, admitted_registry(), signing_key=KEY)

    def test_rights_admission_refuses_unlicensed_unknown_and_unapproved_use(self) -> None:
        for field, value in (("license", "unlicensed"), ("license", " UNKNOWN "),
                             ("training", "unknown"), ("acquisition", "denied"),
                             ("redistribution", "unknown"), ("redistribution_approved", False),
                             ("redistribution", "owner_only_custody"), ("redistribution", "prohibited")):
            document = private_manifest()
            document["sources"][0]["rights"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(LineageError):
                admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        with self.assertRaisesRegex(LineageError, "source_rights_unapproved"):
            admit_source_manifest(private_manifest(), load_registry(), signing_key=KEY)
        document["sources"][0]["use"] = "owner_only"
        document["sources"][0]["rights"].update(redistribution="owner_only_custody", redistribution_approved=False)
        result = admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        self.assertEqual(result["sources"][0]["redistribution"], "owner_only_custody")

    def test_checkpoint_overlap_unknown_and_reviewed_overlap_stay_caveated(self) -> None:
        for state in ("unknown", "overlapping", "known_none"):
            document = private_manifest()
            checkpoint = document["sources"][0]["pretraining"][0]
            checkpoint.update(status=state, corpus_uri="https://example.invalid/private-corpus", review_sha256=SHA256_EMPTY)
            result = admit_source_manifest(document, admitted_registry(), signing_key=KEY)
            source = result["sources"][0]
            self.assertEqual(source["pretraining"][0]["status"], state)
            self.assertEqual(source["pretraining"][0]["caveated"], state != "known_none")
            self.assertEqual(source["admission"], "development_only" if state == "known_none" else "development_only_caveated")
            self.assertFalse(result["confirmation_allowed"])
        checkpoint["review_sha256"] = None
        with self.assertRaisesRegex(LineageError, "source_manifest_schema_invalid"):
            admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        document = private_manifest()
        document["sources"][0]["pretraining"][0]["caveat"] = None
        with self.assertRaises(LineageError):
            admit_source_manifest(document, admitted_registry(), signing_key=KEY)

    def test_checkpoint_caveats_remain_separate_and_admission_does_not_mutate_input(self) -> None:
        document = private_manifest()
        document["sources"][0]["pretraining"].append({
            "checkpoint": "private-tabicl-canary", "checkpoint_sha256": "b" * 64,
            "corpus_uri": "https://example.invalid/corpus", "review_sha256": SHA256_EMPTY,
            "status": "known_none", "caveat": None,
        })
        original = deepcopy(document)
        result = admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        self.assertEqual(document, original)
        states = result["sources"][0]["pretraining"]
        self.assertEqual([item["caveated"] for item in states], [True, False])
        self.assertNotEqual(states[0]["checkpoint_token"], states[1]["checkpoint_token"])
        self.assertEqual(result["sources"][0]["admission"], "development_only_caveated")

    def test_duplicate_sources_and_unreadable_private_registry_errors_are_redacted(self) -> None:
        document = private_manifest()
        document["sources"].append(deepcopy(document["sources"][0]))
        with self.assertRaisesRegex(LineageError, "^source_duplicate_id$"):
            admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        with self.assertRaisesRegex(LineageError, "^source registry could not be read$"):
            load_registry(ROOT / "target/d02-validation/private-missing-canary.json")

    def test_projection_is_opaque_authenticated_and_bound_to_private_metadata(self) -> None:
        document = private_manifest()
        result = admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        self.assertEqual(verify_source_projection(result, signing_key=KEY), result)
        assert_public_provenance(result)
        for secret in ("canary", "dev-lineage", "/home/", "file:", "2026-01-01", SHA256_EMPTY):
            self.assertNotIn(secret, json.dumps(result))
        document["sources"][0]["version"] = "private-version-changed"
        changed = admit_source_manifest(document, admitted_registry(), signing_key=KEY)
        self.assertNotEqual(result["manifest_commitment"], changed["manifest_commitment"])
        self.assertNotEqual(result["signature"], changed["signature"])
        for tampered in (dict(result, confirmation_allowed=True), dict(result, private_path="/home/private"),
                         dict(result, signature="a" * 64)):
            with self.assertRaises(LineageError):
                verify_source_projection(tampered, signing_key=KEY)
        with self.assertRaisesRegex(LineageError, "source_projection_signature_invalid"):
            verify_source_projection(result, signing_key=KEY[::-1])
        with self.assertRaisesRegex(LineageError, "source_projection_key_invalid"):
            admit_source_manifest(document, admitted_registry(), signing_key=b"short")

    def test_custody_hold_and_public_redaction_cannot_be_overridden(self) -> None:
        for field, value in (("custody_hold", "released"), ("custody", "confirmation")):
            document = private_manifest()
            document[field] = value
            with self.assertRaises(LineageError) as error:
                admit_source_manifest(document, admitted_registry(), signing_key=KEY)
            self.assertEqual(str(error.exception), "source_manifest_schema_invalid")
            self.assertNotIn("canary", str(error.exception))
        for document in ({"sample_extrema": [0, 9]}, {"cell_values": ["private-cell-canary"]},
                         {"note": "local /home/private/canary.csv"}, {"note": "C:\\private\\canary.csv"}):
            with self.assertRaises(LineageError):
                assert_public_provenance(document)


if __name__ == "__main__":
    unittest.main()
