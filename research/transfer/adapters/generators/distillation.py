"""Synthetic prototypes. Encoder, decoder, init, and recipe bytes stay charged."""

from types import MappingProxyType

from research.transfer.controls import ControlError, _fail
from research.transfer.evidence import EvidenceError, canonical_json, digest, parse_json


_DEPENDENCIES = ("encoder", "decoder", "init", "recipe")


class TabularCondensation:
    """Distilled prototypes are synthetic. Their training dependencies are not free."""

    def __init__(self, prototypes, dependencies, byte_budget, search_budget):
        _fail("invalid_budget") if type(byte_budget) is not int or type(search_budget) is not int else None
        self.byte_budget = byte_budget
        self.search_budget = search_budget
        self.prototypes = ()
        self.dependencies = {}
        self.charged_bytes = None
        self.prototype_sha256 = None
        self.retained_real_rows = False
        self.adaptations = ("synthetic_prototype",)
        self.valid = False
        self._install(prototypes, dependencies)

    def _install(self, prototypes, dependencies):
        _fail("uncharged_dependency") if set(dependencies) != set(_DEPENDENCIES) else None
        charged = 0
        checked = {}
        for name in _DEPENDENCIES:
            amount = dependencies[name]
            _fail("invalid_bytes") if type(amount) is not int or amount < 0 else None
            charged += amount
            checked[name] = amount
        snapshot, encoded = _accepted_snapshot(prototypes)
        charged += len(encoded)
        _fail("byte_budget_exceeded") if charged > self.byte_budget else None
        retained = [_thaw(item) for item in snapshot]
        self.prototypes = snapshot
        self.dependencies = checked
        self.charged_bytes = charged
        self.prototype_sha256 = digest({"prototypes": retained, "dependencies": checked})
        self.valid = True

    def refit(self, prototypes, dependencies=None):
        """Drop the prior prototypes before installing a new payload.

        A refused payload leaves no prior hash and no prior prototype tuple.
        """
        kept = self.dependencies if dependencies is None else dependencies
        self.prototypes = ()
        self.prototype_sha256 = None
        self.charged_bytes = None
        self.valid = False
        self._install(prototypes, kept)


def _freeze(value):
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return value


def _thaw(value):
    if type(value) is tuple:
        return [_thaw(item) for item in value]
    if type(value) is MappingProxyType:
        return {key: _thaw(item) for key, item in value.items()}
    return value


def _accepted_snapshot(prototypes):
    """Copy the accepted payload into containers the caller does not own."""
    _fail("invalid_prototypes") if type(prototypes) is not tuple or not prototypes else None
    try:
        encoded = canonical_json({"prototypes": list(prototypes)})
        parsed = parse_json(encoded)["prototypes"]
    except EvidenceError:
        _fail("invalid_prototypes")
    snapshot = tuple(_freeze(item) for item in parsed)
    if canonical_json({"prototypes": [_thaw(item) for item in snapshot]}) != encoded:
        _fail("invalid_prototypes")
    return snapshot, encoded
