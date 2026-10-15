"""Synthetic prototypes. Encoder, decoder, init, and recipe bytes stay charged."""

from research.transfer.controls import ControlError, _fail
from research.transfer.evidence import EvidenceError, canonical_json, digest


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

    def _prototype_bytes(self, prototypes):
        try:
            return len(canonical_json({"prototypes": list(prototypes)}))
        except EvidenceError:
            _fail("invalid_prototypes")

    def _install(self, prototypes, dependencies):
        _fail("uncharged_dependency") if set(dependencies) != set(_DEPENDENCIES) else None
        charged = 0
        checked = {}
        for name in _DEPENDENCIES:
            amount = dependencies[name]
            _fail("invalid_bytes") if type(amount) is not int or amount < 0 else None
            charged += amount
            checked[name] = amount
        _fail("invalid_prototypes") if type(prototypes) is not tuple or not prototypes else None
        charged += self._prototype_bytes(prototypes)
        _fail("byte_budget_exceeded") if charged > self.byte_budget else None
        self.prototypes = prototypes
        self.dependencies = checked
        self.charged_bytes = charged
        self.prototype_sha256 = digest({"prototypes": list(prototypes), "dependencies": checked})
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
