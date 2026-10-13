"""Synthetic prototypes. Encoder, decoder, init, and recipe bytes stay charged."""

from research.transfer.controls import ControlError, _fail
from research.transfer.evidence import digest


_DEPENDENCIES = ("encoder", "decoder", "init", "recipe")
_PROTOTYPE_BYTES = 8


class TabularCondensation:
    """Distilled prototypes are synthetic. Their training dependencies are not free."""

    def __init__(self, prototypes, dependencies, byte_budget, search_budget):
        _fail("invalid_budget") if type(byte_budget) is not int or type(search_budget) is not int else None
        _fail("uncharged_dependency") if set(dependencies) != set(_DEPENDENCIES) else None
        charged = 0
        for name in _DEPENDENCIES:
            amount = dependencies[name]
            _fail("invalid_bytes") if type(amount) is not int or amount < 0 else None
            charged += amount
        _fail("invalid_prototypes") if type(prototypes) is not tuple or not prototypes else None
        charged += _PROTOTYPE_BYTES * len(prototypes)
        _fail("byte_budget_exceeded") if charged > byte_budget else None
        self.prototypes = prototypes
        self.dependencies = dict(dependencies)
        self.charged_bytes = charged
        self.byte_budget = byte_budget
        self.search_budget = search_budget
        self.retained_real_rows = False
        self.adaptations = ("synthetic_prototype",)
        self.prototype_sha256 = digest({"prototypes": list(prototypes), "dependencies": self.dependencies})
