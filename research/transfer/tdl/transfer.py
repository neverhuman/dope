"""Paired original-unit losses and informative-task admission (T-030/T-036)."""
from .common import MetricError, failure, mean, number, require, vector


def evaluate(data):
    try:
        require(type(data) is dict)
        required = ("null_losses", "real_losses", "synthetic_losses", "real_gain_lcb", "constant_target")
        if any(data.get(key) is None for key in required):
            return failure("missing_transfer_cell", "incomplete")
        null = vector(data["null_losses"], nonnegative=True)
        real = vector(data["real_losses"], nonnegative=True)
        syn = vector(data["synthetic_losses"], nonnegative=True)
        require(len(null) == len(real) == len(syn), "unpaired_losses")
        require(type(data["constant_target"]) is bool, "invalid_constant_target")
        lcb = number(data["real_gain_lcb"])
        absolute = number(data.get("absolute_gain_floor", 1e-12))
        relative = number(data.get("relative_gain_floor", 0.01))
        require(absolute >= 0 and relative >= 0, "invalid_gain_floor")
        ln, lr, ls = mean(null), mean(real), mean(syn)
        gain = number(ln - lr)
        if data["constant_target"]:
            return failure("constant_target", "uninformative")
        if gain <= 0 or gain < max(absolute, number(relative * abs(ln))) or lcb <= 0:
            return failure("real_gain_not_informative", "uninformative")
        retention = number(number(ln - ls) / gain)
        return {"status": "informative", "reason": None, "null_loss": ln,
                "real_loss": lr, "synthetic_loss": ls, "real_gain": gain,
                "raw_transfer": retention, "clipped_transfer": min(1.0, max(0.0, retention))}
    except MetricError as error:
        return failure(str(error))
