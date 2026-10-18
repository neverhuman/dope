"""Paired original-unit losses and informative-task admission (T-030/T-036)."""
import math

from .common import MetricError, count, failure, mean, number, opt_default, require, sum_checked, vector


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
        absolute = number(opt_default(data, "absolute_gain_floor", 1e-12))
        relative = number(opt_default(data, "relative_gain_floor", 0.01))
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


# Fixed conformance policy: validate the full row, clip every class, renormalize.
PROBABILITY_CLIP = 1e-15
PROBABILITY_SUM_TOLERANCE = 1e-12


def query_harm(ls, lr, l0, tau=0.90):
    ls, lr, l0, tau = map(number, (ls, lr, l0, tau))
    require(min(ls, lr, l0) >= 0 and 0 <= tau <= 1, "invalid_paired_loss_evidence")
    return number(ls - number(tau * lr + (1 - tau) * l0))


def query_benefit(ls, lr, l0, tau=0.90):
    return -query_harm(ls, lr, l0, tau)


def _prediction_losses(data):
    task = data.get("task")
    if task == "paired_losses":
        require(data.get("loss_space") in ("original_mse", "discrete_log_loss"), "invalid_loss_space")
        return [vector(data.get(key), nonnegative=True)
                for key in ("real_losses", "synthetic_losses", "null_losses")], data["loss_space"]
    if task == "regression":
        targets = vector(data.get("targets"))
        offset, scale = number(data.get("prediction_offset")), number(data.get("prediction_scale"))
        require(scale > 0, "invalid_paired_loss_evidence")
        def losses(key):
            predictions = vector(data.get(key))
            require(len(predictions) == len(targets), "invalid_paired_loss_evidence")
            result = []
            for y, p in zip(targets, predictions):
                residual = number(y - number(number(p * scale) + offset))
                result.append(number(residual * residual))
            return result
        space = "original_mse"
    else:
        require(task in ("binary", "multiclass"), "invalid_task")
        order = data.get("class_order")
        require(type(order) is list and len(order) >= 2 and
                all(type(label) is str and label for label in order) and
                len(set(order)) == len(order) and (task != "binary" or len(order) == 2), "invalid_class_order")
        targets = data.get("targets")
        require(type(targets) is list, "invalid_targets")
        targets = [count(y) for y in targets]
        require(all(y < len(order) for y in targets), "invalid_paired_loss_evidence")
        def losses(key):
            predictions = data.get(key)
            require(type(predictions) is list and len(predictions) == len(targets), "invalid_probabilities")
            result = []
            for y, row in zip(targets, predictions):
                row = vector(row, nonnegative=True)
                require(len(row) == len(order) and all(p <= 1 for p in row) and
                        abs(sum_checked(row) - 1) <= PROBABILITY_SUM_TOLERANCE, "invalid_paired_loss_evidence")
                clipped = [min(1 - PROBABILITY_CLIP, max(PROBABILITY_CLIP, p)) for p in row]
                result.append(number(-math.log(clipped[y] / sum_checked(clipped))))
            return result
        space = "discrete_log_loss"
    return [losses(key) for key in ("real_predictions", "synthetic_predictions", "null_predictions")], space


def evaluate_tdl(data):
    """Candidate diagnostics only; point margins cannot establish qualification."""
    try:
        require(type(data) is dict)
        present = data.get("required_model_present")
        if present is None or present is False:
            return failure("missing_required_model", "incomplete")
        require(present is True, "invalid_required_model")
        tau = number(data.get("tau", .90))
        require(0 <= tau <= 1, "invalid_tau")
        (real, syn, null), space = _prediction_losses(data)
        require(bool(real) and len(real) == len(syn) == len(null), "invalid_paired_loss_evidence")
        gate = dict(data, real_losses=real, synthetic_losses=syn, null_losses=null)
        report = evaluate(gate)
        if report["status"] != "informative":
            return report
        harm = [query_harm(s, r, n, tau) for s, r, n in zip(syn, real, null)]
        mean_harm = mean(harm)
        return dict(report, tau=tau, loss_space=space, query_harm_margin=harm,
                    harm_margin=mean_harm, benefit_margin=-mean_harm)
    except MetricError as error:
        return failure(str(error))
