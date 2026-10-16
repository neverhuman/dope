"""Smallest tested passing artifact, with a cumulative budget frontier (T-033)."""
from .common import MetricError, count, failure, number, require, unit

STATES = {"complete", "unqualified", "incomplete", "failed"}


def evaluate(data):
    try:
        require(type(data) is dict)
        cells = data.get("cells")
        if not cells:
            return failure("missing_budget_cell", "incomplete")
        require(type(cells) is list, "invalid_cells")
        required = data.get("required_families")
        gates = data.get("required_gates")
        require(type(required) is list and bool(required) and all(type(f) is str and f for f in required), "invalid_families")
        require(len(required) == len(set(required)), "duplicate_family")
        require(type(gates) is list and bool(gates) and all(type(g) is str and g for g in gates), "invalid_gates")
        require(len(gates) == len(set(gates)), "duplicate_gate")
        tt, dt = unit(data.get("transfer_threshold", 0.90)), unit(data.get("driver_threshold", 0.80))
        budgets, admitted, states = set(), [], []
        for cell in cells:
            require(type(cell) is dict, "invalid_cells")
            size = count(cell.get("artifact_bytes"), 1)
            require(size not in budgets, "duplicate_budget")
            budgets.add(size)
            state = cell.get("status")
            require(type(state) is str and state in STATES, "invalid_cell_status")
            states.append(state)
            if state != "complete":
                continue
            observed, measured = cell.get("families"), cell.get("gates")
            if type(observed) is not list or not all(type(f) is str for f in observed):
                return failure("invalid_families")
            if not set(required) <= set(observed):
                states.append("incomplete")
                continue
            if type(measured) is not dict or any(measured.get(g) is None for g in gates):
                states.append("incomplete")
                continue
            require(all(type(measured[g]) is bool for g in gates), "invalid_gate")
            if cell.get("lcb_transfer") is None or cell.get("driver_fidelity") is None:
                states.append("incomplete")
                continue
            transfer, driver = number(cell["lcb_transfer"]), unit(cell["driver_fidelity"])
            admitted.append((size, transfer, driver, all(measured[g] for g in gates)))
        # Full tested-grid claims need every declared cell. No complete-case
        # fallback is permitted when a required family or gate is missing.
        for state in ("failed", "unqualified", "incomplete"):
            if state in states:
                return failure("budget_cell_" + state, state)
        rows, passing, best = [], [], None
        for size, transfer, driver, passed_gates in sorted(admitted):
            passed = passed_gates and transfer >= tt and driver >= dt
            if passed:
                passing.append(size)
            if passed_gates and driver >= dt:
                best = transfer if best is None else max(best, transfer)
            rows.append({"artifact_bytes": size, "lcb_transfer": transfer,
                         "driver_fidelity": driver, "tdl_pass": passed,
                         "frontier_transfer": best})
        return {"status": "qualified" if passing else "not_reached_in_tested_range",
                "reason": None, "tdl_bytes": min(passing) if passing else None,
                "largest_tested_bytes": max(budgets), "frontier": rows}
    except MetricError as error:
        return failure(str(error))
