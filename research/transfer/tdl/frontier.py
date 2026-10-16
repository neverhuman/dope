"""Smallest tested passing artifact, with a cumulative budget frontier (T-033)."""
from .common import MetricError, count, failure, number, opt_default, require, unit

STATES = {"complete", "unqualified", "incomplete", "failed"}


def _parse_grid(raw):
    if raw is None:
        return None
    require(type(raw) is list and bool(raw), "invalid_byte_grid")
    grid = [count(b, 1) for b in raw]
    require(len(grid) == len(set(grid)), "duplicate_budget")
    return sorted(grid)


def evaluate(data):
    try:
        require(type(data) is dict)
        cells = data.get("cells")
        if cells is None:
            return failure("missing_budget_cell", "incomplete")
        require(type(cells) is list, "invalid_cells")
        if not cells:
            return failure("missing_budget_cell", "incomplete")
        required = data.get("required_families")
        gates = data.get("required_gates")
        require(type(required) is list and bool(required) and all(type(f) is str and f for f in required), "invalid_families")
        require(len(required) == len(set(required)), "duplicate_family")
        require(type(gates) is list and bool(gates) and all(type(g) is str and g for g in gates), "invalid_gates")
        require(len(gates) == len(set(gates)), "duplicate_gate")
        raw_grid = data.get("byte_grid") if data.get("byte_grid") is not None else data.get("declared_grid_bytes")
        declared_grid = _parse_grid(raw_grid)
        tt = unit(opt_default(data, "transfer_threshold", 0.90))
        dt = unit(opt_default(data, "driver_threshold", 0.80))
        has_grid = declared_grid is not None or any(type(c) is dict and c.get("budget_bytes") is not None for c in cells)
        budgets, admitted, states = set(), [], []
        for cell in cells:
            require(type(cell) is dict, "invalid_cells")
            if cell.get("budget_bytes") is not None:
                budget = count(cell["budget_bytes"], 1)
                if cell.get("artifact_bytes") is not None:
                    size = count(cell["artifact_bytes"], 1)
                    require(size <= budget, "artifact_exceeds_budget")
                else:
                    size = None
            else:
                size = count(cell.get("artifact_bytes"), 1)
                budget = size
            if declared_grid is not None:
                require(budget in declared_grid, "undeclared_budget")
            require(budget not in budgets, "duplicate_budget")
            budgets.add(budget)
            state = cell.get("status")
            require(type(state) is str and state in STATES, "invalid_cell_status")
            states.append(state)
            if state != "complete":
                continue
            if size is None:
                states.append("incomplete")
                continue
            observed, measured = cell.get("families"), cell.get("gates")
            if observed is None:
                states.append("incomplete")
                continue
            require(type(observed) is list and all(type(f) is str and bool(f) for f in observed), "invalid_families")
            if not set(required) <= set(observed):
                states.append("incomplete")
                continue
            if measured is None:
                states.append("incomplete")
                continue
            require(type(measured) is dict, "invalid_gate")
            if any(measured.get(g) is None for g in gates):
                states.append("incomplete")
                continue
            require(all(type(measured[g]) is bool for g in gates), "invalid_gate")
            if cell.get("lcb_transfer") is None or cell.get("driver_fidelity") is None:
                states.append("incomplete")
                continue
            transfer, driver = number(cell["lcb_transfer"]), unit(cell["driver_fidelity"])
            gate_map = {g: measured[g] for g in gates}
            admitted.append((budget, size, transfer, driver, all(gate_map.values()), gate_map))
        # Full tested-grid claims need every declared cell. No complete-case
        # fallback is permitted when a required family or gate is missing.
        for state in ("failed", "unqualified"):
            if state in states:
                return failure("budget_cell_" + state, state)
        if declared_grid is not None and budgets != set(declared_grid):
            return failure("missing_budget_cell", "incomplete")
        if "incomplete" in states:
            return failure("budget_cell_incomplete", "incomplete")
        admitted.sort(key=lambda item: item[0])
        if all(not item[4] for item in admitted):
            always_failing = [g for g in sorted(gates) if all(not item[5][g] for item in admitted)]
            failed_gate = always_failing[0] if always_failing else sorted(g for g in gates if any(not item[5][g] for item in admitted))[0]
            return failure("gate_failed_" + failed_gate, "unqualified")
        rows, qualifying_idx, best = [], None, None
        for idx, (budget, size, transfer, driver, passed_gates, _) in enumerate(admitted):
            passed = passed_gates and transfer >= tt and driver >= dt
            if passed and qualifying_idx is None:
                qualifying_idx = idx
            if passed_gates and driver >= dt:
                best = transfer if best is None else max(best, transfer)
            row = {"artifact_bytes": size, "lcb_transfer": transfer,
                   "driver_fidelity": driver, "tdl_pass": passed,
                   "frontier_transfer": best}
            if has_grid:
                row = {"budget_bytes": budget, **row}
            rows.append(row)
        tdl_bytes = admitted[qualifying_idx][1] if qualifying_idx is not None else None
        status = "qualified" if qualifying_idx is not None else "not_reached_in_tested_range"
        if has_grid:
            tdl_budget = admitted[qualifying_idx][0] if qualifying_idx is not None else None
            prev_budget = 0 if qualifying_idx == 0 else (admitted[qualifying_idx - 1][0] if qualifying_idx is not None else None)
            interval = [prev_budget, tdl_budget] if qualifying_idx is not None else None
            return {"status": status, "reason": None, "tdl_bytes": tdl_bytes,
                    "tdl_budget_bytes": tdl_budget, "tdl_interval_bytes": interval,
                    "largest_tested_bytes": max(budgets), "frontier": rows}
        return {"status": status, "reason": None, "tdl_bytes": tdl_bytes,
                "largest_tested_bytes": max(budgets), "frontier": rows}
    except MetricError as error:
        return failure(str(error))
