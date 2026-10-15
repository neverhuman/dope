"""Stress fixtures for interaction support.

Mechanisms are functions of column values. A selector must recover the
columns that generate the target. Nothing here names a fixture for promotion.

`dope-interaction-stress-development` version 1 builds bounded synthetic
columns for mechanism checks. It is not the historical helpers and it does
not feed selection, probes, or learner promotion.
"""


def _fail(code):
    raise ValueError(code)


def signed_bits(rows, salt):
    if type(rows) is not int or type(salt) is not int or rows < 4:
        _fail("invalid_shape")
    values = []
    for row in range(rows):
        mixed = (row * 0x9E3779B1 + salt * 0x85EBCA6B) & 0xFFFFFFFF
        rotated = ((mixed << 13) | (mixed >> 19)) & 0xFFFFFFFF
        values.append(-1.0 if (rotated ^ mixed) & 1 == 0 else 1.0)
    return values


def product_target(left, right):
    if len(left) != len(right) or len(left) < 4:
        _fail("invalid_shape")
    return [left[row] * right[row] for row in range(len(left))]


def sign_gate_target(left, right):
    if len(left) != len(right) or len(left) < 4:
        _fail("invalid_shape")
    return [1.0 if left[row] > 0.0 and right[row] > 0.0 else 0.0 for row in range(len(left))]


def weak_sum(columns, count):
    if not columns or count < 1 or count > len(columns):
        _fail("invalid_shape")
    rows = len(columns[0])
    if any(len(column) != rows for column in columns):
        _fail("invalid_shape")
    return [sum(column[row] for column in columns[:count]) for row in range(rows)]


def permute_columns(columns, order):
    if sorted(order) != list(range(len(columns))):
        _fail("invalid_shape")
    return [columns[index] for index in order]


DEVELOPMENT_STRESS = {
    "format": "dope-interaction-stress-development",
    "version": 1,
    "cell_budget": 4096,
    "qualifies_transfer": False,
    "feeds_selection": False,
    "formal_dp": False,
    "hipaa": False,
}


def _finite_number(value):
    if type(value) is not int and type(value) is not float:
        return False
    return value == value and value != float("inf") and value != float("-inf")


def _development_bounds(rows, features, seed, signal, noise, scale, offset):
    if (
        type(rows) is not int or type(features) is not int or type(seed) is not int
        or rows < 4 or features < 2 or seed < 0
        or rows * features > DEVELOPMENT_STRESS["cell_budget"]
    ):
        _fail("invalid_shape")
    if any(not _finite_number(value) for value in (signal, noise, scale, offset)) or noise < 0:
        _fail("invalid_shape")


def _active_indexes(active, features, accepted):
    if type(active) is not tuple or len(active) not in accepted:
        _fail("invalid_shape")
    if any(type(index) is not int or index < 0 or index >= features for index in active):
        _fail("invalid_shape")
    if len(set(active)) != len(active):
        _fail("invalid_shape")


def _column_order(order, features):
    if order is None:
        return list(range(features))
    if type(order) is not tuple or any(type(index) is not int for index in order):
        _fail("invalid_shape")
    if sorted(order) != list(range(features)):
        _fail("invalid_shape")
    return list(order)


def _walsh(rows, mask):
    column = []
    for row in range(rows):
        parity = (row & mask).bit_count() & 1
        column.append(-1.0 if parity else 1.0)
    return column


def _unit_noise(row, seed):
    mixed = (row * 0x9E3779B1 + seed * 0x85EBCA6B) & 0xFFFFFFFF
    return (mixed % 10001) / 10000.0 * 2.0 - 1.0


def _walsh_columns(rows, features, active, seed):
    bits = len(active)
    period = None
    while bits <= 8:
        candidate = 1 << bits
        if rows % candidate == 0 and candidate - 2 >= features:
            period = candidate
            break
        bits += 1
    if period is None:
        _fail("invalid_shape")
    active_masks = [1 << offset for offset in range(len(active))]
    reserved = 0
    for mask in active_masks:
        reserved ^= mask
    nuisance = [mask for mask in range(1, period) if mask != reserved and mask not in active_masks]
    if features - len(active) > len(nuisance):
        _fail("invalid_shape")
    if nuisance:
        shift = seed % len(nuisance)
        nuisance = nuisance[shift:] + nuisance[:shift]
    masks = [None] * features
    for index, mask in zip(active, active_masks):
        masks[index] = mask
    slots = [index for index, mask in enumerate(masks) if mask is None]
    for slot, mask in zip(slots, nuisance):
        masks[slot] = mask
    return [_walsh(rows, mask) for mask in masks]


def _continuous_columns(rows, features, seed):
    columns = []
    for column in range(features):
        values = []
        for row in range(rows):
            mixed = (
                row * 0x9E3779B1 + (column + 1) * 0x85EBCA6B + seed * 0xC2B2AE35
            ) & 0xFFFFFFFF
            values.append(((mixed % 10007) + 1) / 10008.0 * 2.0 - 1.0)
        columns.append(values)
    return columns


def _development_result(columns, raw, seed, signal, noise, scale, offset, order):
    ordered = permute_columns(columns, _column_order(order, len(columns)))
    target = []
    for row, value in enumerate(raw):
        noise_term = 0.0 if noise == 0 else _unit_noise(row, seed) * float(noise)
        target.append(float(scale) * (float(signal) * value + noise_term) + float(offset))
    return {
        "format": DEVELOPMENT_STRESS["format"],
        "version": DEVELOPMENT_STRESS["version"],
        "columns": tuple(tuple(column) for column in ordered),
        "target": tuple(target),
        "qualifies_transfer": False,
        "feeds_selection": False,
        "formal_dp": False,
        "hipaa": False,
    }


def development_xor_parity(
    rows, features, active, *, seed, signal=1.0, noise=0.0, scale=1.0, offset=0.0, order=None,
):
    """Balanced parity product of the placed columns.

    Two active columns are the XOR case and three are the parity case. The
    columns are Walsh characters on a complete period, so a single column has
    no correlation with the product. The product column itself is not stored.
    """
    _development_bounds(rows, features, seed, signal, noise, scale, offset)
    _active_indexes(active, features, (2, 3))
    columns = _walsh_columns(rows, features, active, seed)
    raw = [1.0] * rows
    for index in active:
        raw = [raw[row] * columns[index][row] for row in range(rows)]
    return _development_result(columns, raw, seed, signal, noise, scale, offset, order)


def development_continuous_product(
    rows, features, active, *, seed, signal=1.0, noise=0.0, scale=1.0, offset=0.0, order=None,
):
    """Product of two continuous columns. The other columns are distractors."""
    _development_bounds(rows, features, seed, signal, noise, scale, offset)
    _active_indexes(active, features, (2,))
    columns = _continuous_columns(rows, features, seed)
    left, right = active
    raw = [columns[left][row] * columns[right][row] for row in range(rows)]
    return _development_result(columns, raw, seed, signal, noise, scale, offset, order)


def development_rare_gate(
    rows, features, active, *, seed, signal=1.0, noise=0.0, scale=1.0, offset=0.0, order=None,
):
    """Indicator that four balanced columns are positive together.

    On a complete period the event count is rows / 16.
    """
    _development_bounds(rows, features, seed, signal, noise, scale, offset)
    _active_indexes(active, features, (4,))
    columns = _walsh_columns(rows, features, active, seed)
    raw = []
    for row in range(rows):
        raw.append(1.0 if all(columns[index][row] > 0.0 for index in active) else 0.0)
    return _development_result(columns, raw, seed, signal, noise, scale, offset, order)


def development_diffuse_additive(
    rows, features, active, *, seed, signal=1.0, noise=0.0, scale=1.0, offset=0.0, order=None,
):
    """Equal sum of two, three, or four orthogonal columns."""
    _development_bounds(rows, features, seed, signal, noise, scale, offset)
    _active_indexes(active, features, (2, 3, 4))
    columns = _walsh_columns(rows, features, active, seed)
    raw = [sum(columns[index][row] for index in active) for row in range(rows)]
    return _development_result(columns, raw, seed, signal, noise, scale, offset, order)


def development_conditional_interaction(
    rows, features, active, *, seed, signal=1.0, noise=0.0, scale=1.0, offset=0.0, order=None,
):
    """Product of the first two placed columns when the third is positive."""
    _development_bounds(rows, features, seed, signal, noise, scale, offset)
    _active_indexes(active, features, (3,))
    columns = _walsh_columns(rows, features, active, seed)
    left, right, regime = active
    raw = []
    for row in range(rows):
        if columns[regime][row] > 0.0:
            raw.append(columns[left][row] * columns[right][row])
        else:
            raw.append(0.0)
    return _development_result(columns, raw, seed, signal, noise, scale, offset, order)
