"""Stress fixtures for interaction support.

Mechanisms are functions of column values. A selector must recover the
columns that generate the target. Nothing here names a fixture for promotion.
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
