"""Offset-token keys used by the original Beauty SID namespace."""
import numpy as np

RADIX = 769
SID_SPACE = RADIX ** 3
PREFIX_SPACE = RADIX ** 2


def sid_keys(tokens):
    x = np.asarray(tokens, dtype=np.int64)
    return (x[..., 0] * RADIX + x[..., 1]) * RADIX + x[..., 2]


def prefix_keys(tokens):
    x = np.asarray(tokens, dtype=np.int64)
    return x[..., 0] * RADIX + x[..., 1]


def lookup(keys, values, query):
    positions = np.searchsorted(keys, query)
    safe = np.minimum(positions, len(keys) - 1)
    return np.where((positions < len(keys)) & (keys[safe] == query), values[safe], 0)
