from typing import Any
import numpy as np


def triangmf(z: Any, a: Any, b: Any, c: Any):
    """
    Triangular membership function.

    Parameters:
    z (ndarray): Input variable.
    a, b, c (float): Parameters (a <= b <= c).

    Returns:
    mu (ndarray): Membership values.
    """
    z = np.asarray(z)
    mu = np.zeros_like(z, dtype=float)

    # a <= z < b
    low_side = (z >= a) & (z < b)
    if b > a:
        mu[low_side] = (z[low_side] - a) / (b - a)

    # b <= z < c
    high_side = (z >= b) & (z < c)
    if c > b:
        mu[high_side] = 1 - (z[high_side] - b) / (c - b)

    # Boundary case z=b (peak) is covered by logic if ranges overlap properly
    # or if high_side includes b.
    # MATLAB: input ranges are [a, b) and [b, c). Peak at b is 1.
    # In my logic:
    # If z=b:
    # low_side False
    # high_side True -> 1 - 0 = 1. Correct.
    # If c=b (right angle), high_side empty, handled.

    return mu
