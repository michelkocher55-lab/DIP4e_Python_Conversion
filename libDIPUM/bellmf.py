from typing import Any
import numpy as np
from .smf import smf


def bellmf(z: Any, a: Any, b: Any):
    """
    Bell-shaped membership function using smf.

    Parameters:
    z (ndarray or scalar): Input variable.
    a, b (float): Shape parameters (a <= b).

    Returns:
    mu (ndarray): Membership values.
    """
    z = np.asarray(z)
    mu = np.zeros_like(z, dtype=float)

    # Left side: z < b
    left_side = z < b
    if np.any(left_side):
        mu[left_side] = smf(z[left_side], a, b)

    # Right side: z >= b
    # mu = smf(2*b - z, a, b)
    right_side = z >= b
    if np.any(right_side):
        mu[right_side] = smf(2 * b - z[right_side], a, b)

    return mu
