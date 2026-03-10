from typing import Any
import numpy as np


def trapezmf(z: Any, a: Any, b: Any, c: Any, d: Any):
    """
    Trapezoidal membership function.

    Parameters:
    z (ndarray): Input variable.
    a, b, c, d (float): Parameters (a <= b <= c <= d).

    Returns:
    mu (ndarray): Membership values.
    """
    z = np.asarray(z)
    mu = np.zeros_like(z, dtype=float)

    # a <= z < b: Rising edge
    if b > a:
        idx = (z >= a) & (z < b)
        mu[idx] = (z[idx] - a) / (b - a)
    elif b == a:
        # If b==a, abrupt rise? Or strictly < b condition fails.
        # Standard trapezmf logic:
        # 0 if z < a
        # (z-a)/(b-a) if a <= z < b
        # 1 if b <= z <= c
        pass

    # b <= z <= c: Plateau
    idx = (z >= b) & (z <= c)
    mu[idx] = 1.0

    # c < z <= d: Falling edge
    if d > c:
        idx = (z > c) & (z <= d)
        mu[idx] = 1 - (z[idx] - c) / (d - c)

    return mu
