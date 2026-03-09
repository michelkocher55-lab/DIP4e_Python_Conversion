from typing import Any
import numpy as np


def pWienerTF4e(H: Any, K: Any):
    """
    Parametric Wiener filter transfer function.

    W = pWienerTF4e(H, K)

    Implements Eq. (5-85) of DIP4E.
    W = (1./(H)).*(Habs2./(Habs2 + K))

    Note: matches MATLAB behavior exactly, including numerical instability
    when H is close to zero (Inverse Filter K=0).
    """

    # Compute square of absolute value
    Habs2 = np.abs(H) ** 2

    # Construct Wiener filter
    # Use errstate to allow division by zero (producing inf/nan) similar to MATLAB
    with np.errstate(divide="ignore", invalid="ignore"):
        # MATLAB: 1./H
        invH = 1.0 / H

        # MATLAB: Habs2 ./ (Habs2 + K)
        ratio = Habs2 / (Habs2 + K)

        # Result
        W = invH * ratio

    return W
