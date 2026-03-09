from typing import Any
import numpy as np


def levelSetHeavimp4e(phi: Any, epsilon: Any = 1.0):
    """
    Computes Regularized Heaviside and Impulse (Dirac) functions.

    HS, IMP = levelsetHeavimp4e(phi, epsilon=1.0)

    Parameters
    ----------
    phi : numpy.ndarray
        Level set function.
    epsilon : float
        Regularization width parameter.

    Returns
    -------
    HS : numpy.ndarray
        Regularized Heaviside function.
    IMP : numpy.ndarray
        Regularized Dirac Delta function (Derivative of HS).
    """

    phi = np.asarray(phi, dtype=float)

    # Eq. (11-115)
    # HS = 0.5 * (1 + 2/pi * atan(phi/epsilon))
    HS = 0.5 * (1 + (2 / np.pi) * np.arctan(phi / epsilon))

    # Eq. (11-116)
    # IMP = 1/pi * epsilon / (epsilon^2 + phi^2)
    IMP = (1 / np.pi) * epsilon / (epsilon**2 + phi**2)

    return HS, IMP
