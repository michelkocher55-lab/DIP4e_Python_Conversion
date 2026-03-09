from typing import Any
import numpy as np


def lpFilterTF4e(filter_type: Any, P: Any, Q: Any, D0: Any, n: Any = 1):
    """
    Circularly-symmetric lowpass filter transfer function.

    Parameters:
    -----------
    filter_type : str
        'ideal', 'gaussian', or 'butterworth'.
    P : int
        Number of rows.
    Q : int
        Number of columns.
    D0 : float
        Cutoff frequency.
    n : float, optional
        Order of the Butterworth filter. Default is 1.

    Returns:
    --------
    H : numpy.ndarray
        P x Q transfer function.
    """

    # Coordinate grid
    # Matches MATLAB [y, x] = meshgrid(Y, X) convention where
    # x corresponds to rows (0..P-1) and y to cols (0..Q-1).

    rows = np.arange(P)
    cols = np.arange(Q)

    # meshgrid(rows, cols, indexing='ij') -> (P, Q)
    # x (rows) varies with axis 0
    # y (cols) varies with axis 1
    x, y = np.meshgrid(rows, cols, indexing="ij")

    # Distance grid centered at (P/2, Q/2)
    D = np.sqrt((x - P / 2) ** 2 + (y - Q / 2) ** 2)

    filter_type = filter_type.lower()

    if filter_type == "ideal":
        H = np.zeros((P, Q), dtype=float)
        H[D <= D0] = 1.0

    elif filter_type == "gaussian":
        # H = exp(-D^2/(2*(D0^2)))
        H = np.exp(-(D**2) / (2 * (D0**2)))

    elif filter_type == "butterworth":
        # H = 1./(1 + (D/D0).^(2*n))
        # Handle division by zero if D0 is 0? usually D0 > 0.
        # Add eps if check required? MATLAB logic doesn't explicitly add eps here like brFilter?
        # MATLAB: H = 1./(1 + (D/D0).^(2*n));
        # If D0 is very small, this blows up. Assume D0 > 0.

        eps = np.finfo(float).eps
        H = 1.0 / (1.0 + (D / (D0 + eps)) ** (2 * n))

    else:
        raise ValueError(
            "Unknown filter type. Choose 'ideal', 'gaussian', or 'butterworth'."
        )

    return H
