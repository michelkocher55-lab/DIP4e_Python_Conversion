from typing import Any
import numpy as np

# Ensure dftuv is importable (assuming it's in the same directory or lib path)
# In normal usage within libDIPUM, it should be importable directly or via relative import if in package.
# Here we assume it is in the path.

try:
    from dftuv import dftuv
except ImportError:
    # If strictly running as script and dftuv is in same dir but not in path
    try:
        from .dftuv import dftuv
    except ImportError:
        pass  # Expect dftuv to be available in path


def lpfilter(type_filter: Any, M: Any, N: Any, D0: Any, n: Any = 1):
    """
    Computes frequency domain lowpass filter transfer functions.

    Parameters:
    type_filter (str): 'ideal', 'butterworth', 'gaussian'.
    M, N (int): Size of the filter.
    D0 (float): Cutoff frequency.
    n (float, optional): Order for Butterworth filter. Default is 1.

    Returns:
    H (ndarray): Uncentered transfer function of size MxN.
    """

    # Use function dftuv to set up the meshgrid arrays.
    U, V = dftuv(M, N)

    # Compute the distances D(U,V).
    D = np.hypot(U, V)

    type_filter = type_filter.lower()

    if type_filter == "ideal":
        H = (D <= D0).astype(float)

    elif type_filter == "butterworth":
        # H = 1./(1 + (D./D0).^(2*n));
        # Handle division by zero if D0 is 0 (though D0 must be positive per doc)
        if D0 == 0:
            # If cutoff is 0, everything is blocked? Or just handle gracefully.
            # MatLAB code assumes D0 positive.
            H = np.zeros_like(D)
        else:
            H = 1.0 / (1.0 + (D / D0) ** (2 * n))

    elif type_filter == "gaussian":
        # H = exp(-(D.^2)./(2*(D0^2)));
        if D0 == 0:
            H = np.zeros_like(D)
        else:
            H = np.exp(-(D**2) / (2 * (D0**2)))

    else:
        raise ValueError(f"Unknown filter type: {type_filter}")

    return H
