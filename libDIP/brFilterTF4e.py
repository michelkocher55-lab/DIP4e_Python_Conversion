from typing import Any
import numpy as np


def brFilterTF4e(filter_type: Any, M: Any, N: Any, C0: Any, W: Any, n: Any = 1):
    """
    Generates a bandreject filter transfer function.

    Parameters:
    -----------
    filter_type : str
        'ideal', 'gaussian', or 'butterworth'.
    M : int
        Number of rows.
    N : int
        Number of columns.
    C0 : float
        Cutoff frequency or center frequency of the band.
    W : float
        Width of the band.
    n : float, optional
        Order of the Butterworth filter. Default is 1.

    Returns:
    --------
    H : numpy.ndarray
        M x N transfer function.
    """

    # Coordinate grid
    # MATLAB: X = 0:M-1; Y = 0:N-1; [y, x] = meshgrid(Y, X);
    # y (cols) varies with Y (0..N-1). x (rows) varies with X (0..M-1).
    # Python meshgrid(..., indexing='xy') is default.
    # xv, yv = meshgrid(x_array, y_array). xv (rows) varies with x_array.
    # Wait.
    # np.meshgrid(x, y) returns
    # X (rows=y_len, cols=x_len) if indexing='xy'? No.
    # default 'xy': X has shape (len(y), len(x)). X[i,j] depends on x[j].
    # So X varies along columns. Y varies along rows.
    # MATLAB [y, x] = meshgrid(Y, X).
    # y = Y repeated in rows. y depends on column index.
    # x = X repeated in cols. x depends on row index.
    # Python convention: rows=M, cols=N.
    # We want grid where 'cols' goes 0..N-1, 'rows' goes 0..M-1.

    rows = np.arange(M)
    cols = np.arange(N)

    # We want 'y' variable to represent column index (0..N-1)
    # We want 'x' variable to represent row index (0..M-1)
    # using np.meshgrid(cols, rows) -> Y_grid (M, N) depends on cols?
    # X, Y = np.meshgrid(x, y). X is (len(y), len(x)).
    # Let's verify defaults.

    # Safer: indexing='ij'.
    # X, Y = np.meshgrid(rows, cols, indexing='ij') -> (M, N).
    # X[i,j] = rows[i]. Y[i,j] = cols[j].
    # This maps 'x' (rows) to rows, 'y' (cols) to cols.
    # Matches MATLAB 'x' (rows), 'y' (cols).

    x, y = np.meshgrid(rows, cols, indexing="ij")

    # Distance grid (Eq. 4-112)
    # Using true center (M/2, N/2)
    D = np.sqrt((x - M / 2) ** 2 + (y - N / 2) ** 2)

    filter_type = filter_type.lower()

    if filter_type == "ideal":
        # H = false(M,N) -> 0
        H = np.zeros((M, N), dtype=float)
        # R1 = D < C0 - W/2
        R1 = D < (C0 - W / 2)
        # R2 = D > C0 + W/2
        R2 = D > (C0 + W / 2)
        # H = R1 | R2
        H = (R1 | R2).astype(float)

    elif filter_type == "gaussian":
        # H = 1 - exp(-((D^2 - C0^2)/(D*W + eps))^2)
        # eps in MATLAB is approx 2.22e-16. Python np.finfo(float).eps.
        eps = np.finfo(float).eps

        term = (D**2 - C0**2) / (D * W + eps)
        H = 1 - np.exp(-(term**2))

    elif filter_type == "butterworth":
        # H = 1 / (1 + ((D*W) / (D^2 - C0^2 + eps))^2*n)
        # Using literal translation: (term)^2 * n
        eps = np.finfo(float).eps

        term = (D * W) / (D**2 - C0**2 + eps)
        H = 1 / (1 + (term**2) * n)

    else:
        raise ValueError(
            "Unknown filter type. Choose 'ideal', 'gaussian', or 'butterworth'."
        )

    return H
