from typing import Any
import numpy as np


def imageScaling4e(f: Any, cx: Any, cy: Any):
    """
    Scales an image using nearest-neighbor interpolation.

    Parameters:
    -----------
    f : numpy.ndarray
        Input image (2D or 3D).
    cx : float
        Scaling factor along x (rows). Must be > 0.
    cy : float
        Scaling factor along y (columns). Must be > 0.

    Returns:
    --------
    g : numpy.ndarray
        Scaled image.
    """
    f = np.array(f)
    if cx <= 0 or cy <= 0:
        raise ValueError("cx and cy must be positive")

    M, N = f.shape[:2]

    # Determine size of output image
    Mout = int(np.round(cx * M))
    Nout = int(np.round(cy * N))

    # Inverse scaling transformation
    # A = [1/cx 0 0; 0 1/cy 0; 0 0 1]
    # We map indices.
    # MATLAB uses 1-based indexing in meshgrid logic but A treats them as "coordinates".
    # See imageRotate4e logic.
    # Since scaling is purely diagonal, coordinate choice (0-based vs 1-based)
    # matters if there's an offset. Here A matches (0,0) to (0,0).

    # Python 0-based indices for coordinates [0, ..., Mout-1]
    # x_out = index / cx.
    # If index=0, x_in=0.
    # If index=Mout-1 approx cx*M-1. x_in approx M.

    # Let's trust the logic: map coordinate p -> orig = round(A*p).
    # [u; v; 1] = A * [xp; yp; 1]
    # u = xp / cx
    # v = yp / cy

    rows = np.arange(Mout)
    cols = np.arange(Nout)

    xp_grid, yp_grid = np.meshgrid(rows, cols, indexing="ij")

    # Flatten
    primeCoords = np.stack(
        [xp_grid.ravel(), yp_grid.ravel(), np.ones_like(xp_grid.ravel())]
    )

    # Matrix A
    A = np.array([[1.0 / cx, 0, 0], [0, 1.0 / cy, 0], [0, 0, 1]])

    # Map
    origCoords = A @ primeCoords

    # Round nearest neighbor
    origCoords = np.round(origCoords).astype(int)

    # Check Valid
    # Indices must be in [0, M-1] and [0, N-1]
    r_coords = origCoords[0, :]
    c_coords = origCoords[1, :]

    valid_mask = (r_coords >= 0) & (r_coords < M) & (c_coords >= 0) & (c_coords < N)

    # Assignment
    if f.ndim == 3:
        n_chan = f.shape[2]
        g = np.zeros((Mout, Nout, n_chan), dtype=f.dtype)

        out_r = xp_grid.ravel()[valid_mask]
        out_c = yp_grid.ravel()[valid_mask]
        in_r = r_coords[valid_mask]
        in_c = c_coords[valid_mask]

        g[out_r, out_c, :] = f[in_r, in_c, :]
    else:
        g = np.zeros((Mout, Nout), dtype=f.dtype)

        out_r = xp_grid.ravel()[valid_mask]
        out_c = yp_grid.ravel()[valid_mask]
        in_r = r_coords[valid_mask]
        in_c = c_coords[valid_mask]

        g[out_r, out_c] = f[in_r, in_c]

    return g
