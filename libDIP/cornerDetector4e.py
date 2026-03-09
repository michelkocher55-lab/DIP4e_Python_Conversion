from typing import Any
import numpy as np


def cornerDetector4e(f: Any, T: Any = 0):
    """
    Detects corners in an image.

    [g, EV1, EV2] = cornerDetector4e(f, T) computes corner values at
    each point in f, where T is a threshold in the range [0, 1].

    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    T : float, optional
        Threshold (default 0).

    Returns:
    --------
    g : numpy.ndarray
        Binary map of corners.
    EV1 : numpy.ndarray
        First eigenvalue map.
    EV2 : numpy.ndarray
        Second eigenvalue map.
    """

    # Scale to [0, 1]
    f = f.astype(float)
    f_min = np.min(f)
    f_max = np.max(f)
    if f_max - f_min > 0:
        f = (f - f_min) / (f_max - f_min)

    # Compute Gradients
    # MATLAB: [fy, fx] = gradient(f)
    # MATLAB gradient returns (dX, dY) i.e. (Cols, Rows).
    # So fy = dCols, fx = dRows.

    # numpy.gradient returns (dRows, dCols) by default for 2D.
    # So grads[0] is dRows, grads[1] is dCols.

    grads = np.gradient(f)
    fx = grads[0]  # dRows (Vertical) -> Matches MATLAB's 'fx'
    fy = grads[1]  # dCols (Horizontal) -> Matches MATLAB's 'fy'

    # Compute Cross Derivatives
    # MATLAB: [~, fxy] = gradient(fx)
    # Input is fx (dRows).
    # MATLAB gradient output 2 is dRows (Vertical).
    # So fxy = dRows(dRows(f)) = d^2 f / dRows^2.

    grads_of_fx = np.gradient(fx)
    fxy = grads_of_fx[0]  # dRows of fx -> Matches MATLAB's 'fxy'

    # Construct Matrix Terms
    # a = fx.^2
    # b = fxy.^2
    # d = fy.^2
    a = fx**2
    b = fxy**2
    d = fy**2

    # Eigenvalues of symmetric matrix A = [a b; b d]
    # Note: MATLAB code line 48: sqrt((a-d).^2 + 4*(b.^2)).
    # If matrix defines off-diagonal as 'b', then char eq is:
    # (a-lambda)(d-lambda) - b^2 = 0
    # lambda^2 - (a+d)lambda + (ad - b^2) = 0
    # Discriminant: (a+d)^2 - 4(ad - b^2) = a^2 + 2ad + d^2 - 4ad + 4b^2 = (a-d)^2 + 4b^2.
    # In MATLAB code 'b' variable holds fxy^2.
    # So the off-diagonal element of the matrix is technically fxy^2?
    # Or is 'b' in the code meant to represent the SQUARE?
    # Line 48: 4*(b.^2). This means 4 * (fxy^2)^2 = 4 * fxy^4 ??
    # OR does the code mean `4*b` if b was already squared?
    # Let's check MATLAB source carefully.
    # 41: a = fx.^2;
    # 42: b = fxy.^2;
    # 48: commonterm = sqrt((a-d).^2 + 4*(b.^2));
    # Use PARENTHESES carefully: 4 * (b squared).
    # So the discriminant term uses b^2. Since b = fxy^2, this term is (fxy^2)^2 = fxy^4.
    # This implies the matrix off-diagonal element is 'b' (fxy^2).
    # Yes, Discriminant is (Tr)^2 - 4*Det = (a-d)^2 + 4*off_diag^2.
    # Here term is (a-d)^2 + 4 * b^2.
    # So off-diag element is b = fxy^2.
    # The eigenvalues are for Matrix [fx^2, fxy^2; fxy^2, fy^2].

    commonterm = np.sqrt((a - d) ** 2 + 4 * (b**2))

    EV1 = ((a + d) + commonterm) / 2.0
    EV2 = ((a + d) - commonterm) / 2.0

    # Output g
    # 1 where min(EV1, EV2) > T
    # EV2 is always the smaller one (since commonterm >= 0).
    # So condition is EV2 > T.
    g = np.zeros_like(f)
    g[np.minimum(EV1, EV2) > T] = 1.0

    return g, EV1, EV2
