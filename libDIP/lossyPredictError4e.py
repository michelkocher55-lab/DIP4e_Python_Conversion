from typing import Any
import numpy as np


def lossyPredictError4e(f: Any, predictor: Any):
    """
    Computes the prediction error associated with four different linear predictors.

    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    predictor : int
        Predictor type (1, 2, 3, or 4).

    Returns:
    --------
    m : float
        Mean of the prediction error.
    s : float
        Standard deviation of the prediction error.
    e_img : numpy.ndarray (uint8)
        Scaled prediction error image with 0 error at gray level 128.
    """
    f = f.astype(float)
    rows, cols = f.shape
    e = f.copy()

    # We apply prediction from (1,1) to (rows-1, cols-1) (0-indexed)
    # i.e., indices [1:, 1:]
    # Neighbors:
    # Left: f[1:, 0:-1]
    # Up:   f[0:-1, 1:]
    # Diag: f[0:-1, 0:-1]

    # Define vector slices
    # Center pixels (Target)
    target = f[1:, 1:]

    # Neighbors
    left = f[1:, :-1]
    up = f[:-1, 1:]
    diag = f[:-1, :-1]

    pred = np.zeros_like(target)

    if predictor == 1:
        # 0.97 * f(x, y-1) -> Left
        # MATLAB: works on columns j=2:n. e(:,j) = f(:,j) - 0.97*f(:,j-1).
        # Wait, MATLAB Case 1 loop is `for j=2:n`. It processes ALL rows for j>=2.
        # So it includes Row 0?
        # MATLAB: for j=2:n, e(:, j) = ...
        # Yes, Case 1 allows prediction for Row 0 (using Left neighbor).
        # Re-evaluating slice for Case 1.

        # Case 1 Slice: Columns 1 to end (all rows).
        ftarget = f[:, 1:]
        fleft = f[:, :-1]

        prediction = 0.97 * fleft
        e[:, 1:] = ftarget - prediction

    elif predictor == 2:
        # 0.5*f(x, y-1) + 0.5*f(x-1, y) -> 0.5*(Left + Up)
        # Loop j=2:n, k=2:m. So Row 0 and Col 0 are skipped.

        prediction = 0.5 * (left + up)
        e[1:, 1:] = target - prediction

    elif predictor == 3:
        # 0.75*f(x,y-1) + 0.75*f(x-1,y) - 0.5*f(x-1,y-1) -> 0.75*(Left + Up) - 0.5*Diag

        prediction = 0.75 * (left + up) - 0.5 * diag
        e[1:, 1:] = target - prediction

    elif predictor == 4:
        # Adaptive:
        # dh = abs(Up - Diag) = abs(f(x-1,y) - f(x-1,y-1))
        # dv = abs(Left - Diag) = abs(f(x,y-1) - f(x-1,y-1))
        # if dh < dv: predict 0.97*Left
        # else:       predict 0.97*Up

        dh = np.abs(up - diag)
        dv = np.abs(left - diag)

        # Mask where dh < dv
        mask = dh < dv

        # Prediction
        prediction = np.zeros_like(target)
        prediction[mask] = 0.97 * left[mask]
        prediction[~mask] = 0.97 * up[~mask]

        e[1:, 1:] = target - prediction

    else:
        raise ValueError(f"Unknown predictor: {predictor}")

    # Stats
    m = np.mean(e)
    s = np.std(e)

    # Scale Image
    e_img = (e / 2) + 128
    e_img = np.clip(e_img, 0, 255).astype(np.uint8)

    return m, s, e_img
