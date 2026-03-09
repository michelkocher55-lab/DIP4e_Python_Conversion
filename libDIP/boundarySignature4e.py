from typing import Any
import numpy as np
from matplotlib.path import Path


def boundarySignature4e(b: Any, xc: Any = None, yc: Any = None):
    """
    Computes the signature of a boundary.

    A signature is defined as the distance from (xc, yc) to the boundary,
    as a function of angle.

    Parameters:
    -----------
    b : numpy.ndarray
        (K, 2) array of boundary coordinates (x, y). Or rows, cols.
        MATLAB: B is an np-by-2 array.
        In this suite, we assume 0-based coordinates.
        Note: The function treats inputs as general Cartesian coordinates for signature calc.
        If inputs are row/col from an image, ensure consistent usage.
        MATLAB variable naming: "b(:,1) = b(:,1) - xc".
        However, later it does: "xcart = b(:,2); ycart = -b(:,1);"
        This suggests b(:,1) is rows (y-like), b(:,2) is cols (x-like).
        xcart = cols = x. ycart = -rows = -y.
        This corresponds to typical image handling where y axis points down.

    xc, yc : float, optional
        Center of the signature. If None, centroid is used.

    Returns:
    --------
    dist : numpy.ndarray
        Distance values.
    angle : numpy.ndarray
        Angle values in degrees.
    xc, yc : float
        The centroid used.

    Raises:
    -------
    ValueError: If inputs are invalid or centroid is outside boundary.
    """

    b = np.array(b, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("Input b must be of size K-by-2")

    # Eliminate last point if duplicate
    if len(b) > 1 and np.array_equal(b[0], b[-1]):
        b = b[:-1]

    np_points = b.shape[0]
    if np_points < 3:
        # A boundary usually implies a shape, so at least 3 points?
        # But code doesn't explicitly fail.
        pass

    # Find centroid if not specified
    if xc is None or yc is None:
        xc = np.mean(b[:, 0])
        yc = np.mean(b[:, 1])

    # Check if (xc, yc) is inside boundary
    # Use matplotlib.path
    # Note: b coordinates are interpreted as vertices of polygon
    poly_path = Path(b)
    # contains_point returns bool
    is_inside = poly_path.contains_point((xc, yc))

    if not is_inside:
        # MATLAB error: 'Point (xc,yc) or default centroid not inside the boundary.'
        # However, for complex shapes (u-shapes), centroid might be outside.
        # The MATLAB code enforces it must be inside.
        raise ValueError("Point (xc,yc) or default centroid not inside the boundary.")

    # Shift origin
    b_shifted = b - np.array([xc, yc])

    # Convert to polar
    # MATLAB: xcart = b(:,2); ycart = -b(:,1);
    # b(:,1) is usually Row (y), b(:,2) is usually Col (x).
    # In standard Cartesian: x is col, y is row (usually inverted in image).
    # MATLAB's ycart = -b(:,1) accounts for image y-axis pointing down.

    xcart = b_shifted[:, 1]
    ycart = -b_shifted[:, 0]

    # [theta, rho] = cart2pol(xcart, ycart)
    # Python: rho = hypot(x, y), theta = atan2(y, x)
    rho = np.hypot(xcart, ycart)
    theta = np.arctan2(ycart, xcart)

    # Convert to degrees
    theta_deg = np.degrees(theta)

    # Helper to clean up angles like MATLAB:
    # "theta = theta.*(0.5*abs(1 + sign(theta))) - 0.5*(-1 + sign(theta)).*(360 + theta);"
    # If theta >= 0: theta. If theta < 0: 360 + theta.
    # Effectively theta_deg % 360, but ensuring positive in range [0, 360).
    theta_deg = np.where(theta_deg < 0, 360 + theta_deg, theta_deg)

    # Round to 1 degree increments
    theta_rounded = np.round(theta_deg)

    # Handle wrap around 360 -> 0 for sorting unique?
    # No, MATLAB code preserves 0, but unique handles them.
    # If rounded results in 360, it should be 0?
    # MATLAB doesn't explicitly mod 360 here immediately.
    # It does: "if tr(end,1) == tr(1) + 360, delete last"

    # Stack and sort
    # tr = [theta, rho]
    # unique sorts by first column

    tr = np.column_stack((theta_rounded, rho))

    # Unique angles
    # numpy.unique returns sorted uniques.
    # But we want to average distances for same angle?
    # MATLAB: [~, u] = unique(tr(:,1)); tr = tr(u, :);
    # unique(..., return_index=True) returns FIRST occurrence indices in `u`?
    # MATLAB `unique` returns "sort order". The legacy behavior or flag might differ.
    # MATLAB unique(A, 'rows') returns unique rows.
    # MATLAB unique(x) on vector returns sorted unique values.
    # [C, ia, ic] = unique(x). C = x(ia).
    # Default is 'sorted'. It typically picks the LAST occurrence if 'legacy' off?
    # Or first?
    # Actually for 1D array, if there represent multiple distances for same angle (e.g. non-convex shape),
    # a signature is only defined single-valued if star-shaped wrt centroid?
    # If multiple rhos for same theta exist, simply dropping duplicates blindly picks one.
    # MATLAB code: "u identifies the rows kept".
    # Relying on `unique` implies we just pick one.

    unique_angles, indices = np.unique(theta_rounded, return_index=True)

    # np.unique returns indices of the *first* occurrence?
    # Documentation says: "The indices of the first occurrences of the unique values in the original array."
    # So we keep the first point in the list that maps to that angle.

    tr_unique = tr[indices]

    # Sort by angle (np.unique indices might not preserve original order if we just used indices,
    # but unique_angles is sorted, so tr_unique should be sorted by angle if we use the sort order of angles?
    # No, `indices` points to where unique values are.
    # If we just take tr[indices], it picks those rows.
    # But are they sorted by angle?
    # np.unique returns unique_angles in sorted order.
    # But `indices` are not necessarily sorted.
    # We want the output to be sorted by angle.

    # Let's reconstruct properly.
    # We have unique_angles (sorted).
    # We need corresponding rho.
    # Since we want to emulate MATLAB `tr = tr(u, :)`, we need `u` to be indices that produce sorted result.
    # If `unique_angles` is sorted, we want the rho corresponding to those.

    # Actually, simpler:
    # MATLAB: [~, u] = unique(tr(:,1)) -> C = A(u).
    # C is sorted. So u orders them sorted.

    # So tr_unique is correct IF we trust that `indices` from np.unique corresponds to `unique_angles`.
    # Yes, unique_angles[i] == theta_rounded[indices[i]].

    final_angles = tr_unique[:, 0]
    final_dists = tr_unique[:, 1]

    # Check 360 wrap around
    # "If the last angle equals 360 degrees plus the first angle, delete the last angle."
    # Since we rounded angles, max should be 360?
    # If we have 0 and 360?
    if len(final_angles) > 1:
        if final_angles[-1] == final_angles[0] + 360:
            # This implies 0 and 360 both exist?
            # Or -180 and 180? (But we converted to 0..360).
            # So if we have 0 and 360.
            final_angles = final_angles[:-1]
            final_dists = final_dists[:-1]

    return final_dists, final_angles, xc, yc
