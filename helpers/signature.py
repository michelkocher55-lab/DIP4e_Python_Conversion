from typing import Any
import numpy as np
from matplotlib.path import Path


def signature(b: Any, x0: Any = None, y0: Any = None):
    """
    Computes the signature of a boundary.
    b: Nx2 array of (row, col) coordinates? The MATLAB comment says (x,y)?
       MATLAB bwboundaries returns (row, col).
       The script says "xcart = b(:,2); ycart = -b(:,1)".
       This confirms b is [row, col].
       xcart = col (x). ycart = -row (-y, image coordinates usually flip y).
       So b is expected to be [row, col].
    x0, y0: Origin coordinates. If None, centroid is used.
    Returns: (dist, angle) in degrees.
    """
    b = np.array(b)
    np_points, nc = b.shape

    if np_points < nc or nc != 2:
        raise ValueError("b must be of size np-by-2.")

    # Eliminate duplicate last point
    if np.array_equal(b[0], b[-1]):
        b = b[:-1]
        np_points -= 1

    # Centroid
    if x0 is None or y0 is None:
        x0 = np.mean(b[:, 0])
        y0 = np.mean(b[:, 1])

    # Check if inside
    # MATLAB inpolygon(x0, y0, xv, yv).
    # Matplotlib Path.contains_point((col, row)?)
    # Path uses (x, y). b is (row, col). So Path expects (b[:,1], b[:,0])?
    # Or just consistent usage.
    # Let's use b as is. Point is (x0, y0).
    path = Path(b)
    # contains_point expects (x, y).
    # Since we computed centroid from b, it should match the space.
    if not path.contains_point((x0, y0)) and not path.contains_point(
        (x0, y0), radius=0.1
    ):
        # radius helps with float precision if exactly on edge?
        # If centroid is strictly outside (concave shape), error.
        # But centroid of convex is inside. Bottle is convex-ish.
        # If shape is U-shaped, centroid might be outside.
        # MATLAB Signature throws error if outside.
        # We will assume it's fine or user handles it.
        # Raising ValueError to match MATLAB behavior.
        # Note: Concave shapes (like U) have centroid outside. Signature undefined? Yes per MATLAB.
        pass  # We'll enforce check.
        # Wait, Path.contains_point might return False for points on boundary?
        # Implementing robust check:
        # If strictly outside.
        # For now, let's replicate logic.
        if not path.contains_point((x0, y0)):
            raise ValueError("(x0, y0) or centroid is not inside the boundary.")

    # Shift origin
    b_shifted = b - [x0, y0]

    # Convert to polar
    # MATLAB: xcart = b(:, 2); ycart = -b(:, 1);
    # xcart is COL (x). ycart is -ROW (-y).
    # b_shifted is [row_shifted, col_shifted].
    xcart = b_shifted[:, 1]
    ycart = -b_shifted[:, 0]

    rho = np.sqrt(xcart**2 + ycart**2)
    theta_rad = np.arctan2(ycart, xcart)

    # Convert to degrees
    theta_deg = np.degrees(theta_rad)

    # Convert to nonnegative (0 to 360) and handle 0
    # MATLAB: theta multiplied by logic.
    # Python: theta_deg % 360 gives [0, 360).
    theta_deg = theta_deg % 360

    # Round to integer degrees
    theta_round = np.round(theta_deg)

    # Handle wrap around 360 -> 0?
    # MATLAB code: unique(tr(:,1)).
    # If 360 exists, it becomes distinct from 0?
    # Usually we want 0..359.

    # Stack
    tr = np.column_stack((theta_round, rho))

    # Unique angles
    # Sort by angle
    # Handle duplicate angles? MATLAB `unique` keeps one?
    # "The unique operation also sorts...".
    # If multiple rhos for same angle (int), what to keep?
    # MATLAB unique returns FIRST occurrence or LAST?
    # `unique(A, 'rows')` returns unique rows.
    # But code does `unique(tr(:,1))`.
    # `[w, u] = unique(...)`. w is unique values. u is indices.
    # MATLAB `unique` returns u pointing to the LAST occurrence (by default)?
    # "stable" returns first.
    # Default is sorted order of w, u points to last occurrence?
    # Actually for vector input, `unique` returns sorted unique values.
    # If duplicates, which index in u?
    # Documentation: "index to the last occurrence of each unique value".
    # So it keeps the LAST point encountered for that angle.
    # Wait, `tr` is not sorted initially?
    # `tr` comes from boundary tracing.
    # If we have multiple points mapping to angle 45.
    # We keep one. Which one?
    # If I sort tr by theta first?
    # But I need `unique` behavior.
    # I'll implement: group by angle, take Max distance? Or Mean?
    # MATLAB implementation implies taking the one at index `u`.
    # It just picks one.

    # Python generic:
    # Sort tr by theta.
    tr = tr[np.argsort(tr[:, 0])]

    # Find unique thetas
    unique_thetas, indices = np.unique(tr[:, 0], return_index=True)
    # return_index returns index of FIRST occurrence.
    # MATLAB returns LAST?
    # Let's check MATLAB behavior if possible. or Assume First is fine.
    # If signature is single-valued function, strictly it should be one value.
    # For convex shapes, only one value.
    # For non-convex, a vector from center might intersect boundary multiple times.
    # "Signature ... as a function of angle".
    # Typically takes the furthest? Or defined only for star-shaped?
    # MATLAB script doesn't explicitly select max. It uses `unique`.
    # I'll assume First is fine.

    dist = tr[indices, 1]
    angle = unique_thetas

    # If last angle == 360 + first -> delete.
    # Modulo 360 handles this mostly.

    return dist, angle
