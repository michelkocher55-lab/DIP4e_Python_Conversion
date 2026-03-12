from typing import Any
import numpy as np


def polyangles(x: Any, y: Any):
    """
    Computes internal polygon angles.

    angles = polyangles(x, y) computes the interior angles (in degrees) of
    an arbitrary polygon whose vertices have x- and y-coordinates given
    in x and y. The vertices must be arranged in a clockwise manner.

    Parameters
    ----------
    x, y : array_like
        Coordinates of vertices.

    Returns
    -------
    angles : numpy.ndarray
        Interior angles in degrees.
    """

    # 1. Preliminaries
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    if len(x) == 0:
        return np.array([])

    xy = np.column_stack((x, y))

    # 2. Close the polygon if necessary
    # if (size(xy,1) == 1) || ~isequal(xy(1,:),xy(end,:))
    if len(xy) == 1 or not np.array_equal(xy[0], xy[-1]):
        xy = np.vstack((xy, xy[0]))

    # 3. Eliminate duplicate vertices
    # xy(all(diff(xy,1,1) == 0,2),:) = []
    # diff between rows. If delta is 0, remove.
    if len(xy) > 2:
        deltas = np.diff(xy, axis=0)
        # Check rows where both x and y diff are 0
        keep_mask = np.any(deltas != 0, axis=1)
        # Keep the FIRST row always (index 0 based diff corresponds to change from i to i+1)
        # Actually diff result length is N-1.
        # We want to keep row `i` if `row i != row i+1`?
        # MATLAB: xy(all(diff(xy)==0, 2), :) = [].
        # It removes the row `i` if `xy(i+1) == xy(i)`.
        # So we keep row `i` if it's different from next.
        # BUT we must keep the last row as it is the closing point?
        # Let's reconstruct.
        # xy_clean = xy[0]... then only add if different?

        # Vectorized approach matching MATLAB:
        # Indices to remove: where diff is all 0.
        to_remove = np.all(deltas == 0, axis=1)
        # Note: to_remove[i] True means xy[i+1] == xy[i].
        # MATLAB removes row `i`? Or `i+1`?
        # MATLAB: diff(xy) returns difference. Indicies match first element of pair?
        # Yes. If X = [A; A; B]. diff = [0; B-A].
        # Removing row 1 -> [A; B]. Correct.

        # So we keep rows where to_remove is False.
        # But wait, we have N rows, diff is N-1.
        # We filter first N-1 rows based on mask.
        # And always keep the last row?

        xy_trimmed = xy[:-1][~to_remove]
        xy = np.vstack((xy_trimmed, xy[-1]))  # Add last back

    # 4. Form vectors
    # v2 = diff(xy,1,1)
    v2 = np.diff(xy, axis=0)

    # 5. Shift vectors
    # v1 = circshift(v2, [1, 0])
    # Shifts down by 1. Last element moves to top.
    v1 = np.roll(v2, 1, axis=0)

    # 6. Components
    v1x = v1[:, 0]
    v1y = v1[:, 1]
    v2x = v2[:, 0]
    v2y = v2[:, 1]

    # 7. Angles
    # angles = (180/pi)*(atan2(v2y,v2x) - atan2(v1y,v1x))
    angles_rad = np.arctan2(v2y, v2x) - np.arctan2(v1y, v1x)
    angles_deg = np.degrees(angles_rad)

    # 8. Modulo
    # angles = mod(angles + 180, 360)
    angles = np.mod(angles_deg + 180, 360)

    return angles
