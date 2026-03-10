from typing import Any
import numpy as np
from skimage.measure import regionprops, label
from skimage.segmentation import find_boundaries


def prune_pixel_list(r: Any, c: Any):
    """
    Removes pixels from vectors r and c that cannot be endpoints of the major axis.
    Based on geometrical constraints (Russ, Image Processing Handbook).
    """
    if len(r) == 0:
        return r, c

    top = np.min(r)
    bottom = np.max(r)
    left = np.min(c)
    right = np.max(c)

    # Upper circle
    x = (left + right) / 2
    y = top
    radius_sq = (bottom - top) ** 2
    inside_upper = ((c - x) ** 2 + (r - y) ** 2) < radius_sq

    # Lower circle
    y = bottom
    inside_lower = ((c - x) ** 2 + (r - y) ** 2) < radius_sq

    # Left circle
    x = left
    y = (top + bottom) / 2
    radius_sq = (right - left) ** 2
    inside_left = ((c - x) ** 2 + (r - y) ** 2) < radius_sq

    # Right circle
    x = right
    inside_right = ((c - x) ** 2 + (r - y) ** 2) < radius_sq

    # Eliminate points inside all 4 circles
    keep_mask = ~(inside_left & inside_right & inside_upper & inside_lower)

    return r[keep_mask], c[keep_mask]


def compute_diameter_region(props: Any, image_shape: Any):
    """
    Computes diameter and major axis for a single region.
    """
    # Boundary pixels
    # props.image is the bounding box image.
    # We need coordinates in the global image frame.

    # regionprops 'coords' gives all pixel coords. 'perimeter' gives length.
    # We want boundary pixel COORDINATES.
    # We can rely on find_boundaries on the mask.

    padded_mask = np.pad(props.image, 1, mode="constant", constant_values=0)
    boundary_mask = find_boundaries(padded_mask, mode="inner")[1:-1, 1:-1]

    r_local, c_local = np.where(boundary_mask)

    # Convert to global coordinates
    r = r_local + props.bbox[0]
    c = c_local + props.bbox[1]

    # Prune
    rp, cp = prune_pixel_list(r, c)

    num_pixels = len(rp)

    if num_pixels == 0:
        return -np.inf, np.ones((2, 2)), r, c
    elif num_pixels == 1:
        return 0.0, np.array([[rp[0], cp[0]], [rp[0], cp[0]]]), r, c
    elif num_pixels == 2:
        d = (rp[1] - rp[0]) ** 2 + (cp[1] - cp[0]) ** 2
        return d, np.column_stack((rp, cp)), r, c
    else:
        # Pairwise distances
        # Optimization: Use pdist if num_pixels is small enough, else...
        points = np.column_stack((rp, cp))

        # Brute force distance (Memory intensive if too many points? Pruning helps)
        # Expansion: (x-x')^2 = x^2 - 2xx' + x'^2

        # Let's verify size.
        if num_pixels > 2000:
            print(
                f"Warning: computing diameter with {num_pixels} points. Might be slow."
            )

        # We can use scipy.spatial.distance.pdist
        from scipy.spatial.distance import pdist, squareform

        # pdist returns condensed distance matrix
        dists = pdist(points, "sqeuclidean")
        max_dist_sq = np.max(dists)
        argmax = np.argmax(dists)

        # Convert 1D index to 2D indices (i, j)
        # argmax is index in condensed matrix.
        # It's better to use squareform if RAM allows, or convert index manually.
        # num_pixels ~ 1000 -> matrix 1M entry -> 8MB. Cheap.

        D_mat = squareform(dists)
        i, j = np.unravel_index(np.argmax(D_mat), D_mat.shape)

        d = np.sqrt(max_dist_sq)
        majoraxis = np.array([points[i], points[j]])  # [r1 c1; r2 c2]

        return d, majoraxis, r, c


def compute_basic_rectangle(props: Any, majoraxis: Any, perim_r: Any, perim_c: Any):
    """
    Computes Basic Rectangle and Minor Axis aligned with Major Axis.
    """
    # Major Axis: [[r1, c1], [r2, c2]]
    # Angle
    dy = majoraxis[1, 0] - majoraxis[0, 0]  # r2 - r1 (y is row)
    dx = majoraxis[1, 1] - majoraxis[0, 1]  # c2 - c1 (x is col)

    # theta = atan2(dy, dx) ?
    # MATLAB: atan2(MajorAxis(2,1) - MajorAxis(1,1), MajorAxis(2,2) - MajorAxis(1,2))
    # where col 1 is row(y), col 2 is col(x).
    # So atan2(delta_row, delta_col).
    # Note: image coords, Y is down (row), X is right (col).
    # Typically atan2(y, x).

    theta = np.arctan2(dy, dx)

    # Rotation Matrix
    # T = [cos sin; -sin cos]
    c_th = np.cos(theta)
    s_th = np.sin(theta)
    T = np.array([[c_th, s_th], [-s_th, c_th]])

    # Rotate perimeter pixels
    # Points P = [col row] ? MATLAB: p = [perim_c perim_r] -> x, y
    p = np.column_stack((perim_c, perim_r))
    p_rot = p @ T.T  # p * T'

    # Min/Max
    x_rot = p_rot[:, 0]
    y_rot = p_rot[:, 1]

    min_x = np.min(x_rot)
    max_x = np.max(x_rot)
    min_y = np.min(y_rot)
    max_y = np.max(y_rot)

    corners_x = np.array([min_x, max_x, max_x, min_x])
    corners_y = np.array([min_y, min_y, max_y, max_y])

    # Rotate corners back
    corners_rot = np.column_stack((corners_x, corners_y))
    corners = corners_rot @ T  # * T (inverse of T' is T for rotation matrix?)
    # T = [c s; -s c]. T' = [c -s; s c]. Inv(T') = T.
    # MATLAB: corners = [cx cy] * T;
    # Wait. MATLAB T = [cos sin; -sin cos].
    # p = p * T'.
    # corners_orig = corners_rot * inv(T') = corners_rot * T. Correct.

    # Basic Rectangle coords [row, col]
    # corners has [x, y] -> [col, row]
    basicrect = np.column_stack((corners[:, 1], corners[:, 0]))

    # Minor Axis endpoints (rotated)
    # x = (min_x + max_x) / 2
    # y1 = min_y, y2 = max_y
    x_mid = (min_x + max_x) / 2
    endpoints_rot = np.array([[x_mid, min_y], [x_mid, max_y]])

    # Rotate back
    endpoints = endpoints_rot @ T
    minoraxis = np.column_stack((endpoints[:, 1], endpoints[:, 0]))

    return basicrect, minoraxis


class DiameterResult:
    def __init__(self):
        """__init__."""
        self.Diameter = 0
        self.MajorAxis = None  # [[r1, c1], [r2, c2]]
        self.MinorAxis = None
        self.BasicRectangle = None


def diameter(label_image: Any):
    """
    Computes diameter and related properties for labeled regions.
    Returns a list of DiameterResult objects (or a single one if only 1 region?).
    MATLAB returns array of structs. We return list of objects.
    """
    if label_image.dtype == bool:
        label_image = label(label_image)

    props_list = regionprops(label_image)
    results = []

    for props in props_list:
        res = DiameterResult()
        d, maj_ax, pr, pc = compute_diameter_region(props, label_image.shape)
        res.Diameter = d
        res.MajorAxis = maj_ax

        res.BasicRectangle, res.MinorAxis = compute_basic_rectangle(
            props, maj_ax, pr, pc
        )
        results.append(res)

    return results
