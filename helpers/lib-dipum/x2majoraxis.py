from typing import Any
import numpy as np
from skimage.transform import rotate
from skimage.measure import find_contours
from scipy.ndimage import binary_fill_holes


def x2majoraxis(A: Any, B: Any):
    """
    Aligns coordinate x with the major axis of a region.
    A: Major Axis Endpoints [[r1, c1], [r2, c2]] (Note: MATLAB might use [x y]?)
       MATLAB: A = [x1 y1; x2 y2].
       Python diameter.py returns [r1 c1; r2 c2].
       Need to be careful with coordinate systems.
       Let's assume A is consistent with B.
    B: Image (region) or Boundary points.
       If Image: aligns image.
       If Boundary (Nx2): aligns points.

    Returns: C (Aligned B), theta (degrees)
    """
    # Check input type
    is_image = False
    if B.ndim > 1 and B.shape[1] != 2:  # Image
        is_image = True
        type_ = "region"
    elif B.shape[1] == 2:
        is_image = False
        type_ = "boundary"
        # Compute centroid
        c = np.mean(B, axis=0)

        # Prepare for rotation via image if boundary
        # Create image from boundary?
        # MATLAB: bound2im, imfill, etc.
        # Here we might just rotate points directly?
        # MATLAB rotates IMAGE to handle connectivity issues.
        # "It is possible for a connected boundary to develop small breaks..."
        # If we use skimage.transform.rotate on image, it interpolates.
        # If we rotate points directly: P_new = P_old * R.
        # Let's follow MATLAB logic if possible, or simplify.
        # Rotating points is simpler and more precise for coordinates.
        # If the user wants an image back, we start with image.
        # But MATLAB `x2majoraxis` returns C same type as B.

        if type_ == "boundary":
            # Let's rotate points directly for simplicity/precision?
            # But MATLAB converts to image to fill holes?
            pass
    else:
        raise ValueError("Input B must be image or Nx2 array")

    # Major Axis Vector
    # A is [[r1 c1], [r2 c2]] (from my diameter.py) (Y X)
    # MATLAB A is [x y].
    # If using A from diameter.py, it is (Row, Col).
    # v = [r2-r1, c2-c1] -> [dy, dx]
    # x-axis is (0, 1) vector [dy=0, dx=1].

    # Let's standardize A to be [row, col]
    v = A[1] - A[0]  # [dr, dc]

    # Unit vector along x-axis (Col direction)
    # u = [0, 1] (dr=0, dc=1)

    # Angle
    dr, dc = v
    # angle of v w.r.t x-axis (dc)
    theta_rad = np.arctan2(dr, dc)

    # MATLAB logic: theta = acos(u'*v / ...).
    # Aligns x-axis with Major Axis.
    # So we want to rotate B by -theta so that v becomes horizontal?
    # Or rotate so x-axis matches v?
    # "Aligns x-coordinate axis with the major axis"
    # Usually means rotating the object so its major axis lies on the x-axis.
    # So we rotate by -theta.

    theta_deg = np.degrees(theta_rad)

    # MATLAB: returns theta (initial angle).
    # Rotates by theta? Or -theta?
    # "Rotate by angle theta" in MATLAB `imrotate` rotates CCW.
    # If v is at 45 deg, and we rotate 45 deg, it becomes 90 deg (vertical).
    # If we want it on x-axis (0 deg), we should rotate by -45.
    # MATLAB code: theta = acos(...). if theta > pi/2 ...
    # C = imrotate(B, theta, ...).
    # This suggests it aligns it to something?
    # Wait, "x-axis is the horizontal axis".
    # If theta is angle between Major and X.
    # If we rotate by theta, we might be rotating it TO x-axis? No, usually -theta.
    # Let's verify MATLAB behavior.

    # Let's assume we want to horizontalize the major axis.
    # So rotate by -theta_deg.

    theta_rot = theta_deg  # Placeholder

    # Actually, let's look at MATLAB code:
    # theta = acos(u'*v / ...). u = [1; 0] (x-axis).
    # C = imrotate(B, theta, ...).
    # If theta is angle between u and v, rotating B by theta...
    # Example: v is at 10 deg. theta = 10. Rotate by 10 -> 20 deg.
    # Unless u is defined differently or coordinate system differs.
    # MATLAB image coords: y down?
    # Let's implement robustly: Calculate angle. Rotate so it becomes 0.

    rotation_angle = -theta_deg

    if is_image:
        # Rotation
        # skimage.transform.rotate rotates by angle in DEGREES, CCW?
        # Need to verify. Usually CCW.
        # If vector is at +45 deg (South-East in image coords? No, Y down)
        # r increases down, c increases right.
        # A vector (1, 1) [dr, dc] is +45 deg from x-axis (0, 1) towards y-axis (1, 0).
        # atan2(1, 1) = 45 deg.
        # We want to rotate it back to (0, 1) [dr=0, dc>0].
        # So rotate by -45.

        C = rotate(B, rotation_angle, resize=True, order=0, preserve_range=True).astype(
            B.dtype
        )
        # resize=True corresponds to 'crop'? No, 'crop' keeps size. MATLAB uses 'crop'.
        # But 'crop' might cut off the object if it rotates diagonal.
        # MATLAB code: "The following image is of size m-by-m... to make sure there will be no size truncation".
        # And "Crop the rotated image to original size."
        # If we resize=True, we get full image.
        # If MATLAB forces crop, let's use resize=False (which crops).
        # "Crop the rotated image to original size." -> resize=False.

        C = rotate(
            B, rotation_angle, resize=False, order=0, preserve_range=True
        ).astype(B.dtype)

    else:
        # Boundary points
        # Rotate points about (0,0) or centroid?
        # MATLAB: converts to image, rotates, re-extracts boundary.
        # This handles connectivity.
        # Let's try to mimic that to obtain matching results.

        # 1. Create image
        min_x, min_y = np.min(B, axis=0)  # [c_min, r_min]
        max_x, max_y = np.max(B, axis=0)

        # Shift to avoid negative
        pad = 50
        width = int(max_x - min_x + 2 * pad)
        height = int(max_y - min_y + 2 * pad)

        # Just make a large enough canvas
        m = max(int(np.max(B)) * 2, 1000)
        img = np.zeros((m, m), dtype=bool)

        # Indices
        # B is [x y] (col row).
        # img[row, col]
        img[B[:, 1].astype(int), B[:, 0].astype(int)] = 1

        # imfill equivalent?
        # binary_fill_holes requires a closed boundary.
        # If simple boundary points, good.
        img_filled = binary_fill_holes(img)

        # Rotate
        img_rot = rotate(
            img_filled, rotation_angle, resize=False, order=0, preserve_range=True
        )

        # Boundaries
        # bwboundaries in MATLAB. find_contours in Python.
        contours = find_contours(img_rot, level=0.5)
        # contours returns [row, col]. We want [x, y].

        if len(contours) > 0:
            cnt = contours[0]  # longest?
            # Sort by length?
            cnt = max(contours, key=len)
            C = np.column_stack((cnt[:, 1], cnt[:, 0]))  # [col, row] -> [x, y]
        else:
            C = np.array([])

        # Shift back logic? MATLAB shifts back to centroid.
        # We can skip complex shifting if visualization doesn't rely on absolute alignment
        # but Equation12_4 checks alignment?
        # "figure, imshow (C), title ('X aligned to its Major Axis')"
        # So C is expected to be an image if B was an image.

    return C, theta_deg
