from typing import Any
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes
from skimage.measure import find_contours
from helpers.bound2im import bound2im


def bsubsamp(b: Any, scale: Any, M: Any, N: Any):
    """
    Subsamples a boundary.
    b: Nx2 boundary coordinates.
    scale: resizing factor.
    M, N: Original image dimensions.
    Returns: B (subsampled boundary).
    """
    # 1. Create image
    I = bound2im(b, M, N)

    # 2. Fill holes
    I_filled = binary_fill_holes(I)

    # 3. Scale
    # Resize expects float or bool.
    # order=0 (Nearest neighbor)
    # output shape? implicit from scale.
    I_resized = resize(
        I_filled,
        (int(M * scale), int(N * scale)),
        order=0,
        mode="constant",
        anti_aliasing=False,
    )

    # 4. Get boundary
    # skimage find_contours returns float coordinates standard (0.5 shifted?).
    # level=0.5
    # For binary image, level 0.5 works.
    contours = find_contours(I_resized, 0.5)

    if len(contours) == 0:
        return np.empty((0, 2))

    # Pick longest? MATLAB bsubsamp returns B{1}.
    # Usually the external boundary is first or longest.
    # find_contours doesn't guarantee order.
    # Sort by length
    contours.sort(key=lambda x: len(x), reverse=True)
    B = contours[0]

    # Convert to integer pixel coordinates?
    # MATLAB bwboundaries returns pixel indices.
    # find_contours returns subpixel. e.g. 0.5, 0.5 for center of pixel (0,0).
    # To get pixel coords, roughly round?
    # Or just return as is?
    # Figure1205 says `B1 = B1 * r`. If B1 is pixels, B1*r scales back to original logic.
    # If B1 is float, it's fine.

    return B
