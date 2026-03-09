from typing import Any


def pixVal4e(f: Any, r: Any, c: Any):
    """
    Gets pixel value at specified coordinates in an image.

    v = pixVal4e(f, r, c)

    Parameters:
    -----------
    f : numpy.ndarray
        Input image.
    r : int
        Row index (0-based).
    c : int
        Column index (0-based).

    Returns:
    --------
    v : scalar
        Value of image f at coordinates (r, c).
    """

    # Simple wrapper for array indexing
    v = f[r, c]

    return v
