from typing import Any
import numpy as np
from skimage.filters.rank import equalize
from skimage.morphology import rectangle
from lib.imPad4e import imPad4e
from lib.intScaling4e import intScaling4e


def localHistEqual4e(f: Any, m: Any = 3, n: Any = 3):
    """
    Local histogram equalization.

    Parameters:
    -----------
    f : numpy.ndarray
        Input image. Must be uint8.
    m : int, optional
        Height of the neighborhood. Default is 3.
    n : int, optional
        Width of the neighborhood. Default is 3.

    Returns:
    --------
    g : numpy.ndarray
        Local histogram equalized image (uint8).
    """
    f = np.array(f)

    # Input must be of class uint8
    if f.dtype != np.uint8:
        raise ValueError("f must be an 8-bit image with integer values (uint8).")

    if m % 2 == 0 or n % 2 == 0:
        raise ValueError("The dimensions of the neighborhood must be odd.")

    # Pad f using imPad4e (replicate padding)
    # We pad by half the kernel size.
    pad_r = m // 2
    pad_c = n // 2

    f_padded = imPad4e(f, pad_r, pad_c, padtype="replicate")

    # Ensure padded image is uint8 (imPad4e might return float if input was float-ish,
    # but input is uint8. imPad4e preserves dtype usually?
    # Let's check imPad4e... checking previous file view...
    # imPad4e uses np.pad. If f is uint8, np.pad returns uint8.

    # Perform local histogram equalization using skimage.filters.rank.equalize
    # rank.equalize(image, selem)
    # The 'selem' defines the neighborhood.
    # We use a rectangle of size (m, n).
    selem = rectangle(m, n)

    # Apply to the padded image.
    # Note: rank.equalize handles boundaries by reflection or similar defaults usually.
    # But here we are feeding it an already padded image.
    # We want the output 'g_padded' to have correct values in the valid region.
    # The valid region starts at pad_r, pad_c.
    # The neighborhood for a pixel in the valid region is fully contained within f_padded.
    # So boundary handling of rank.equalize shouldn't interfere with the central valid region.

    g_padded = equalize(f_padded, selem)

    # Remove padding
    # Original shape
    rows, cols = f.shape[:2]

    # Crop
    # g = g_padded[pad_r : pad_r + rows, pad_c : pad_c + cols]
    # Use end-based indexing for safety
    if pad_r > 0:
        g = g_padded[pad_r:-pad_r, :]
    else:
        g = g_padded

    if pad_c > 0:
        g = g[:, pad_c:-pad_c]

    # Output should be of class uint8 with correct scaling (rank.equalize returns uint8).
    # Use intScaling4e as requested to finalize.
    # 'integer' mode suggests 0..255 output.
    g = intScaling4e(g, type_out="integer")

    return g
