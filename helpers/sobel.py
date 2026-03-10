from typing import Any
import numpy as np

try:
    from helpers.imgradient import imgradient
except ImportError:
    # Package-relative fallback when imported from project root package context.
    from ..General.imgradient import imgradient


def sobel(f: Any):
    """
    [Gmag, Gdir] = sobel(f)

    Compute Sobel gradient magnitude and direction for a grayscale input
    image f. Outputs are floating-point arrays (double precision by default).
    """
    # MATLAB: f = im2double(f)
    f = np.asarray(f)
    if np.issubdtype(f.dtype, np.integer):
        info = np.iinfo(f.dtype)
        f = f.astype(np.float64) / float(info.max)
    else:
        f = f.astype(np.float64)

    # MATLAB equivalent:
    # [Gx, Gy] = imgradientxy(f, 'Sobel');
    # [Gmag, Gdir] = imgradient(Gx, Gy);
    # Using imgradient(f,'sobel') gives the same Gmag/Gdir pair.
    Gmag, Gdir = imgradient(f, "sobel")

    return Gmag, Gdir
