from typing import Any
import numpy as np
from scipy import ndimage


def localmean(f: Any, nhood: Any):
    """
    Computes local mean of image f using neighborhood nhood.
    Equivalent to convolution with normalized nhood.
    """
    nhood = np.array(nhood, dtype=np.float64)
    # Normalize neighborhood kernel
    kernel = nhood / np.sum(nhood)

    # Convolve
    # MATLAB: imfilter(f, nhood, 'replicate')
    # scipy.ndimage.convolve uses 'nearest' for 'replicate'?
    # 'reflect' is default. 'nearest' repeats edge.
    # MATLAB 'replicate': Input array values outside the bounds of the array are assumed to equal the nearest array border value.
    # SciPy 'nearest': The input is extended by replicating the last pixel.
    mean = ndimage.convolve(f, kernel, mode="nearest")
    return mean


def stdfilt_custom(f: Any, nhood: Any):
    """
    Computes local standard deviation using efficiently.
    std = sqrt( E[x^2] - (E[x])^2 )

    Note: This formula can be unstable for small variances if not careful with float precision.
    Using float64 is recommended.
    """
    f = f.astype(np.float64)

    # E[x]
    Ex = localmean(f, nhood)

    # E[x^2]
    Ex2 = localmean(f**2, nhood)

    # Var = E[x^2] - (E[x])^2
    var = Ex2 - Ex**2

    # Clip negative values due to precision errors
    var = np.maximum(var, 0)

    std = np.sqrt(var)
    return std


def localthresh(f: Any, nhood: Any, a: Any, b: Any, meantype: Any = "local"):
    """
    Local thresholding.

    G = LOCALTHRESH(F, NHOOD, A, B, MEANTYPE)

    Parameters:
        f: Input image.
        nhood: Neighborhood mask (0s and 1s).
        a: Constant for std deviation weight.
        b: Constant for mean weight.
        meantype: 'local' (default) or 'global'.

    Returns:
        g: Segmented image (boolean/binary).
    """
    f = f.astype(np.float64)

    # Local std deviation
    # In MATLAB stdfilt(f, nhood) calculates std of the neighbors (excluding center? No, usually including if nhood says so).
    # MATLAB stdfilt doc: "calculates the local standard deviation... of 3-by-3 neighborhood".
    # nhood defines the domain.
    SIG = stdfilt_custom(f, nhood)

    # Mean
    if meantype == "global":
        MEAN = np.mean(f)
    else:
        # 'local'
        MEAN = localmean(f, nhood)

    # Segment
    # g = (f > a*SIG) & (f > b*MEAN)
    g = (f > a * SIG) & (f > b * MEAN)

    return g
