from typing import Any
from skimage import img_as_float
from skimage.filters import sobel, threshold_otsu, gaussian
from skimage.morphology import thin


def snakeMap4e(f: Any, T: Any = None, sig: Any = 1, nsig: Any = 3, order: Any = "none"):
    """
    Computes edge map for use in snake iterative algorithm.

    emap = snakeMap4e(f, T=None, sig=1, nsig=3, order='none')

    Parameters
    ----------
    f : numpy.ndarray
        Input image.
    T : float, 'auto', or None
        Threshold for edge map.
        - None: Returns Gradient Magnitude (Continuous).
        - 'auto': Uses Otsu thresholding -> Binary Thinned Edges.
        - float: Uses specified threshold -> Binary Thinned Edges.
    sig : float
        Standard deviation of Gaussian kernel for filtering.
    nsig : float
        Size of kernel in sigmas (kernel size = floor(nsig*sig)).
    order : str
        'before', 'after', 'both', or 'none'. Determines when filtering is applied.

    Returns
    -------
    emap : numpy.ndarray
        Edge map (float in [0, 1]).
    """

    f = img_as_float(f)
    order = order.lower()

    # 1. Pre-filtering
    if order in ["before", "both"]:
        # Truncate governs the size. default is 4.0 in skimage.
        # MATLAB: size = floor(nsig*sig).
        # skimage gaussian: sigma=sig.
        f = gaussian(f, sigma=sig, truncate=nsig)

    # 2. Compute Gradient Magnitude
    # sobel(f) computes magnitude sqrt(dx^2 + dy^2)
    emap = sobel(f)

    # Normalizing Magnitude? sobel output is theoretically bounded but scales with values.
    # Usually we want it normalized for thresholding comparisons.
    # However, let's proceed.

    # 3. Thresholding
    if T is not None:
        # Determine Threshold
        if isinstance(T, str) and T.lower() == "auto":
            try:
                thresh_val = threshold_otsu(emap)
            except ValueError:  # e.g. uniform image
                thresh_val = 0
        else:
            thresh_val = float(T)

        # Binarize
        binary_map = emap > thresh_val

        # Thinning (Morphological)
        # matches MATLAB edge() behavior which thins edges
        thinned_map = thin(binary_map)

        emap = img_as_float(thinned_map)

    # 4. Post-filtering
    if order in ["after", "both"]:
        emap = gaussian(emap, sigma=sig, truncate=nsig)

    # 5. Normalize to [0, 1]
    if emap.max() > 0:
        emap = emap / emap.max()

    return emap
