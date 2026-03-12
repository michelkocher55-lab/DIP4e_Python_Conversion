from typing import Any
from skimage.color import gray2rgb


def superpixels(
    A: Any, N: Any, method: Any = "slic", compactness: Any = 20, **kwargs: Any
):
    """
    Computes superpixels using the SLIC algorithm (Simple Linear Iterative Clustering).

    Parameters:
        A: Input image (2D or 3D).
        N: Desired number of superpixels.
        method: Algorithm to use. Only 'slic' is supported.
        compactness: Balances color proximity and space proximity.
                     Default 20.
        **kwargs: Additional arguments passed to skimage.segmentation.slic.

    Returns:
        L: Label matrix (M, N) with 1-based labels.
        NumLabels: Actual number of superpixels found.
    """
    try:
        from skimage.segmentation import slic
    except ImportError:
        raise ImportError(
            "skimage is required for superpixels. Please install scikit-image."
        )

    # Handle Grayscale scaling issue:
    # skimage.slic on 2D float [0,1] uses raw intensity for distance (~0.1).
    # MATLAB converts grayscale to Lab (L=0..100).
    # To match MATLAB behavior and make 'compactness' parameter comparable,
    # we convert grayscale to RGB, which slic then converts to Lab.
    # This scales intensity importance by ~100x.

    image = A
    if A.ndim == 2:
        image = gray2rgb(A)

    try:
        # start_label=1 to mimic MATLAB's 1-based indexing
        segments = slic(
            image, n_segments=N, compactness=compactness, start_label=1, **kwargs
        )
    except TypeError:
        # Fallback for older skimage
        segments = slic(image, n_segments=N, compactness=compactness, **kwargs)
        if segments.min() == 0:
            segments += 1

    L = segments
    NumLabels = L.max()

    return L, NumLabels
