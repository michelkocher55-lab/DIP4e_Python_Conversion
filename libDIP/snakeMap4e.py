from typing import Any
import math
from libDIP.intScaling4e import intScaling4e
from libDIP.gaussKernel4e import gaussKernel4e
from libDIP.twodConv4e import twodConv4e
from General.edge import edge


def snakeMap4e(
    f: Any, T: Any = None, sig: Any = None, nsig: Any = None, order: Any = "none"
):
    """
    Computes edge map for use in snake iterative algorithm.

    Parameters
    ----------
    f : numpy.ndarray
        Input image.
    T : float, str, or None, optional
        Threshold for edge map.
        - If None (default), no thresholding is performed (equivalent to edge(..., 0)).
        - If 'auto', thresholding is done automatically.
        - If float, uses specified threshold.
    sig : float, optional
        Standard deviation of Gaussian kernel. If None, no filtering is performed.
    nsig : float, optional
        Size of kernel in sigmas.
    order : str, optional
        'before', 'after', 'both', or 'none'. Determines when filtering is applied.

    Returns
    -------
    emap : numpy.ndarray
        Edge map (scaled to [0, 1]).
    """

    # -Preliminaries.
    # --Scale image to the floating point intensity range [0,1].
    f = intScaling4e(f)

    # --Set defaults.
    thresholding = True
    filtering = True

    # Check arguments to mimick MATLAB nargin behavior
    if T is None and sig is None and nsig is None and order is None:
        # Only one input (f). No thresholding.
        thresholding = False
        filtering = False
    elif sig is None and nsig is None and order is None:
        # Only two inputs (f, T).
        filtering = False

    # -If filtering is called for, form the Gaussian lowpass kernel.
    if filtering:
        # --Form Gaussian kernel using function gaussKernel4e.
        kernelSize = math.floor(nsig * sig)
        # --Find the closest odd number of kernelSize.
        if kernelSize % 2 == 0:
            kernelSize = kernelSize + 1
        # --Gaussian kernel.
        w = gaussKernel4e(kernelSize, sig)

    # -Check to see if filtering is to be done here.
    if filtering:
        if order == "before" or order == "both":
            f = twodConv4e(f, w)

    # -Compute edge map using the Sobel kernels.
    if thresholding:
        if isinstance(T, str) and T == "auto":
            # emap = double (edge (f, 'sobel'));
            emap = edge(f, method="sobel", threshold=None)
        else:
            # emap = double (edge (f, 'sobel', T));
            emap = edge(f, method="sobel", threshold=T)
    else:
        # ---If there is not thresholding, just use edge with a 0 threshold.
        # emap = double (edge (f, 'sobel', 0));
        emap = edge(f, method="sobel", threshold=0)

    # Cast to float (double)
    emap = emap.astype(float)

    # -Check to see if filtering was called for.
    if filtering:
        # --If order is 'after', or 'both' then filter now.
        if order == "after" or order == "both":
            # emap = twodConv4e(intScaling4e(emap,'full','floating'),w,'replicate');
            # Use 'ns' in twodConv4e to avoid default scaling, since we scale explicitly.
            scaled_emap = intScaling4e(emap, mode="full", type_out="floating")
            emap = twodConv4e(scaled_emap, w, param="ns")

    # -Scale output emap intensities to the range [0,1].
    emap = intScaling4e(emap, mode="full", type_out="floating")

    return emap
