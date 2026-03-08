import numpy as np
from scipy.ndimage import rank_filter, median_filter


def adpmedian(g, Smax):
    """
    Adaptive median filtering (Gonzalez & Woods / DIPUM).

    Parameters
    ----------
    g : ndarray
        Input image (2D).
    Smax : int
        Maximum window size (odd integer > 1).

    Returns
    -------
    f : ndarray
        Filtered image.
    """

    # --- Parameter check (faithful to MATLAB code)
    if (Smax <= 1) or (Smax % 2 == 0) or (Smax != int(Smax)):
        raise ValueError("Smax must be an odd integer > 1")

    g = np.asarray(g)
    f = np.zeros_like(g)

    alreadyProcessed = np.zeros(g.shape, dtype=bool)

    # --- Main adaptive loop
    for k in range(3, Smax + 1, 2):
        window_size = (k, k)

        # ordfilt2 equivalents
        zmin = rank_filter(g, rank=0, size=window_size, mode='reflect')
        zmax = rank_filter(g, rank=k*k - 1, size=window_size, mode='reflect')

        # medfilt2 equivalent
        zmed = median_filter(g, size=window_size, mode='reflect')

        # Level A
        processUsingLevelB = (
            (zmed > zmin) &
            (zmax > zmed) &
            (~alreadyProcessed)
        )

        # Level B
        zB = (g > zmin) & (zmax > g)

        outputZxy  = processUsingLevelB & zB
        outputZmed = processUsingLevelB & (~zB)

        f[outputZxy]  = g[outputZxy]
        f[outputZmed] = zmed[outputZmed]

        alreadyProcessed |= processUsingLevelB

        if np.all(alreadyProcessed):
            break

    # --- Remaining pixels use final zmed
    f[~alreadyProcessed] = zmed[~alreadyProcessed]

    return f