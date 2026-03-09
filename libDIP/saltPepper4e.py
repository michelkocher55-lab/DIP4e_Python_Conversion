from typing import Any
import numpy as np
from skimage import img_as_float


def saltPepper4e(f: Any, pp: Any, ps: Any):
    """
    Corrupts an image with salt-and-pepper noise.

    g = saltPepper4e(f, pp, ps)

    Parameters
    ----------
    f : numpy.ndarray
        Input image.
    pp : float
        Probability of pepper noise (0).
    ps : float
        Probability of salt noise (1).

    Returns
    -------
    g : numpy.ndarray
        Corrupted image with values in range [0, 1] (float).
    """

    # Check sum
    if (pp + ps) > 1.0:
        raise ValueError("The sum pp + ps must not exceed 1")

    # Scale to [0, 1]
    g = img_as_float(f).copy()

    # Generate random matrix [0, 1)
    X = np.random.random(g.shape)

    # Pepper: X <= pp -> 0
    g[X <= pp] = 0.0

    # Salt: pp < X <= pp + ps -> 1
    # Optimization with logical indexing
    # We can do: X > pp AND X <= (pp + ps)
    salt_mask = (X > pp) & (X <= (pp + ps))
    g[salt_mask] = 1.0

    return g
