import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric


def _as_source_points(source_points, ndim):
    sp = np.asarray(source_points, dtype=float)
    if sp.ndim == 1:
        sp = sp.reshape(ndim, 1)
    if sp.shape[0] != ndim and sp.shape[1] == ndim:
        sp = sp.T
    if sp.shape[0] != ndim:
        raise ValueError('SourcePoints must have shape (ndim, N) or (N, ndim).')
    return np.rint(sp).astype(int) - 1


def msfm(F, SourcePoints, UseSecond=False, UseCross=False):
    """
    Robust Python MSFM-compatible API.

    Notes
    -----
    - Uses skimage.graph.MCP_Geometric (compiled/Cython) for fast multi-source
      geodesic propagation.
    - `UseSecond` kept for API compatibility.
    - Returns (T, Y) where:
        T = geodesic travel-time map
        Y = Euclidean distance to nearest source point
    """
    F = np.asarray(F, dtype=float)
    if F.ndim not in (2, 3):
        raise ValueError('F must be a 2D or 3D array.')
    if np.any(F <= 0):
        raise ValueError('Speed image must be strictly positive.')

    shape = F.shape
    ndim = F.ndim
    sources = _as_source_points(SourcePoints, ndim)

    starts = []
    for k in range(sources.shape[1]):
        p = tuple(int(sources[d, k]) for d in range(ndim))
        if all(0 <= p[d] < shape[d] for d in range(ndim)):
            starts.append(p)
    if not starts:
        raise ValueError('No valid source points inside F bounds.')

    costs = 1.0 / np.maximum(F, np.finfo(float).tiny)
    mcp = MCP_Geometric(costs, fully_connected=bool(UseCross))
    T, _ = mcp.find_costs(starts=starts)
    T = T.astype(float, copy=False)

    src_mask = np.ones(shape, dtype=bool)
    for p in starts:
        src_mask[p] = False
    Y = distance_transform_edt(src_mask).astype(float, copy=False)
    Y[~np.isfinite(T)] = 0.0

    return T, Y
