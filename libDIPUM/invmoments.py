import numpy as np


def _compute_m(f):
    M, N = f.shape
    x, y = np.meshgrid(np.arange(1, N + 1, dtype=float),
                       np.arange(1, M + 1, dtype=float))

    # Flatten for direct summations (MATLAB-style).
    x = x.ravel()
    y = y.ravel()
    fv = f.ravel()

    m = {}
    m['m00'] = np.sum(fv)
    if m['m00'] == 0:
        m['m00'] = np.finfo(float).eps

    m['m10'] = np.sum(x * fv)
    m['m01'] = np.sum(y * fv)
    m['m11'] = np.sum(x * y * fv)
    m['m20'] = np.sum((x ** 2) * fv)
    m['m02'] = np.sum((y ** 2) * fv)
    m['m30'] = np.sum((x ** 3) * fv)
    m['m03'] = np.sum((y ** 3) * fv)
    m['m12'] = np.sum(x * (y ** 2) * fv)
    m['m21'] = np.sum((x ** 2) * y * fv)

    return m


def _compute_eta(m):
    xbar = m['m10'] / m['m00']
    ybar = m['m01'] / m['m00']

    e = {}
    e['eta11'] = (m['m11'] - ybar * m['m10']) / (m['m00'] ** 2)
    e['eta20'] = (m['m20'] - xbar * m['m10']) / (m['m00'] ** 2)
    e['eta02'] = (m['m02'] - ybar * m['m01']) / (m['m00'] ** 2)

    e['eta30'] = (m['m30'] - 3 * xbar * m['m20'] + 2 * (xbar ** 2) * m['m10']) / (m['m00'] ** 2.5)
    e['eta03'] = (m['m03'] - 3 * ybar * m['m02'] + 2 * (ybar ** 2) * m['m01']) / (m['m00'] ** 2.5)

    e['eta21'] = (m['m21'] - 2 * xbar * m['m11'] - ybar * m['m20'] + 2 * (xbar ** 2) * m['m01']) / (m['m00'] ** 2.5)
    e['eta12'] = (m['m12'] - 2 * ybar * m['m11'] - xbar * m['m02'] + 2 * (ybar ** 2) * m['m10']) / (m['m00'] ** 2.5)

    return e


def _compute_phi(e):
    phi = np.zeros(7, dtype=float)

    phi[0] = e['eta20'] + e['eta02']
    phi[1] = (e['eta20'] - e['eta02']) ** 2 + 4 * (e['eta11'] ** 2)
    phi[2] = (e['eta30'] - 3 * e['eta12']) ** 2 + (3 * e['eta21'] - e['eta03']) ** 2
    phi[3] = (e['eta30'] + e['eta12']) ** 2 + (e['eta21'] + e['eta03']) ** 2

    phi[4] = (
        (e['eta30'] - 3 * e['eta12'])
        * (e['eta30'] + e['eta12'])
        * ((e['eta30'] + e['eta12']) ** 2 - 3 * (e['eta21'] + e['eta03']) ** 2)
        + (3 * e['eta21'] - e['eta03'])
        * (e['eta21'] + e['eta03'])
        * (3 * (e['eta30'] + e['eta12']) ** 2 - (e['eta21'] + e['eta03']) ** 2)
    )

    phi[5] = (
        (e['eta20'] - e['eta02'])
        * ((e['eta30'] + e['eta12']) ** 2 - (e['eta21'] + e['eta03']) ** 2)
        + 4 * e['eta11'] * (e['eta30'] + e['eta12']) * (e['eta21'] + e['eta03'])
    )

    phi[6] = (
        (3 * e['eta21'] - e['eta03'])
        * (e['eta30'] + e['eta12'])
        * ((e['eta30'] + e['eta12']) ** 2 - 3 * (e['eta21'] + e['eta03']) ** 2)
        + (3 * e['eta12'] - e['eta30'])
        * (e['eta21'] + e['eta03'])
        * (3 * (e['eta30'] + e['eta12']) ** 2 - (e['eta21'] + e['eta03']) ** 2)
    )

    return phi


def invmoments(f):
    """
    Compute invariant moments of image.

    Parameters
    ----------
    f : ndarray
        2D, real, nonsparse numeric/logical image.

    Returns
    -------
    phi : ndarray, shape (7,)
        Seven moment invariants (DIPUM3E Table 13.8).
    """
    arr = np.asarray(f)

    if arr.ndim != 2:
        raise ValueError('F must be a 2-D, real, nonsparse, numeric or logical matrix.')
    if np.iscomplexobj(arr):
        raise ValueError('F must be a 2-D, real, nonsparse, numeric or logical matrix.')
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_):
        raise ValueError('F must be a 2-D, real, nonsparse, numeric or logical matrix.')

    fd = arr.astype(float)

    return _compute_phi(_compute_eta(_compute_m(fd)))
