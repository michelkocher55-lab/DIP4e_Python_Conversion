import numpy as np


_ALL_STATS = ['Contrast', 'Correlation', 'Energy', 'Homogeneity']


def _validate_glcm(glcm):
    g = np.asarray(glcm)

    if g.ndim > 3:
        raise ValueError('GLCM must be 2D or 3D.')

    if np.iscomplexobj(g):
        raise ValueError('GLCM must be real.')

    if np.any(~np.isfinite(g)):
        raise ValueError('GLCM must be finite.')

    if np.any(g < 0):
        raise ValueError('GLCM must be nonnegative.')

    # MATLAB requires integer-valued entries.
    if not np.all(np.equal(g, np.round(g))):
        raise ValueError('GLCM must contain integer-valued entries.')

    return g.astype(float, copy=False)


def _normalize_glcm(glcm2d):
    s = np.sum(glcm2d)
    if s != 0:
        return glcm2d / s
    return glcm2d


def _mean_index(index, glcm):
    return np.sum(index * glcm.ravel())


def _std_index(index, glcm, m):
    return np.sqrt(np.sum(((index - m) ** 2) * glcm.ravel()))


def _calculate_contrast(glcm, r, c):
    term1 = np.abs(r - c) ** 2
    term = term1 * glcm.ravel()
    return np.sum(term)


def _calculate_correlation(glcm, r, c):
    mr = _mean_index(r, glcm)
    sr = _std_index(r, glcm, mr)

    mc = _mean_index(c, glcm)
    sc = _std_index(c, glcm, mc)

    term2 = np.sum((r - mr) * (c - mc) * glcm.ravel())

    denom = sr * sc
    if denom == 0:
        return np.nan
    return term2 / denom


def _calculate_energy(glcm):
    return np.sum(glcm ** 2)


def _calculate_homogeneity(glcm, r, c):
    term = glcm.ravel() / (1.0 + np.abs(r - c))
    return np.sum(term)


def _parse_properties(properties):
    if properties is None:
        req = list(_ALL_STATS)
        req.sort()
        return req

    # Accept: string 'all', space-separated string, list/tuple of strings.
    if isinstance(properties, str):
        tokens = properties.split()
        if len(tokens) == 0:
            req = list(_ALL_STATS)
            req.sort()
            return req
        items = tokens
    elif isinstance(properties, (list, tuple)):
        if len(properties) == 0:
            req = list(_ALL_STATS)
            req.sort()
            return req
        items = list(properties)
    else:
        # MATLAB also allows passing several comma-separated strings; in Python
        # users can pass a tuple/list for equivalent behavior.
        items = [properties]

    anyprop = _ALL_STATS + ['all']
    req = []

    for k, raw in enumerate(items, start=1):
        if not isinstance(raw, str):
            raise ValueError(f'Invalid property at position {k}. Must be a string.')
        key = raw.strip().lower()
        matches = [name for name in anyprop if name.lower().startswith(key)]
        if len(matches) == 0:
            raise ValueError(f"Unknown property '{raw}'.")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous property abbreviation '{raw}'.")
        m = matches[0]
        if m == 'all':
            req = list(_ALL_STATS)
            break
        req.append(m)

    req = sorted(set(req))
    if len(req) == 0:
        raise ValueError('No valid properties were requested.')
    return req


def graycoprops(glcm, properties='all'):
    """
    Properties of gray-level co-occurrence matrix.

    Parameters
    ----------
    glcm : ndarray
        m x n or m x n x p gray-level co-occurrence matrix array.
    properties : str or list/tuple of str, optional
        Requested properties; abbreviations are accepted:
        Contrast, Correlation, Energy, Homogeneity, or 'all'.

    Returns
    -------
    stats : dict
        Keys are requested properties. Each value is a 1D numpy array
        of length p (or 1 for a single 2D GLCM).
    """
    g = _validate_glcm(glcm)
    req_stats = _parse_properties(properties)

    if g.ndim == 2:
        g = g[:, :, np.newaxis]

    num_glcm = g.shape[2]
    stats = {name: np.zeros(num_glcm, dtype=float) for name in req_stats}

    for p in range(num_glcm):
        tglcm = _normalize_glcm(g[:, :, p])

        s = tglcm.shape
        # Keep MATLAB ordering used in the source:
        # [c,r] = meshgrid(1:s(1),1:s(2)); r=r(:); c=c(:)
        c, r = np.meshgrid(np.arange(1, s[0] + 1), np.arange(1, s[1] + 1))
        r = r.ravel().astype(float)
        c = c.ravel().astype(float)

        for name in req_stats:
            if name == 'Contrast':
                stats[name][p] = _calculate_contrast(tglcm, r, c)
            elif name == 'Correlation':
                stats[name][p] = _calculate_correlation(tglcm, r, c)
            elif name == 'Energy':
                stats[name][p] = _calculate_energy(tglcm)
            elif name == 'Homogeneity':
                stats[name][p] = _calculate_homogeneity(tglcm, r, c)

    return stats
