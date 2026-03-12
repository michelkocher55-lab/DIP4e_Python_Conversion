from typing import Any
import warnings
import numpy as np


def _getrangefromclass(arr: Any):
    """_getrangefromclass."""
    dt = np.asarray(arr).dtype
    if np.issubdtype(dt, np.bool_):
        return np.array([0.0, 1.0], dtype=float)
    if np.issubdtype(dt, np.floating):
        return np.array([0.0, 1.0], dtype=float)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return np.array([float(info.min), float(info.max)], dtype=float)
    raise TypeError("Unsupported input dtype for GrayLimits default.")


def _parse_kwargs(I: Any, kwargs: Any):
    """_parse_kwargs."""
    # Defaults
    offset = np.array([[0, 1]], dtype=float)
    if np.asarray(I).dtype == np.bool_:
        nl = 2.0
    else:
        nl = 8.0
    gl = _getrangefromclass(I)
    sym = False

    # MATLAB-like parameter abbreviations (case-insensitive)
    valid = ["offset", "numlevels", "graylimits", "symmetric"]

    for k, v in kwargs.items():
        key = str(k).lower()
        matches = [name for name in valid if name.startswith(key)]
        if len(matches) == 0:
            raise ValueError(f"Unknown parameter '{k}'.")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous parameter abbreviation '{k}'.")
        p = matches[0]

        if p == "offset":
            off = np.asarray(v)
            if off.ndim != 2 or off.shape[1] != 2:
                raise ValueError("Offset must be a p-by-2 array.")
            if not np.all(np.isfinite(off)):
                raise ValueError("Offset must be finite integers.")
            if not np.all(np.equal(off, np.round(off))):
                raise ValueError("Offset values must be integers.")
            offset = off.astype(float)

        elif p == "numlevels":
            nlv = float(v)
            if nlv < 0 or round(nlv) != nlv:
                raise ValueError("NumLevels must be a nonnegative integer.")
            if np.asarray(I).dtype == np.bool_ and int(nlv) != 2:
                raise ValueError("NumLevels must be 2 if I is logical.")
            nl = float(int(nlv))

        elif p == "graylimits":
            glv = np.asarray(v, dtype=float)
            if glv.size == 0:
                arr = np.asarray(I, dtype=float)
                gl = np.array([np.nanmin(arr), np.nanmax(arr)], dtype=float)
            else:
                if glv.ndim != 1 or glv.size != 2:
                    raise ValueError("GrayLimits must be a two-element vector or [].")
                gl = glv.astype(float)

        elif p == "symmetric":
            if not isinstance(v, (bool, np.bool_)):
                raise ValueError("Symmetric must be a logical scalar.")
            sym = bool(v)

    return offset, int(nl), gl.astype(float), sym


def _compute_glcm_for_offset(si: Any, offset_rc: Any, nl: Any):
    """_compute_glcm_for_offset."""
    n_row, n_col = si.shape

    rr, cc = np.meshgrid(
        np.arange(1, n_row + 1), np.arange(1, n_col + 1), indexing="ij"
    )
    r = rr.ravel()
    c = cc.ravel()

    r2 = r + int(offset_rc[0])
    c2 = c + int(offset_rc[1])

    valid = (r2 >= 1) & (r2 <= n_row) & (c2 >= 1) & (c2 <= n_col)

    v1 = si.ravel()[valid]
    idx_r2 = r2[valid] - 1
    idx_c2 = c2[valid] - 1
    v2 = si[idx_r2, idx_c2].reshape(-1)

    bad = np.isnan(v1) | np.isnan(v2)
    if np.any(bad):
        warnings.warn("scaledImageContainsNan", RuntimeWarning)
        v1 = v1[~bad]
        v2 = v2[~bad]

    one = np.zeros((nl, nl), dtype=float)
    if v1.size == 0:
        return one

    # SI values are in [1..NL], convert to 0-based indices.
    i1 = v1.astype(int) - 1
    i2 = v2.astype(int) - 1

    good = (i1 >= 0) & (i1 < nl) & (i2 >= 0) & (i2 < nl)
    i1 = i1[good]
    i2 = i2[good]

    np.add.at(one, (i1, i2), 1)
    return one


def graycomatrix(I: Any, **kwargs: Any):
    """
    Create gray-level co-occurrence matrix (GLCM).

    Parameters (MATLAB-like names, abbreviations allowed):
      Offset, NumLevels, GrayLimits, Symmetric

    Returns
    -------
    GLCMS : ndarray, shape (NumLevels, NumLevels, p)
    SI    : ndarray, scaled image (double), values in [1..NumLevels]
    """
    arr = np.asarray(I)
    if arr.ndim != 2:
        raise ValueError("I must be 2D.")
    if np.iscomplexobj(arr):
        raise ValueError("I must be real.")

    offset, nl, gl, make_symmetric = _parse_kwargs(arr, kwargs)

    arrd = arr.astype(float)

    # Scale to [1..NL]
    if gl[1] == gl[0]:
        si = np.ones(arr.shape, dtype=float)
    else:
        slope = nl / (gl[1] - gl[0])
        intercept = 1.0 - slope * gl[0]
        si = np.floor(slope * arrd + intercept)

    # Handle inf and clipping analogous to MATLAB behavior.
    si[np.isposinf(si)] = nl
    si[np.isneginf(si)] = 1
    si[si > nl] = nl
    si[si < 1] = 1

    num_offsets = offset.shape[0]

    if nl == 0:
        glcms = np.zeros((0, 0, num_offsets), dtype=float)
        return glcms, si

    glcms = np.zeros((nl, nl, num_offsets), dtype=float)
    for k in range(num_offsets):
        one = _compute_glcm_for_offset(si, offset[k, :], nl)
        if make_symmetric:
            one = one + one.T
        glcms[:, :, k] = one

    return glcms, si
