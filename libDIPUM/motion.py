import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _im2col_distinct(img, n):
    rows, cols = img.shape
    if rows % n != 0 or cols % n != 0:
        raise ValueError('The image size must be divisible by the block size.')

    br = rows // n
    bc = cols // n
    out = np.empty((n * n, br * bc), dtype=img.dtype)
    k = 0
    # MATLAB-like block ordering: row blocks vary fastest (column-major in block grid)
    for by in range(bc):
        for bx in range(br):
            block = img[bx * n:(bx + 1) * n, by * n:(by + 1) * n]
            out[:, k] = block.reshape(-1, order='F')
            k += 1
    return out


def _col2im_distinct(cols, n, shape):
    rows, cols_n = shape
    br = rows // n
    bc = cols_n // n
    out = np.empty(shape, dtype=cols.dtype)
    k = 0
    for by in range(bc):
        for bx in range(br):
            block = cols[:, k].reshape((n, n), order='F')
            out[bx * n:(bx + 1) * n, by * n:(by + 1) * n] = block
            k += 1
    return out


def _im2col_sliding(img, n):
    r, c = img.shape
    msx = r - n + 1
    msy = c - n + 1
    out = np.empty((n * n, msx * msy), dtype=img.dtype)
    k = 0
    # MATLAB-like sliding column order: x varies fastest, then y
    for y in range(msy):
        for x in range(msx):
            out[:, k] = img[x:x + n, y:y + n].reshape(-1, order='F')
            k += 1
    return out


def _ci2i(col, n, mi):
    # 1-based in / out
    col = col - 1
    x = (n * (col % mi)) + 1
    y = 1 + ((col // mi) * n)
    return x, y


def _i2b(x, y, n):
    # 1-based
    bx = 1 + ((x - 1) // n)
    by = 1 + ((y - 1) // n)
    return bx, by


def _s2i(col, ms):
    # 1-based
    col = col - 1
    x = (col % ms) + 1
    y = 1 + (col // ms)
    return x, y


def _i2s(x, y, ms):
    # 1-based
    return x + (ms * (y - 1))


def _line(a, x0, y0, x1, y1):
    # All coordinates are 1-based image coordinates.
    ao = a.copy()
    rows, cols = ao.shape

    def set_px(x, y, val):
        if 1 <= x <= rows and 1 <= y <= cols:
            ao[x - 1, y - 1] = val

    dx = x1 - x0
    dy = y1 - y0
    if dx == 0 and dy == 0:
        return ao

    set_px(x0, y0, 255)

    if abs(dx) > abs(dy):  # slope < 1
        m = dy / dx if dx != 0 else 0.0
        b = y0 - m * x0
        sx = -1 if dx < 0 else 1
        while x0 != x1:
            x0 += sx
            yy = int(m * x0 + b)
            set_px(x0, yy, 0)
    else:  # slope >= 1
        if dy != 0:
            m = dx / dy
            b = x0 - m * y0
            sy = -1 if dy < 0 else 1
            while y0 != y1:
                y0 += sy
                xx = int(m * y0 + b)
                set_px(xx, y0, 0)

    return ao


def motion(i, j, n, delta, subP):
    """
    Motion estimation between current frame i and previous frame j.

    Parameters
    ----------
    i, j : 2-D ndarray
        Current and previous frames.
    n : int
        Macroblock size (even expected by original code).
    delta : array-like length 2
        +/- search shift [deltaX, deltaY].
    subP : float
        Sub-pixel step: 1, 0.5, or 0.25.

    Returns
    -------
    e : ndarray
        Error image.
    a : ndarray (uint8)
        Motion-arrow image.
    dx, dy : ndarray
        Block motion vectors (vertical and horizontal offsets).
    """
    if subP not in (1, 0.5, 0.25):
        raise ValueError('The sub-pixel resolution must be 1, 0.5, or 0.25.')

    i = np.asarray(i, dtype=float)
    j = np.asarray(j, dtype=float)
    sz = i.shape
    if i.ndim != 2 or j.ndim != 2 or j.shape != sz:
        raise ValueError('i and j must be 2-D arrays of equal shape.')

    if (sz[0] % n != 0) or (sz[1] % n != 0):
        raise ValueError('The image size must be divisible by the block size.')

    delta = np.asarray(delta).astype(int).ravel()
    if delta.size != 2:
        raise ValueError('delta must have two elements [deltaX deltaY].')

    a = np.zeros(sz, dtype=np.uint8) + 192
    ci = _im2col_distinct(i, n)

    # Sub-pixel interpolated previous frame (MATLAB interp2 with bilinear)
    y = np.arange(1, sz[0] + 1, dtype=float)
    x = np.arange(1, sz[1] + 1, dtype=float)
    yi = np.arange(1, sz[0] + 1e-12, subP, dtype=float)
    xi = np.arange(1, sz[1] + 1e-12, subP, dtype=float)

    interp = RegularGridInterpolator((y, x), j, method='linear', bounds_error=False, fill_value=None)
    YY, XX = np.meshgrid(yi, xi, indexing='ij')
    ji = interp(np.stack([YY.ravel(), XX.ravel()], axis=-1)).reshape(len(yi), len(xi))

    w = int(round(1.0 / subP))
    mssub = np.zeros((w, w), dtype=int)
    cji_map = {}

    for u1 in range(1, w + 1):
        for v1 in range(1, w + 1):
            jisub = ji[(u1 - 1)::w, (v1 - 1)::w]
            mssub[u1 - 1, v1 - 1] = jisub.shape[0]
            cji_map[(u1, v1)] = _im2col_sliding(jisub, n)

    mi = sz[0] // n
    bx_count = sz[0] // n
    by_count = sz[1] // n
    dx = np.zeros((bx_count, by_count), dtype=float)
    dy = np.zeros((bx_count, by_count), dtype=float)

    for col in range(1, ci.shape[1] + 1):
        x0, y0 = _ci2i(col, n, mi)  # 1-based

        curMad = np.inf
        ms = mssub[0, 0] - n + 1
        vector = [1, 1, _i2s(x0, y0, ms)]

        for u1 in range(1, w + 1):
            for v1 in range(1, w + 1):
                cji = cji_map[(u1, v1)]
                ms = mssub[u1 - 1, v1 - 1] - n + 1
                plus = 1 if (u1 == 1 and v1 == 1) else 0

                x1s = max(1, x0 - delta[0])
                x1e = min(sz[0] - n + plus, x0 + delta[0])
                y1s = max(1, y0 - delta[1])
                y1e = min(sz[1] - n + plus, y0 + delta[1])

                for x1 in range(x1s, x1e + 1):
                    for y1 in range(y1s, y1e + 1):
                        c1 = _i2s(x1, y1, ms)
                        mv = cji[:, c1 - 1] - ci[:, col - 1]
                        mad = np.sum(np.abs(mv))
                        if mad <= curMad:
                            curMad = mad
                            vector = [u1, v1, c1]

        best_cji = cji_map[(vector[0], vector[1])]
        ci[:, col - 1] = ci[:, col - 1] - best_cji[:, vector[2] - 1]

        ms = mssub[vector[0] - 1, vector[1] - 1] - n + 1
        vx, vy = _s2i(vector[2], ms)
        vx = vx + ((vector[0] - 1) * subP)
        vy = vy + ((vector[1] - 1) * subP)

        bx, by = _i2b(x0, y0, n)
        dx[bx - 1, by - 1] = vx - x0
        dy[bx - 1, by - 1] = vy - y0

        a = _line(a, x0, y0, int(round(vx)), int(round(vy)))

    e = _col2im_distinct(ci, n, sz)
    return e, a, dx, dy
