import numpy as np
from scipy import sparse
from scipy.ndimage import label, binary_fill_holes, binary_erosion

try:
    from General.qtdecomp import qtdecomp
    from General.qtgetblk import qtgetblk
    from General.qtsetblk import qtsetblk
except Exception:
    from qtdecomp import qtdecomp
    from qtgetblk import qtgetblk
    from qtsetblk import qtsetblk

try:
    from libDIPUM.bwboundaries import bwboundaries
    from libDIPUM.boundarydir import boundarydir
except Exception:
    from bwboundaries import bwboundaries
    from boundarydir import boundarydir


def im2minperpoly(I, cellsize):
    """Minimum perimeter polygon approximation for a single binary region.

    Parameters
    ----------
    I : ndarray
        Binary image containing one region/boundary.
    cellsize : int
        Cell size (>1) for the cellular complex.

    Returns
    -------
    X, Y : ndarray
        Polygon vertex coordinates (row, col) as 1D vectors.
    R : ndarray
        Region enclosed by the cellular complex.
    """
    cellsize = int(cellsize)
    if cellsize <= 1:
        raise ValueError('cellsize must be an integer > 1.')

    I = np.asarray(I)
    I = I > 0

    # MATLAB bwlabel default connectivity is 8.
    lbl, num = label(I, structure=np.ones((3, 3), dtype=bool))
    if num > 1:
        raise ValueError('Input image cannot contain more than one region.')

    R = cellcomplex(I, cellsize)
    X, Y = mppvertices(R, cellsize)
    return X, Y, R


def cellcomplex(I, cellsize):
    """Compute 4-connected region enclosed by the cellular complex."""
    I = np.asarray(I).astype(bool)

    # Fill and extract 4-connected perimeter.
    I = binary_fill_holes(I)
    se4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    I = I & (~binary_erosion(I, structure=se4, border_value=0))

    M, N = I.shape

    # Compute padded size K = cellsize * power_of_2 >= max(M,N).
    ratio = max(M, N) / float(cellsize)
    K = int((2 ** int(np.ceil(np.log2(max(ratio, 1.0))))) * cellsize)

    Ipad = np.zeros((K, K), dtype=np.uint8)
    Ipad[:M, :N] = I.astype(np.uint8)

    # Quadtree decomposition and extraction of cellsize blocks.
    Q = qtdecomp(Ipad, threshold=0, min_dim=cellsize)
    vals, r, c = qtgetblk(Ipad, Q, cellsize)

    if vals.size == 0:
        return np.zeros((M, N), dtype=bool)

    # Keep blocks containing at least one boundary pixel.
    sums = vals.reshape(vals.shape[0], -1).sum(axis=1)
    idx = np.where(sums >= 1)[0]

    if len(idx) == 0:
        return np.zeros((M, N), dtype=bool)

    rr = r[idx]
    cc = c[idx]

    # Build sparse block-selector and fill selected blocks with 1 using qtsetblk.
    Ssel = sparse.coo_matrix(
        (np.full(len(rr), cellsize, dtype=int), (rr, cc)), shape=Ipad.shape
    ).tocsr()

    values = np.ones((cellsize, cellsize, len(rr)), dtype=Ipad.dtype)
    Ifilled = qtsetblk(Ipad, Ssel, cellsize, values)

    BF = binary_fill_holes(Ifilled > 0)
    R = BF & (~(Ifilled > 0))
    return R[:M, :N]


def mppvertices(R, cellsize):
    """Output MPP vertices around region R."""
    B = bwboundaries(R, conn=4)
    if not B:
        return np.array([]), np.array([])

    B = np.asarray(B[0])
    if len(B) > 1 and np.all(B[0] == B[-1]):
        B = B[:-1]

    if len(B) == 0:
        return np.array([]), np.array([])

    x = B[:, 0].astype(int)
    y = B[:, 1].astype(int)

    L = vertexlist(x, y, cellsize)
    NV = L.shape[0]
    if NV == 0:
        return np.array([]), np.array([])

    count = 1  # 1-based to match MATLAB logic

    X = [int(L[0, 0])]
    Y = [int(L[0, 1])]

    cMPPV = np.array([L[0, 0], L[0, 1]], dtype=int)
    cWH = cMPPV.copy()
    cBL = cMPPV.copy()

    while True:
        count += 1
        if count > NV + 1:
            break

        if count == NV + 1:
            cV = np.array([L[0, 0], L[0, 1]], dtype=int)
            classV = int(L[0, 2])
        else:
            cV = np.array([L[count - 1, 0], L[count - 1, 1]], dtype=int)
            classV = int(L[count - 1, 2])

        Inew, newMPPV, W, Bk = mppVtest(cMPPV, cV, classV, cWH, cBL)

        if Inew == 1:
            cMPPV = newMPPV
            kidx = np.where((L[:, 0] == newMPPV[0]) & (L[:, 1] == newMPPV[1]))[0]
            if len(kidx) == 0:
                break
            count = int(kidx[0]) + 1
            cWH = newMPPV.copy()
            cBL = newMPPV.copy()
            X.append(int(newMPPV[0]))
            Y.append(int(newMPPV[1]))
        else:
            cWH = W
            cBL = Bk

    return np.asarray(X), np.asarray(Y)


def vertexlist(x, y, cellsize):
    x = np.asarray(x).astype(int).ravel()
    y = np.asarray(y).astype(int).ravel()

    # Remove duplicate points (consecutive and non-consecutive) while
    # preserving traversal order. boundarydir requires uniqueness.
    pts = np.column_stack([x, y])
    if len(pts) > 1:
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
        pts = pts[keep]
    if len(pts) > 1:
        _, first_idx = np.unique(pts, axis=0, return_index=True)
        pts = pts[np.sort(first_idx)]
    x = pts[:, 0]
    y = pts[:, 1]

    # Top-left-most starting point.
    cx = np.where(x == np.min(x))[0]
    x1 = x[cx[0]]
    y1 = np.min(y[cx])

    I = np.where((x == x1) & (y == y1))[0][0]
    x = np.roll(x, -I)
    y = np.roll(y, -I)

    # Keep only direction-change points.
    K = len(x)
    xext = np.concatenate([x, [x[0]]])
    yext = np.concatenate([y, [y[0]]])

    xnew = [xext[0]]
    ynew = [yext[0]]
    for k in range(1, K):
        s = vsign([xext[k - 1], yext[k - 1]], [xext[k], yext[k]], [xext[k + 1], yext[k + 1]])
        if s != 0:
            xnew.append(xext[k])
            ynew.append(yext[k])

    x = np.asarray(xnew, dtype=int)
    y = np.asarray(ynew, dtype=int)

    # Force counter-clockwise order.
    _, x, y = boundarydir(x, y, orderout='ccw')
    x = np.asarray(x).astype(int)
    y = np.asarray(y).astype(int)

    K = len(x)
    if K == 0:
        return np.zeros((0, 3), dtype=int)

    L = np.column_stack([x, y, np.zeros(K, dtype=int)])
    C = np.zeros(K, dtype=int)

    # First vertex.
    s = vsign([x[K - 1], y[K - 1]], [x[0], y[0]], [x[1], y[1]])
    if s > 0:
        C[0] = 1
    elif s < 0:
        C[0] = -1
        rx, ry = vreplacement([x[K - 1], y[K - 1]], [x[0], y[0]], [x[1], y[1]], cellsize)
        L[0, 0], L[0, 1] = rx, ry

    # Last vertex.
    s = vsign([x[K - 2], y[K - 2]], [x[K - 1], y[K - 1]], [x[0], y[0]])
    if s > 0:
        C[K - 1] = 1
    elif s < 0:
        C[K - 1] = -1
        rx, ry = vreplacement([x[K - 2], y[K - 2]], [x[K - 1], y[K - 1]], [x[0], y[0]], cellsize)
        L[K - 1, 0], L[K - 1, 1] = rx, ry

    # Middle vertices.
    for k in range(1, K - 1):
        s = vsign([x[k - 1], y[k - 1]], [x[k], y[k]], [x[k + 1], y[k + 1]])
        if s > 0:
            C[k] = 1
        elif s < 0:
            C[k] = -1
            rx, ry = vreplacement([x[k - 1], y[k - 1]], [x[k], y[k]], [x[k + 1], y[k + 1]], cellsize)
            L[k, 0], L[k, 1] = rx, ry

    L[:, 2] = C
    return L


def vsign(v1, v2, v3):
    A = np.array(
        [[v1[0], v1[1], 1], [v2[0], v2[1], 1], [v3[0], v3[1], 1]], dtype=float
    )
    return int(np.round(np.linalg.det(A)))


def vreplacement(v1, v, v2, cellsize):
    if v[0] > v1[0] and v[1] == v1[1] and v[0] == v2[0] and v[1] > v2[1]:
        return v[0] - cellsize, v[1] - cellsize
    if v[0] == v1[0] and v[1] > v1[1] and v[0] < v2[0] and v[1] == v2[1]:
        return v[0] + cellsize, v[1] - cellsize
    if v[0] < v1[0] and v[1] == v1[1] and v[0] == v2[0] and v[1] < v2[1]:
        return v[0] + cellsize, v[1] + cellsize
    if v[0] == v1[0] and v[1] < v1[1] and v[0] > v2[0] and v[1] == v2[1]:
        return v[0] - cellsize, v[1] + cellsize
    raise ValueError('Vertex configuration is not valid.')


def mppVtest(cMPPV, cV, classcV, cWH, cBL):
    I = 0
    newMPPV = np.array([0, 0], dtype=int)
    W = cWH.copy()
    B = cBL.copy()

    sW = vsign(cMPPV, cWH, cV)
    sB = vsign(cMPPV, cBL, cV)

    if sW > 0:
        I = 1
        newMPPV = cWH.copy()
        W = newMPPV.copy()
        B = newMPPV.copy()
    elif sB < 0:
        I = 1
        newMPPV = cBL.copy()
        W = newMPPV.copy()
        B = newMPPV.copy()
    elif (sW <= 0) and (sB >= 0):
        if classcV == 1:
            W = cV.copy()
        else:
            B = cV.copy()

    return I, newMPPV, W, B


__all__ = [
    'im2minperpoly',
    'cellcomplex',
    'mppvertices',
    'vertexlist',
    'vsign',
    'vreplacement',
    'mppVtest',
]
