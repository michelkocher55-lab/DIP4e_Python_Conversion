import numpy as np
import ia870 as ia


def AddSE2Image(f, B, gris):
    """
    MATLAB translation of AddSE2Image(f, B, gris).

    Parameters
    ----------
    f : ndarray
        Input image (binary, uint8, or uint16).
    B : structuring element
        ia870 structuring element.
    gris : int
        Gray level of the SE overlay.

    Returns
    -------
    g : ndarray
        Output image, same shape as f.
    """
    f = np.asarray(f)
    BImg = np.asarray(ia.iaseshow(B))

    # mmdatatype equivalent
    if f.dtype == np.bool_:
        g = np.zeros(f.shape, dtype=bool)
    elif f.dtype == np.uint8:
        if gris == 0:
            g = np.full(f.shape, 255, dtype=np.uint8)
        else:
            g = np.zeros(f.shape, dtype=np.uint8)
    elif f.dtype == np.uint16:
        if gris == 0:
            g = np.full(f.shape, 255, dtype=np.uint16)
        else:
            g = np.zeros(f.shape, dtype=np.uint16)
    else:
        raise ValueError('Data type not supported')

    # MATLAB: g(round(size(BImg,1)/2)+5, round(size(BImg,2)/2)+5) = gris;
    r = int(round(BImg.shape[0] / 2.0) + 5) - 1
    c = int(round(BImg.shape[1] / 2.0) + 5) - 1
    r = max(0, min(r, g.shape[0] - 1))
    c = max(0, min(c, g.shape[1] - 1))

    if g.dtype == np.bool_:
        g[r, c] = bool(gris)
    else:
        g[r, c] = np.asarray(gris, dtype=g.dtype)

    if gris == 0:
        g = ia.iaero(g, B)
        g = ia.iaintersec(g, f)
    else:
        g = ia.iadil(g, B)
        g = ia.iaunion(g, f)

    return g
