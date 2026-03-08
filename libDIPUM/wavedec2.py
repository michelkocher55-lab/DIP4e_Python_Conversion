import numpy as np


def wavedec2(x, n, wname):
    """
    MATLAB-compatible wavedec2 using PyWavelets.
    Returns (c, s) where c is a row vector and s is bookkeeping matrix.
    """
    try:
        import pywt
    except Exception as exc:
        raise ImportError("PyWavelets (pywt) is required for wavedec2.") from exc

    x = np.asarray(x, dtype=float)
    coeffs = pywt.wavedec2(x, wname, level=int(n), mode='periodization')
    cA = coeffs[0]
    details = coeffs[1:]

    # Build s (bookkeeping)
    s_rows = [list(cA.shape)]
    for (cH, cV, cD) in details:
        s_rows.append(list(cH.shape))
    s_rows.append(list(x.shape))
    s = np.array(s_rows, dtype=int)

    # Build c vector in MATLAB order (column-major)
    c_list = [cA.ravel(order='F')]
    for (cH, cV, cD) in details:
        c_list.append(cH.ravel(order='F'))
        c_list.append(cV.ravel(order='F'))
        c_list.append(cD.ravel(order='F'))
    c = np.concatenate(c_list)
    c = c.reshape(1, -1)

    return c, s
