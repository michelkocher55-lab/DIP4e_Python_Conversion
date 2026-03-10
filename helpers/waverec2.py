from typing import Any
import numpy as np

from helpers.wavefilter import wavefilter


def waverec2(c: Any, s: Any, wname: Any, mode: Any = "symmetric"):
    """
    MATLAB-compatible waverec2 using PyWavelets.
    Expects c as row vector and s as bookkeeping matrix.
    """
    try:
        import pywt
    except Exception as exc:
        raise ImportError("PyWavelets (pywt) is required for waverec2.") from exc

    c = np.asarray(c).reshape(-1)
    s = np.asarray(s, dtype=int)
    n = s.shape[0] - 2

    # First approximation size
    a_shape = tuple(s[0])
    idx = 0
    a_size = int(np.prod(a_shape))
    cA = c[idx : idx + a_size].reshape(a_shape, order="F")
    idx += a_size

    details = []
    for level in range(n):
        d_shape = tuple(s[1 + level])
        d_size = int(np.prod(d_shape))
        cH = c[idx : idx + d_size].reshape(d_shape, order="F")
        idx += d_size
        cV = c[idx : idx + d_size].reshape(d_shape, order="F")
        idx += d_size
        cD = c[idx : idx + d_size].reshape(d_shape, order="F")
        idx += d_size
        details.append((cH, cV, cD))

    coeffs = [cA] + details

    name = str(wname).lower()
    if name == "jpeg9.7":
        # Build exact wavelet from DIPUM filter bank.
        ld, hd = wavefilter("jpeg9.7", "d")
        lr, hr = wavefilter("jpeg9.7", "r")
        w = pywt.Wavelet(
            "jpeg9_7_dipum",
            filter_bank=[ld.tolist(), hd.tolist(), lr.tolist(), hr.tolist()],
        )
        x = pywt.waverec2(coeffs, w, mode=mode)
    else:
        x = pywt.waverec2(coeffs, wname, mode=mode)
    return x
