import numpy as np
import pywt

def wavedec(x, n, in3, in4=None):
    """
    Multi-level 1-D wavelet decomposition (MATLAB-compatible layout).

    [c, l] = wavedec(x, n, 'wname')
    [c, l] = wavedec(x, n, Lo_D, Hi_D)
    """
    try:
        import pywt
    except Exception as exc:
        raise ImportError("PyWavelets (pywt) is required for wavedec.") from exc

    x = np.asarray(x)
    if x.ndim != 1 and not (x.ndim == 2 and 1 in x.shape):
        raise ValueError("Input x must be a vector.")
    x_is_col = (x.ndim == 2 and x.shape[0] > 1)
    x = x.reshape(-1)

    if in4 is None:
        # Wavelet name
        wname = in3
        wavelet = pywt.Wavelet(wname)
    else:
        # Custom filters
        Lo_D = np.asarray(in3).reshape(-1)
        Hi_D = np.asarray(in4).reshape(-1)
        # Build custom wavelet filter bank: (dec_lo, dec_hi, rec_lo, rec_hi)
        # For decomposition only, rec filters can mirror dec filters.
        wavelet = pywt.Wavelet('custom', filter_bank=(Lo_D, Hi_D, Lo_D, Hi_D))

    # PyWavelets returns [cA_n, cD_n, ..., cD_1]
    # Use periodization to keep total coefficient length equal to len(x),
    # matching MATLAB wavedec output size expectations in DIP4e scripts.
    coeffs = pywt.wavedec(x, wavelet, level=int(n), mode='periodization')
    cA_n = coeffs[0]
    details = coeffs[1:]

    # Build C and L (MATLAB layout)
    c_parts = [cA_n] + details
    c = np.concatenate(c_parts)

    l = np.zeros(len(details) + 2, dtype=int)
    l[0] = len(cA_n)
    for i, d in enumerate(details, start=1):
        l[i] = len(d)
    l[-1] = len(x)

    if x_is_col:
        c = c.reshape(-1, 1)
        l = l.reshape(-1, 1)

    return c, l
