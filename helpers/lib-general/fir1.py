from typing import Any
import numpy as np
from scipy.signal import firwin


def _scale_filter(b: Any, first_band: Any, freq: Any, L: Any):
    """_scale_filter."""
    b = np.asarray(b, dtype=float)
    if first_band:
        s = np.sum(b)
        return b / s if s != 0 else b

    if np.isclose(freq[3], 1.0):
        f0 = 1.0
    else:
        f0 = np.mean(freq[2:4])

    n = np.arange(L)
    Hf0 = np.abs(np.exp(-1j * 2.0 * np.pi * n * (f0 / 2.0)) @ b)
    return b / Hf0 if Hf0 != 0 else b


def fir1(N: Any, Wn: Any, *args: Any):
    """
    MATLAB-like FIR1 using the window method.

    Supported options in *args (any order):
    - filter type: 'low', 'high', 'bandpass', 'stop', 'dc-0', 'dc-1'
    - window vector (length N+1)
    - 'scale' / 'noscale'

    Returns
    -------
    b, a : ndarray, ndarray
        FIR numerator and denominator (a = [1.0]).
    """
    N = int(np.asarray(N).item())
    if N <= 0:
        raise ValueError("N must be positive")

    Wn = np.asarray(Wn, dtype=float).ravel()
    if Wn.size == 0:
        raise ValueError("Wn must be nonempty")
    if np.any(Wn <= 0) or np.any(Wn >= 1):
        raise ValueError("fir1:FreqsOutOfRange")
    if np.any(np.diff(Wn) < 0):
        raise ValueError("fir1:FreqsMustBeMonotonic")

    # Defaults like MATLAB fir1
    if Wn.size == 1:
        ftype = "LOW"
    elif Wn.size == 2:
        ftype = "BANDPASS"
    else:
        ftype = "DC-0"

    scaling = True
    window_vec = None

    ftype_options = {"LOW", "HIGH", "BANDPASS", "STOP", "DC-0", "DC-1"}
    scale_options = {"SCALE", "NOSCALE"}

    for arg in args:
        if isinstance(arg, str):
            s = arg.strip().upper()
            if s in ftype_options:
                ftype = s
            elif s in scale_options:
                scaling = s == "SCALE"
            elif s == "H":
                raise NotImplementedError(
                    "Hilbert ('h') option is not implemented in this fir1.py"
                )
            else:
                raise ValueError(f"Unknown option: {arg}")
        else:
            arr = np.asarray(arg)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError("Window must be a nonempty vector")
            if window_vec is not None:
                raise ValueError("Conflicting window specifications")
            window_vec = arr.astype(float)

    # For Nyquist-pass filters, order must be even (MATLAB behavior)
    nyq_pass = ftype in {"HIGH", "STOP", "DC-1"}
    if nyq_pass and (N % 2 == 1):
        N += 1

    L = N + 1

    if window_vec is not None and len(window_vec) != L:
        raise ValueError("signal:fir1:MismatchedWindowLength")

    # Map to scipy firwin pass_zero
    if ftype == "LOW":
        cutoff = Wn[0]
        pass_zero = True
    elif ftype == "HIGH":
        cutoff = Wn[0]
        pass_zero = False
    elif ftype == "BANDPASS":
        if Wn.size < 2:
            raise ValueError("Bandpass requires two cutoff frequencies")
        cutoff = [Wn[0], Wn[1]]
        pass_zero = False
    elif ftype == "STOP":
        if Wn.size < 2:
            raise ValueError("Bandstop requires two cutoff frequencies")
        cutoff = [Wn[0], Wn[1]]
        pass_zero = True
    elif ftype == "DC-0":
        cutoff = Wn
        pass_zero = False
    elif ftype == "DC-1":
        cutoff = Wn
        pass_zero = True
    else:
        raise ValueError(f"Unsupported filter type: {ftype}")

    # Frequency vector used for MATLAB-like manual scaling when needed.
    ff = np.repeat(Wn, 2)
    freq = np.concatenate(([0.0], ff, [1.0]))
    first_band = (ftype != "DC-0") and (ftype != "HIGH")

    if window_vec is None:
        b = firwin(
            numtaps=L,
            cutoff=cutoff,
            window="hamming",
            pass_zero=pass_zero,
            scale=scaling,
            fs=2.0,
        )
    else:
        # Build ideal FIR with boxcar, then apply user window manually.
        b0 = firwin(
            numtaps=L,
            cutoff=cutoff,
            window="boxcar",
            pass_zero=pass_zero,
            scale=False,
            fs=2.0,
        )
        b = b0 * window_vec
        if scaling:
            b = _scale_filter(b, first_band, freq, L)

    a = np.array([1.0], dtype=float)
    return b, a
