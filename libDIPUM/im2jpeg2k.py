from typing import Any
import numpy as np

from libDIPUM.mat2huff import mat2huff
from libDIPUM.wavecopy import wavecopy
from libDIPUM.wavefast import wavefast
from libDIPUM.wavepaste import wavepaste


def stepsize(n: Any, p: Any):
    """
    Create subband quantization step sizes ordered by decomposition level.
    """
    p = np.asarray(p, dtype=float).reshape(-1)

    if p.size == 2:  # Implicit quantization.
        q = []
        qn = 2 ** (8 - p[1] + n) * (1 + p[0] / 2**11)
        for k in range(1, n + 1):
            qk = 2 ** (-k) * qn
            q.extend([2 * qk, 2 * qk, 4 * qk])
        q.append(qk)
        q = np.asarray(q, dtype=float)
    else:  # Explicit quantization.
        q = p.astype(float)

    q = np.round(q * 100) / 100

    if np.any(100 * q > 65535):
        raise ValueError("The quantizing steps are not UINT16 representable.")
    if np.any(q == 0):
        raise ValueError("A quantizing step of 0 is not allowed.")

    return q


def im2jpeg2k(x: Any, n: Any, q: Any):
    """
    Compress an image using a JPEG 2000 approximation.

    Parameters
    ----------
    x : ndarray
        Input uint8 2-D image.
    n : int
        Number of wavelet decomposition levels.
    q : array_like
        Quantization parameters. Length must be 2 or 3*n + 1.

    Returns
    -------
    y : dict
        Encoding structure for decoding.
    store : dict
        Debug/intermediate structures (compatibility with MATLAB function).
    """
    x = np.asarray(x)
    q = np.asarray(q, dtype=float).reshape(-1)

    if x.ndim != 2 or not np.isrealobj(x) or x.dtype != np.uint8:
        raise ValueError("The input must be a UINT8 image.")

    if q.size != 2 and q.size != (3 * int(n) + 1):
        raise ValueError("The quantization step size vector is bad.")

    n = int(n)

    # Level shift and transform.
    x_shift = x.astype(float) - 128
    c, s = wavefast(x_shift, n, "jpeg9.7")

    store = {
        "c": c.copy(),
        "s": s.copy(),
    }

    # Quantize wavelet coefficients.
    qv = stepsize(n, q)
    store["q"] = qv.copy()

    sgn = np.sign(c)
    sgn[sgn == 0] = 1
    c = np.abs(c)

    qi = None
    for k in range(1, n + 1):
        qi = 3 * k - 2
        c = wavepaste("h", c, s, k, wavecopy("h", c, s, k) / qv[qi - 1])
        c = wavepaste("v", c, s, k, wavecopy("v", c, s, k) / qv[qi])
        c = wavepaste("d", c, s, k, wavecopy("d", c, s, k) / qv[qi + 1])

    # MATLAB uses last k/qi from the loop here.
    c = wavepaste("a", c, s, n, wavecopy("a", c, s, n) / qv[qi + 2])

    c = np.floor(c)
    c = c * sgn

    store["cq"] = c.copy()
    store["sq"] = s.copy()

    # Run-length coding of long zero runs.
    zrc = np.min(c) - 1
    eoc = zrc - 1

    runs = [65535]

    z = (c == 0).astype(int)
    z = z - np.r_[0, z[:-1]]
    plus = np.where(z == 1)[0]
    minus = np.where(z == -1)[0]

    # Remove any terminating zero run.
    if plus.size != minus.size:
        c = c[: plus[-1]]
        c = np.r_[c, eoc]

    def runcode(xrun: Any):
        """runcode."""
        for idx, rv in enumerate(runs, start=1):
            if rv == xrun:
                return idx
        runs.append(int(xrun))
        return len(runs)

    # Replace internal long runs (>10) by run codes.
    for i in range(minus.size - 1, -1, -1):
        run = int(minus[i] - plus[i])
        if run > 10:
            ovrflo = run // 65535
            run = run - ovrflo * 65535
            prefix = c[: plus[i]]
            rep = np.tile(np.array([zrc, 1.0]), ovrflo)
            rc = runcode(run)
            suffix = c[minus[i] :]
            c = np.r_[prefix, rep, zrc, rc, suffix]

    y = {
        "runs": np.asarray(runs, dtype=np.uint16),
        "s": np.asarray(s, dtype=np.uint16).reshape(-1, order="F"),
        "zrc": np.uint16(-zrc),
        "q": np.asarray(np.round(100 * qv), dtype=np.uint16),
        "n": np.uint16(n),
        "huffman": mat2huff(c),
    }

    return y, store
