import numpy as np

from libDIPUM.huff2mat import huff2mat
from libDIPUM.waveback import waveback
from libDIPUM.wavecopy import wavecopy
from libDIPUM.wavepaste import wavepaste


def jpeg2k2im(y):
    """
    Decode an IM2JPEG2K compressed image.

    Parameters
    ----------
    y : dict
        Encoding structure returned by im2jpeg2k.

    Returns
    -------
    x : ndarray
        Reconstructed uint8 image approximation.
    """
    if not isinstance(y, dict):
        raise ValueError("Input must be a dictionary structure produced by im2jpeg2k.")

    required = ('n', 'q', 'runs', 'zrc', 's', 'huffman')
    for key in required:
        if key not in y:
            raise KeyError(f"Missing key '{key}' in input structure.")

    # Decode parameters.
    n = int(np.asarray(y['n']).reshape(-1)[0])
    q = np.asarray(y['q'], dtype=float).reshape(-1) / 100.0
    runs = np.asarray(y['runs'], dtype=float).reshape(-1)
    zrc = -float(np.asarray(y['zrc']).reshape(-1)[0])
    eoc = zrc - 1

    s_flat = np.asarray(y['s'], dtype=float).reshape(-1)
    if s_flat.size != (n + 2) * 2:
        raise ValueError("Bookkeeping array size is inconsistent with N.")
    s = s_flat.reshape((n + 2, 2), order='F')

    # Compute total coefficient length.
    cl = int(np.prod(s[0, :]))
    for i in range(1, n + 1):
        cl += 3 * int(np.prod(s[i, :]))

    # Huffman decode then run-length decode.
    r = np.asarray(huff2mat(y['huffman']), dtype=float).reshape(-1)

    c_parts = []
    zpos = np.where(r == zrc)[0]
    i = 0
    for j in zpos:
        c_parts.append(r[i:j])
        if j + 1 >= r.size:
            raise ValueError("Malformed run-length code stream.")
        ridx = int(r[j + 1])
        if ridx < 1 or ridx > len(runs):
            raise ValueError("Run-length table index out of range.")
        c_parts.append(np.zeros(int(runs[ridx - 1]), dtype=float))
        i = j + 2

    epos = np.where(r == eoc)[0]
    if epos.size == 1:
        c_parts.append(r[i:epos[0]])
        c = np.concatenate(c_parts) if c_parts else np.array([], dtype=float)
        if c.size < cl:
            c = np.r_[c, np.zeros(cl - c.size, dtype=float)]
        elif c.size > cl:
            c = c[:cl]
    else:
        c_parts.append(r[i:])
        c = np.concatenate(c_parts) if c_parts else np.array([], dtype=float)

    # Denormalize quantized coefficients.
    c = c + (c > 0).astype(float) - (c < 0).astype(float)

    qi = None
    for k in range(1, n + 1):
        qi = 3 * k - 2
        c = wavepaste('h', c, s, k, wavecopy('h', c, s, k) * q[qi - 1])
        c = wavepaste('v', c, s, k, wavecopy('v', c, s, k) * q[qi])
        c = wavepaste('d', c, s, k, wavecopy('d', c, s, k) * q[qi + 1])

    c = wavepaste('a', c, s, n, wavecopy('a', c, s, n) * q[qi + 2])

    # Inverse wavelet transform and level shift.
    x = waveback(c, s, 'jpeg9.7')
    x = np.clip(np.round(x + 128), 0, 255).astype(np.uint8)
    return x
