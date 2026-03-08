import numpy as np
from libDIPUM.wavecut import wavecut
from libDIPUM.wavecopy import wavecopy
from General.padarray import padarray


def _mat2gray(a, v=None):
    a = np.array(a, dtype=float)
    if v is None:
        vmin = np.min(a)
        vmax = np.max(a)
    else:
        vmin, vmax = v
    if vmax == vmin:
        return np.zeros_like(a)
    return (a - vmin) / (vmax - vmin)


def wavedisplay(c, s, scale=1, border='absorb'):
    """
    Display wavelet decomposition coefficients and return image.
    """
    c = np.asarray(c).reshape(-1)
    s = np.asarray(s)

    absflag = scale < 0
    scale = abs(scale)
    if scale == 0:
        scale = 1

    cd, w = wavecut('a', c, s)
    w = _mat2gray(w)
    cdx = np.max(np.abs(cd)) / scale

    if absflag:
        cd = _mat2gray(np.abs(cd), [0, cdx])
        fill = 0.0
    else:
        cd = _mat2gray(cd, [-cdx, cdx])
        fill = 0.5

    # Build gray image one decomposition at a time.
    for i in range(s.shape[0] - 2, 0, -1):
        ws = w.shape

        h = wavecopy('h', cd, s, i)
        pad = np.array(ws) - np.array(h.shape)
        front = np.round(pad / 2).astype(int)
        h = padarray(h, front, fill, 'pre')
        h = padarray(h, pad - front, fill, 'post')

        v = wavecopy('v', cd, s, i)
        pad = np.array(ws) - np.array(v.shape)
        front = np.round(pad / 2).astype(int)
        v = padarray(v, front, fill, 'pre')
        v = padarray(v, pad - front, fill, 'post')

        d = wavecopy('d', cd, s, i)
        pad = np.array(ws) - np.array(d.shape)
        front = np.round(pad / 2).astype(int)
        d = padarray(d, front, fill, 'pre')
        d = padarray(d, pad - front, fill, 'post')

        if border.lower() == 'append':
            w = padarray(w, [1, 1], 1, 'post')
            h = padarray(h, [1, 0], 1, 'post')
            v = padarray(v, [0, 1], 1, 'post')
        elif border.lower() == 'absorb':
            w[:, -1] = 1
            w[-1, :] = 1
            h[-1, :] = 1
            v[:, -1] = 1
        else:
            raise ValueError("Unrecognized BORDER parameter.")

        w = np.block([[w, h], [v, d]])

    return w
