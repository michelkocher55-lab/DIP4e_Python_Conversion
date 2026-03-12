from typing import Any
import numpy as np
from scipy.signal import convolve2d

from helpers.libdipum.wavefilter import wavefilter
from helpers.libdipum.wavecopy import wavecopy
from helpers.libgeneral.padarray import padarray


def waveback(c: Any, s: Any, *args: Any):
    """
    Inverse FWT for decomposition [c, s], matching DIPUM waveback.
    """
    # Fast path: use waverec2 for full reconstruction (matches wavefast format)
    if (
        len(args) >= 1
        and isinstance(args[0], str)
        and (len(args) == 1 or isinstance(args[1], str))
    ):
        from helpers.libdipum.waverec2 import waverec2

        return waverec2(c, s, args[0], mode="symmetric")
    # Parse inputs
    wname = args[0]
    filterchk = False
    nchk = False

    if len(args) == 1:
        if isinstance(wname, str):
            lp, hp = wavefilter(wname, "r")
            n = s.shape[0] - 2
            extmode = "SYM"
        else:
            raise ValueError("Undefined filter.")
    elif len(args) == 2:
        if isinstance(wname, str):
            lp, hp = wavefilter(wname, "r")
            if isinstance(args[1], str):
                extmode = args[1]
                n = s.shape[0] - 2
            else:
                n = int(args[1])
                extmode = "SYM"
        else:
            lp, hp = args[0], args[1]
            filterchk = True
            n = s.shape[0] - 2
            extmode = "SYM"
    elif len(args) == 3:
        if isinstance(wname, str):
            lp, hp = wavefilter(wname, "r")
            extmode = args[1]
            n = int(args[2])
            nchk = True
        else:
            lp, hp = args[0], args[1]
            filterchk = True
            if isinstance(args[2], str):
                extmode = args[2]
                n = s.shape[0] - 2
            else:
                n = int(args[2])
                extmode = "SYM"
    elif len(args) == 4:
        lp, hp = args[0], args[1]
        filterchk = True
        extmode = args[2]
        n = int(args[3])
        nchk = True
    else:
        raise ValueError("Improper number of input arguments.")

    lp = np.asarray(lp).reshape(-1)
    hp = np.asarray(hp).reshape(-1)
    fl = len(lp)

    if filterchk:
        if fl != len(hp) or fl % 2 != 0:
            raise ValueError("LP and HP must be even and equal length.")

    nmax = s.shape[0] - 2
    if n < 1 or n > nmax:
        raise ValueError("Invalid number (N) of reconstructions requested.")

    nc = np.asarray(c).reshape(-1)
    ns = np.asarray(s, dtype=int)
    nnmax = nmax

    for _ in range(n):
        a = (
            _convup(wavecopy("a", nc, ns), lp, lp, fl, ns[2, :2], extmode)
            + _convup(wavecopy("h", nc, ns, nnmax), hp, lp, fl, ns[2, :2], extmode)
            + _convup(wavecopy("v", nc, ns, nnmax), lp, hp, fl, ns[2, :2], extmode)
            + _convup(wavecopy("d", nc, ns, nnmax), hp, hp, fl, ns[2, :2], extmode)
        )

        # Update decomposition
        drop = 4 * int(np.prod(ns[0, :2]))
        nc = nc[drop:]
        nc = np.concatenate([a.ravel(order="F"), nc])
        ns = ns[2:, :]
        ns = np.vstack([ns[0, :], ns])
        nnmax = ns.shape[0] - 2

    # Complete reconstruction
    a = nc
    out = np.zeros(tuple(ns[0, :2]), dtype=float)
    out.ravel(order="F")[:] = a
    return out


def _convup(x: Any, f1: Any, f2: Any, fln: Any, keep: Any, extmode: Any):
    """_convup."""
    # MATLAB indices are 1-based. For SYM:
    # zi = fln-1 : fln+keep(1)-2  (inclusive)
    # Convert to 0-based: start = fln-2, stop = fln+keep(1)-2 (exclusive)
    zi = np.arange(fln - 2, fln + keep[0] - 2)
    zj = np.arange(fln - 2, fln + keep[1] - 2)

    if extmode.upper() == "SYM":
        y = np.zeros((x.shape[0] * 2, x.shape[1]))
        y[::2, :] = x
        y = convolve2d(y, f1.reshape(-1, 1), mode="full")
        z = np.zeros((y.shape[0], y.shape[1] * 2))
        z[:, ::2] = y
        z = convolve2d(z, f2.reshape(1, -1), mode="full")
        z = z[zi[:, None], zj]
    else:
        y = np.zeros((x.shape[0] * 2, x.shape[1]))
        y[::2, :] = x
        y = padarray(y, [len(f1) // 2, 0], "circular", "both")
        y = convolve2d(y, f1.reshape(-1, 1), mode="full")
        z = np.zeros((y.shape[0], y.shape[1] * 2))
        z[:, ::2] = y
        z = padarray(z, [0, len(f2) // 2], "circular", "both")
        z = convolve2d(z, f2.reshape(1, -1), mode="full")
        z = z[fln - 1 : fln - 1 + keep[0], fln - 1 : fln - 1 + keep[1]]

    return z
