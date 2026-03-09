from typing import Any
import numpy as np
from skimage.io import imread
from numpy.lib.stride_tricks import sliding_window_view

from libDIPUM.huff2mat import huff2mat
from libDIPUM.im2jpeg import im2jpeg
from libDIPUM.jpeg2im import jpeg2im
from libDIPUM.mat2huff import mat2huff


def _load_multiframe_tiff(path: Any):
    """_load_multiframe_tiff."""
    arr = np.asarray(imread(path))

    if arr.ndim == 2:
        return [arr]

    if arr.ndim == 3:
        # TIFF stack is typically (frames, rows, cols).
        if arr.shape[-1] in (3, 4) and arr.shape[0] != arr.shape[1]:
            raise ValueError("Input must be a grayscale multi-frame TIFF.")
        return [arr[i, :, :] for i in range(arr.shape[0])]

    raise ValueError("Input must be a 2-D frame or 3-D grayscale frame stack.")


def _get_distinct_block(mat: Any, m: Any, col: Any, rows: Any):
    """_get_distinct_block."""
    # MATLAB col index is 1-based; here col is 0-based.
    x = 1 + (m * col) % rows
    y = 1 + m * ((col * m) // rows)
    blk = mat[x - 1 : x - 1 + m, y - 1 : y - 1 + m]
    return blk, x, y


def _blocks_to_cols_distinct(mat: Any, m: Any):
    """_blocks_to_cols_distinct."""
    rows, cols = mat.shape
    nb = (rows // m) * (cols // m)
    out = np.zeros((m * m, nb), dtype=float)
    for col in range(nb):
        blk, _, _ = _get_distinct_block(mat, m, col, rows)
        out[:, col] = blk.reshape(-1, order="F")
    return out


def _cols_to_blocks_distinct(cols_data: Any, out_shape: Any, m: Any):
    """_cols_to_blocks_distinct."""
    rows, cols = out_shape
    nb = (rows // m) * (cols // m)
    out = np.zeros((rows, cols), dtype=float)
    for col in range(nb):
        _, x, y = _get_distinct_block(out, m, col, rows)
        out[x - 1 : x - 1 + m, y - 1 : y - 1 + m] = cols_data[:, col].reshape(
            (m, m), order="F"
        )
    return out


def tifs2cv(f: Any, m: Any, d: Any, q: Any = 0):
    """
    Compress a multi-frame TIFF image sequence.
    MATLAB-faithful translation of DIPUM tifs2cv.m.

    Parameters
    ----------
    f : str
        Path to multi-frame TIFF.
    m : int
        Macroblock size.
    d : sequence of 2 ints
        Search displacement [dx, dy].
    q : float/int, optional
        JPEG quality for im2jpeg. If 0, lossless Huffman coding is used.

    Returns
    -------
    dict
        Encoding structure with fields:
        blksz, frames, quality, motion, video.
    """
    d = np.asarray(d).astype(int).reshape(-1)
    if d.size != 2:
        raise ValueError("D must contain two displacement values [dx, dy].")

    frames = _load_multiframe_tiff(f)
    if len(frames) == 0:
        raise ValueError("No frames found in TIFF file.")

    # Compress frame 1 and reconstruct for the initial reference frame.
    if q == 0:
        cv = [mat2huff(frames[0])]
        r = huff2mat(cv[0]).astype(float)
    else:
        cv = [im2jpeg(frames[0], q)]
        r = jpeg2im(cv[0]).astype(float)

    fsz = r.shape

    # Verify dimensions are multiples of block size.
    if (fsz[0] % m) != 0 or (fsz[1] % m) != 0:
        raise ValueError("Image dimensions must be multiples of the block size.")

    fcnt = len(frames)
    mvsz = (fsz[0] // m, fsz[1] // m, 2, fcnt)
    mv = np.zeros(mvsz, dtype=float)

    # For all frames except the first, compute motion-compensated residuals.
    for i in range(1, fcnt):
        frm = np.asarray(frames[i], dtype=float)
        frmC = _blocks_to_cols_distinct(frm, m)
        eC = np.zeros_like(frmC)

        for col in range(frmC.shape[1]):
            lookfor = frmC[:, col].reshape((m, m), order="F")

            x = 1 + (m * col) % fsz[0]
            y = 1 + m * ((col * m) // fsz[0])
            x1 = max(1, x - d[0])
            x2 = min(fsz[0], x + m + d[0] - 1)
            y1 = max(1, y - d[1])
            y2 = min(fsz[1], y + m + d[1] - 1)

            here = r[x1 - 1 : x2, y1 - 1 : y2]
            wins = sliding_window_view(here, (m, m))
            nr, nc = wins.shape[:2]

            # SAD for all candidate blocks (vectorized).
            s = np.sum(np.abs(wins - lookfor), axis=(2, 3))
            mins = np.min(s)

            # Match MATLAB find() order: column-major scan.
            lin = np.flatnonzero(s.reshape(-1, order="F") == mins)
            sx_all = (lin % nr) + 1
            sy_all = (lin // nr) + 1

            ns = np.abs(sx_all) + np.abs(sy_all)
            n = int(np.flatnonzero(ns == np.min(ns))[0])
            sx = int(sx_all[n])
            sy = int(sy_all[n])

            bi = (x - 1) // m
            bj = (y - 1) // m
            mv[bi, bj, :, i] = [x - (x1 + sx - 1), y - (y1 + sy - 1)]

            best = wins[sx - 1, sy - 1] - lookfor
            eC[:, col] = best.reshape(-1, order="F")

        # Code residual and reconstruct for next reference frame.
        e = _cols_to_blocks_distinct(eC, fsz, m)
        if q == 0:
            cvi = mat2huff(np.int16(np.round(e)))
            e = huff2mat(cvi).astype(float)
        else:
            cvi = im2jpeg(np.uint16(np.round(e + 255)), q, 9)
            e = jpeg2im(cvi).astype(float) - 255
        cv.append(cvi)

        # Decode next reference frame.
        rC = _blocks_to_cols_distinct(e, m)
        for col in range(rC.shape[1]):
            u = 1 + (m * col) % fsz[0]
            v = 1 + m * ((col * m) // fsz[0])

            bi = (u - 1) // m
            bj = (v - 1) // m
            rx = int(round(u - mv[bi, bj, 0, i]))
            ry = int(round(v - mv[bi, bj, 1, i]))

            temp = r[rx - 1 : rx - 1 + m, ry - 1 : ry - 1 + m]
            rC[:, col] = temp.reshape(-1, order="F") - rC[:, col]

        r = _cols_to_blocks_distinct(np.uint16(rC).astype(float), fsz, m)

    y = {
        "blksz": np.uint16(m),
        "frames": np.uint16(fcnt),
        "quality": np.uint16(q),
        "motion": mat2huff(mv.reshape(-1, order="F")),
        "video": cv,
    }
    return y
