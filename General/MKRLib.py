"""MKRLib compatibility helpers (Python 3).

Ported from legacy MKRLib __init__.py with pragmatic cleanups while preserving
original function names and behavior as much as possible.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from skimage.draw import line_aa
import ia870 as ia


def mmreadgray(filename: Any):
    """Read image and return a grayscale array.

    If input is RGB/RGBA, returns channel 1 (legacy behavior).
    """
    im = plt.imread(filename)
    if im.ndim == 2:
        return im
    if im.ndim == 3:
        if im.shape[2] >= 2:
            return im[:, :, 1]
        return im[:, :, 0]
    raise ValueError("Unsupported image dimensions")


def iabggmodel(dim: Any, B: Any = None, R: Any = 3, p: Any = 0.01):
    """iabggmodel."""
    if B is None:
        B = ia.iasecross()
    dy, dx = dim
    a = ia.iabinary(np.random.rand(dy, dx) < p)
    b = np.random.rand(dy, dx) * (R - 1) + 1.5
    a = ia.iaintersec(ia.iagray(a), b)
    y = ia.iaskelmrec(a, B)
    return y


def iaropen(f: Any, l: Any, rtheta: Any):
    """iaropen."""
    y = ia.iaintersec(f, 0)
    for t in rtheta:
        g = ia.iaopen(f, ia.iaseline(l, t))
        y = ia.iaunion(y, g)
    return y


def newline(p1: Any, p2: Any):
    """newline."""
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    ln = mlines.Line2D([xmin, xmax], [ymin, ymax])
    ax.add_line(ln)
    return ln


def DrawSEAxis(f: Any, dX: Any = 20, dY: Any = 20):
    """Legacy helper: pad a binary structuring element canvas."""
    g = np.zeros((f.shape[0] + 2 * dY, f.shape[1] + 2 * dX), dtype=bool)
    g[dY:-dY, dX:-dX] = f
    return g


def draw_se_axis(f: Any, dX: Any = 20, dY: Any = 20):
    """Draw padded SE with simple x/y axes and arrowheads.

    Returns an RGB uint8 image.
    """
    g = DrawSEAxis(f, dX=dX, dY=dY)
    h, w = g.shape
    yc = h // 2
    xc = w // 2

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[g] = 255

    axis_color = np.array([136, 136, 136], dtype=np.uint8)

    rr, cc, _ = line_aa(yc, xc, yc, w - 1)
    rgb[rr, cc] = axis_color
    rr, cc, _ = line_aa(yc, xc, 0, xc)
    rgb[rr, cc] = axis_color

    # Arrow heads
    for r1, c1, r2, c2 in [
        (yc - 2, w - 3, yc, w - 1),
        (yc + 2, w - 3, yc, w - 1),
        (2, xc - 2, 0, xc),
        (2, xc + 2, 0, xc),
    ]:
        rr, cc, _ = line_aa(r1, c1, r2, c2)
        rr = np.clip(rr, 0, h - 1)
        cc = np.clip(cc, 0, w - 1)
        rgb[rr, cc] = axis_color

    return rgb


def magic(n: Any):
    """magic."""
    n = int(n)
    if n < 3:
        raise ValueError("Size must be at least 3")
    if n % 2 == 1:
        p = np.arange(1, n + 1)
        return (
            n * np.mod(p[:, None] + p - (n + 3) // 2, n)
            + np.mod(p[:, None] + 2 * p - 2, n)
            + 1
        )
    if n % 4 == 0:
        j = np.mod(np.arange(1, n + 1), 4) // 2
        k = j[:, None] == j
        m = np.arange(1, n * n + 1, n)[:, None] + np.arange(n)
        m[k] = n * n + 1 - m[k]
        return m

    p = n // 2
    m = magic(p)
    m = np.block([[m, m + 2 * p * p], [m + 3 * p * p, m + p * p]])
    i = np.arange(p)
    k = (n - 2) // 4
    j = np.concatenate((np.arange(k), np.arange(n - k + 1, n)))
    m[np.ix_(np.concatenate((i, i + p)), j)] = m[np.ix_(np.concatenate((i + p, i)), j)]
    m[np.ix_([k, k + p], [0, k])] = m[np.ix_([k + p, k], [0, k])]
    return m


def mmlblshow(fl: Any = 0):
    """mmlblshow."""
    temp = ia.iaglblshow(fl)
    return np.moveaxis(temp, 0, -1)


def mmshow(
    m: Any = 0,
    RedOv: Any = 0,
    GreenOv: Any = 0,
    BlueOv: Any = 0,
    MagentaOv: Any = 0,
    YellowOv: Any = 0,
    CyanOv: Any = 0,
):
    """mmshow."""
    m = np.asarray(m)
    if m.ndim != 2:
        raise ValueError("m must be a 2-D grayscale array")

    base = m.astype(np.uint8)
    n = np.dstack([base, base, base])

    overlays = [
        (RedOv, np.array([255, 0, 0], dtype=np.uint8)),
        (GreenOv, np.array([0, 255, 0], dtype=np.uint8)),
        (BlueOv, np.array([0, 0, 255], dtype=np.uint8)),
        (MagentaOv, np.array([255, 0, 255], dtype=np.uint8)),
        (YellowOv, np.array([255, 255, 0], dtype=np.uint8)),
        (CyanOv, np.array([0, 255, 255], dtype=np.uint8)),
    ]

    for ov, color in overlays:
        try:
            mask = np.asarray(ov) == 1
            if mask.shape == m.shape:
                n[mask] = color
        except Exception:
            pass

    return n


def SSIMIndex(X: Any, Y: Any, k1: Any = 0.01, k2: Any = 0.03):
    """Global SSIM index (legacy single-value implementation)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    mu_x = X.mean()
    mu_y = Y.mean()

    sigma_xy = ((X.ravel() - mu_x) * (Y.ravel() - mu_y)).sum() / (X.size - 1)
    sigma_y = np.sqrt(((Y.ravel() - mu_y) ** 2).sum() / (Y.size - 1))
    sigma_x = np.sqrt(((X.ravel() - mu_x) ** 2).sum() / (X.size - 1))

    C1 = (k1 * 255) ** 2
    C2 = (k2 * 255) ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x * sigma_x + sigma_y * sigma_y + C2)
    )
    return ssim


__all__ = [
    "mmreadgray",
    "iabggmodel",
    "iaropen",
    "newline",
    "DrawSEAxis",
    "draw_se_axis",
    "magic",
    "mmlblshow",
    "mmshow",
    "SSIMIndex",
]
