"""2-D adaptive noise-removal filtering (MATLAB wiener2 equivalent)."""

from __future__ import annotations
from typing import Any

import numpy as np
from scipy.signal import convolve2d


def _to_double_image(g: np.ndarray):
    """MATLAB-like im2double conversion for supported image classes."""
    dt = g.dtype

    if dt == np.float64:
        return g.astype(np.float64, copy=False), dt, False
    if dt == np.float32:
        return g.astype(np.float64), dt, True
    if dt == np.uint8:
        return g.astype(np.float64) / 255.0, dt, True
    if dt == np.uint16:
        return g.astype(np.float64) / 65535.0, dt, True
    if dt == np.int16:
        return g.astype(np.float64) / 32768.0, dt, True

    # Fallback for other numeric types.
    return g.astype(np.float64), dt, True


def _from_double_image(f: np.ndarray, classin: np.dtype):
    """Approximate MATLAB images.internal.changeClass behavior."""
    if classin == np.float64:
        return f
    if classin == np.float32:
        return f.astype(np.float32)
    if classin == np.uint8:
        return np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)
    if classin == np.uint16:
        return np.clip(np.rint(f * 65535.0), 0, 65535).astype(np.uint16)
    if classin == np.int16:
        out = np.rint(f * 32768.0)
        return np.clip(out, -32768, 32767).astype(np.int16)

    return f.astype(classin)


def _parse_inputs(*args: Any):
    """_parse_inputs."""
    nhood = np.array([3, 3], dtype=int)
    noise = None

    nargin = len(args)
    if nargin == 0:
        raise ValueError("wiener2: too few inputs")
    if nargin > 4:
        raise ValueError("wiener2: too many inputs")

    g = np.asarray(args[0])

    if nargin == 2:
        a2 = np.asarray(args[1])
        if a2.size == 1:
            noise = float(a2.reshape(-1)[0])
        elif a2.size == 2:
            nhood = np.asarray(a2, dtype=int).reshape(-1)
        else:
            raise ValueError("invalid syntax")

    elif nargin == 3:
        a3 = np.asarray(args[2])
        if a3.size == 2:
            raise ValueError("Removed syntax: WIENER2(I,[m n],[mblock nblock])")
        nhood = np.asarray(args[1], dtype=int).reshape(-1)
        noise = float(np.asarray(args[2]).reshape(-1)[0])

    elif nargin == 4:
        raise ValueError("Removed syntax: WIENER2(I,[m n],[mblock nblock],noise)")

    if g.ndim == 3:
        raise ValueError("wiener2 does not support 3-D truecolor input")
    if g.ndim != 2:
        raise ValueError("wiener2 expects a 2-D image")

    if nhood.size != 2:
        raise ValueError("Neighborhood must have 2 elements [m n]")
    if np.any(nhood <= 0):
        raise ValueError("Neighborhood sizes must be positive")

    return g, tuple(int(v) for v in nhood), noise


def wiener2(*args: Any):
    """Adaptive Wiener filtering on 2-D images.

    Supported call forms:
    - f, noise = wiener2(I)
    - f, noise = wiener2(I, noise)
    - f, noise = wiener2(I, [m, n])
    - f, noise = wiener2(I, [m, n], noise)
    """
    g_in, nhood, noise = _parse_inputs(*args)

    g, classin, class_changed = _to_double_image(g_in)

    kernel = np.ones(nhood, dtype=np.float64)
    denom = float(np.prod(nhood))

    # MATLAB filter2 defaults to zero-padding with same-size output.
    local_mean = (
        convolve2d(g, kernel, mode="same", boundary="fill", fillvalue=0.0) / denom
    )
    local_var = (
        convolve2d(g * g, kernel, mode="same", boundary="fill", fillvalue=0.0) / denom
        - local_mean * local_mean
    )

    if noise is None:
        noise = float(np.mean(local_var))

    # Split computation to minimize temporaries (MATLAB style).
    f = g - local_mean
    gtmp = local_var - noise
    gtmp = np.maximum(gtmp, 0.0)
    local_var = np.maximum(local_var, noise)
    f = f / local_var
    f = f * gtmp
    f = f + local_mean

    if class_changed:
        f = _from_double_image(f, classin)

    return f, float(noise)


__all__ = ["wiener2"]
