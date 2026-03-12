from typing import Any
import numpy as np
from scipy.ndimage import map_coordinates

from helpers.libdipum.pointmin import pointmin


def _sample_vector_field(field: Any, p: Any):
    """_sample_vector_field."""
    # field shape: (*shape, ndim), p is 0-based continuous coordinate (ndim,)
    ndim = field.shape[-1]
    out = np.zeros(ndim, dtype=float)
    coords = [np.array([p[d]], dtype=float) for d in range(ndim)]
    for d in range(ndim):
        out[d] = map_coordinates(field[..., d], coords, order=1, mode="nearest")[0]
    return out


def _e1(p: Any, grad_vol: Any, step: Any):
    """_e1."""
    v = _sample_vector_field(grad_vol, p)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return p + step * v


def _rk4(p: Any, grad_vol: Any, step: Any):
    """_rk4."""

    def f(x: Any):
        """f."""
        v = _sample_vector_field(grad_vol, x)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v

    k1 = f(p)
    k2 = f(p + 0.5 * step * k1)
    k3 = f(p + 0.5 * step * k2)
    k4 = f(p + step * k3)
    return p + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _s1(p: Any, D: Any):
    """_s1."""
    # Move to lowest-distance direct neighbor.
    idx = tuple(np.clip(np.rint(p).astype(int), 0, np.array(D.shape) - 1))
    ndim = D.ndim
    best = idx
    bestv = D[idx]

    ranges = [(-1, 0, 1)] * ndim
    for off in np.array(np.meshgrid(*ranges, indexing="ij")).reshape(ndim, -1).T:
        if np.all(off == 0):
            continue
        q = tuple(np.array(idx) + off)
        if not all(0 <= q[d] < D.shape[d] for d in range(ndim)):
            continue
        if D[q] < bestv:
            bestv = D[q]
            best = q
    return np.asarray(best, dtype=float)


def shortestpath(
    DistanceMap: Any,
    StartPoint: Any,
    SourcePoint: Any = None,
    Stepsize: Any = 0.5,
    Method: Any = "rk4",
):
    """
    Python transcription of MATLAB shortestpath.m.

    Parameters use MATLAB-style points (1-based column vectors).
    Output ShortestLine is Nx2 or Nx3, also in 1-based coordinates.
    """
    D = np.asarray(DistanceMap, dtype=float)
    ndim = D.ndim
    if ndim not in (2, 3):
        raise ValueError("DistanceMap must be 2D or 3D.")

    sp = np.asarray(StartPoint, dtype=float).reshape(-1)
    if sp.size != ndim:
        raise ValueError("StartPoint has wrong dimension.")
    p = sp - 1.0  # convert 1-based to 0-based

    src = None
    if SourcePoint is not None and np.size(SourcePoint) > 0:
        src = np.asarray(SourcePoint, dtype=float)
        if src.ndim == 1:
            src = src.reshape(ndim, 1)
        if src.shape[0] != ndim and src.shape[1] == ndim:
            src = src.T
        if src.shape[0] != ndim:
            raise ValueError("SourcePoint must have shape (ndim, N).")
        src = src - 1.0

    if ndim == 2:
        fy, fx = pointmin(D)
        grad_vol = np.stack([-fx, -fy], axis=-1)
    else:
        fy, fx, fz = pointmin(D)
        grad_vol = np.stack([-fx, -fy, -fz], axis=-1)

    line = []
    i = 0
    dist_to_end = np.inf

    while True:
        method = Method.lower()
        if method == "rk4":
            q = _rk4(p, grad_vol, Stepsize)
        elif method == "euler":
            q = _e1(p, grad_vol, Stepsize)
        elif method == "simple":
            q = _s1(p, D)
        else:
            raise ValueError("unknown method")

        # out-of-bound stop
        if np.any(q < 0) or np.any(q >= np.array(D.shape) - 1e-9):
            break

        if src is not None:
            dif = src.T - q.reshape(1, -1)
            d = np.sqrt(np.sum(dif * dif, axis=1))
            ind = int(np.argmin(d))
            dist_to_end = float(d[ind])
        else:
            ind = -1

        if i > 10:
            movement = float(np.linalg.norm(q - line[i - 10]))
        else:
            movement = Stepsize + 1.0

        if movement < Stepsize:
            break

        line.append(q.copy())
        i += 1

        if src is not None and dist_to_end < Stepsize:
            line.append(src[:, ind].copy())
            break

        p = q

    if not line:
        return np.zeros((0, ndim), dtype=float)

    out = np.vstack(line) + 1.0  # back to MATLAB 1-based coordinates
    return out
