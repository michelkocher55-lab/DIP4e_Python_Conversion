from typing import Any
import numpy as np


def _as_size(v: Any, default: Any):
    """_as_size."""
    if v is None:
        return np.array(default, dtype=int)
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size == 0:
        return np.array(default, dtype=int)
    if a.size == 1:
        n = int(a[0])
        if n <= 0:
            raise ValueError("HSIZE must be positive.")
        return np.array([n, n], dtype=int)
    if a.size == 2:
        r, c = int(a[0]), int(a[1])
        if r <= 0 or c <= 0:
            raise ValueError("HSIZE values must be positive.")
        return np.array([r, c], dtype=int)
    raise ValueError("HSIZE must be scalar or length-2.")


def _parse_inputs(filter_type: Any, *args: Any):
    """_parse_inputs."""
    t = str(filter_type).lower()
    valid = {
        "average",
        "disk",
        "gaussian",
        "laplacian",
        "log",
        "motion",
        "prewitt",
        "sobel",
        "unsharp",
    }
    if t not in valid:
        raise ValueError(f"Unknown filter TYPE: {filter_type}")

    if t == "average":
        if len(args) == 0:
            return t, _as_size(None, [3, 3]), None
        if len(args) == 1:
            return t, _as_size(args[0], [3, 3]), None
        raise ValueError("average takes at most one additional argument.")

    if t == "disk":
        if len(args) == 0:
            return t, 5.0, None
        if len(args) == 1:
            r = float(args[0])
            if r <= 0:
                raise ValueError("RADIUS must be positive.")
            return t, r, None
        raise ValueError("disk takes at most one additional argument.")

    if t == "gaussian":
        if len(args) == 0:
            return t, _as_size(None, [3, 3]), 0.5
        if len(args) == 1:
            return t, _as_size(args[0], [3, 3]), 0.5
        if len(args) == 2:
            sigma = float(args[1])
            if sigma <= 0:
                raise ValueError("SIGMA must be positive.")
            size_arg = args[0]
            if (
                isinstance(size_arg, (list, tuple, np.ndarray))
                and np.asarray(size_arg).size == 0
            ):
                n = int(2 * np.ceil(2 * sigma) + 1)
                siz = np.array([n, n], dtype=int)
            else:
                siz = _as_size(size_arg, [3, 3])
            return t, siz, sigma
        raise ValueError("gaussian takes at most two additional arguments.")

    if t in ("laplacian", "unsharp"):
        if len(args) == 0:
            return t, 0.2, None
        if len(args) == 1:
            alpha = float(args[0])
            if alpha < 0 or alpha > 1:
                raise ValueError("ALPHA must be in [0, 1].")
            return t, alpha, None
        raise ValueError(f"{t} takes at most one additional argument.")

    if t == "log":
        if len(args) == 0:
            return t, _as_size(None, [5, 5]), 0.5
        if len(args) == 1:
            return t, _as_size(args[0], [5, 5]), 0.5
        if len(args) == 2:
            sigma = float(args[1])
            if sigma <= 0:
                raise ValueError("SIGMA must be positive.")
            size_arg = args[0]
            if (
                isinstance(size_arg, (list, tuple, np.ndarray))
                and np.asarray(size_arg).size == 0
            ):
                n = int(2 * np.ceil(2 * sigma) + 1)
                siz = np.array([n, n], dtype=int)
            else:
                siz = _as_size(size_arg, [5, 5])
            return t, siz, sigma
        raise ValueError("log takes at most two additional arguments.")

    if t == "motion":
        if len(args) == 0:
            return t, 9.0, 0.0
        if len(args) == 1:
            l = float(args[0])
            if l <= 0:
                raise ValueError("LEN must be positive.")
            return t, l, 0.0
        if len(args) == 2:
            l = float(args[0])
            theta = float(args[1])
            if l <= 0:
                raise ValueError("LEN must be positive.")
            return t, l, theta
        raise ValueError("motion takes at most two additional arguments.")

    return t, None, None


def fspecial(filter_type: Any, *args: Any):
    """
    MATLAB-like FSPECIAL for predefined 2-D filters.

    Supported types:
    average, disk, gaussian, laplacian, log, motion, prewitt, sobel, unsharp.
    """
    t, p2, p3 = _parse_inputs(filter_type, *args)

    if t == "average":
        siz = p2
        return np.ones((siz[0], siz[1]), dtype=float) / float(np.prod(siz))

    if t == "disk":
        rad = float(p2)
        crad = int(np.ceil(rad - 0.5))
        x, y = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
        maxxy = np.maximum(np.abs(x), np.abs(y)).astype(float)
        minxy = np.minimum(np.abs(x), np.abs(y)).astype(float)

        a1 = (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2
        a2 = (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2

        m1 = np.where(
            rad**2 < a1,
            minxy - 0.5,
            np.sqrt(np.maximum(rad**2 - (maxxy + 0.5) ** 2, 0.0)),
        )
        m2 = np.where(
            rad**2 > a2,
            minxy + 0.5,
            np.sqrt(np.maximum(rad**2 - (maxxy - 0.5) ** 2, 0.0)),
        )

        m1r = np.clip(m1 / rad, -1.0, 1.0)
        m2r = np.clip(m2 / rad, -1.0, 1.0)

        term = (
            rad**2
            * (
                0.5 * (np.arcsin(m2r) - np.arcsin(m1r))
                + 0.25 * (np.sin(2 * np.arcsin(m2r)) - np.sin(2 * np.arcsin(m1r)))
            )
            - (maxxy - 0.5) * (m2 - m1)
            + (m1 - minxy + 0.5)
        )

        cond = (
            (rad**2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2)
            & (rad**2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)
        ) | ((minxy == 0) & (maxxy - 0.5 < rad) & (maxxy + 0.5 >= rad))

        sgrid = term * cond.astype(float)
        sgrid = sgrid + (((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2) < rad**2).astype(
            float
        )

        sgrid[crad, crad] = min(np.pi * rad**2, np.pi / 2)

        if (crad > 0) and (rad > crad - 0.5) and (rad**2 < (crad - 0.5) ** 2 + 0.25):
            m1s = np.sqrt(np.maximum(rad**2 - (crad - 0.5) ** 2, 0.0))
            m1n = np.clip(m1s / rad, -1.0, 1.0)
            sg0 = 2 * (
                rad**2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n)))
                - m1s * (crad - 0.5)
            )

            sgrid[2 * crad, crad] = sg0
            sgrid[crad, 2 * crad] = sg0
            sgrid[crad, 0] = sg0
            sgrid[0, crad] = sg0

            sgrid[2 * crad - 1, crad] -= sg0
            sgrid[crad, 2 * crad - 1] -= sg0
            sgrid[crad, 1] -= sg0
            sgrid[1, crad] -= sg0

        sgrid[crad, crad] = min(sgrid[crad, crad], 1.0)
        h = sgrid / np.sum(sgrid)
        return h

    if t == "gaussian":
        siz = (p2 - 1) / 2.0
        std = float(p3)
        x, y = np.meshgrid(
            np.arange(-int(siz[1]), int(siz[1]) + 1),
            np.arange(-int(siz[0]), int(siz[0]) + 1),
        )
        arg = -(x * x + y * y) / (2.0 * std * std)
        h = np.exp(arg)
        h[h < np.finfo(float).eps * np.max(h)] = 0.0
        s = np.sum(h)
        if s != 0:
            h = h / s
        return h

    if t == "laplacian":
        alpha = float(np.clip(p2, 0.0, 1.0))
        h1 = alpha / (alpha + 1.0)
        h2 = (1.0 - alpha) / (alpha + 1.0)
        return np.array(
            [[h1, h2, h1], [h2, -4.0 / (alpha + 1.0), h2], [h1, h2, h1]],
            dtype=float,
        )

    if t == "log":
        siz = (p2 - 1) / 2.0
        std2 = float(p3) ** 2
        x, y = np.meshgrid(
            np.arange(-int(siz[1]), int(siz[1]) + 1),
            np.arange(-int(siz[0]), int(siz[0]) + 1),
        )
        arg = -(x * x + y * y) / (2.0 * std2)
        h = np.exp(arg)
        h[h < np.finfo(float).eps * np.max(h)] = 0.0
        s = np.sum(h)
        if s != 0:
            h = h / s
        h1 = h * (x * x + y * y - 2.0 * std2) / (std2**2)
        h = h1 - np.sum(h1) / float(np.prod(p2))
        return h

    if t == "motion":
        length = max(1.0, float(p2))
        half = (length - 1.0) / 2.0
        phi = np.mod(float(p3), 180.0) / 180.0 * np.pi

        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xsign = np.sign(cosphi)
        if xsign == 0:
            xsign = 1.0
        linewdt = 1.0

        sx = int(np.fix(half * cosphi + linewdt * xsign - length * np.finfo(float).eps))
        sy = int(np.fix(half * sinphi + linewdt - length * np.finfo(float).eps))

        x_vals = np.arange(0, sx + int(xsign), int(xsign), dtype=float)
        y_vals = np.arange(0, sy + 1, dtype=float)
        x, y = np.meshgrid(x_vals, y_vals)

        dist2line = y * cosphi - x * sinphi
        rad = np.sqrt(x * x + y * y)

        lastpix = np.where((rad >= half) & (np.abs(dist2line) <= linewdt))
        if lastpix[0].size > 0 and np.abs(cosphi) > np.finfo(float).eps:
            x2lastpix = half - np.abs(
                (x[lastpix] + dist2line[lastpix] * sinphi) / cosphi
            )
            dist2line[lastpix] = np.sqrt(dist2line[lastpix] ** 2 + x2lastpix**2)

        dist2line = linewdt + np.finfo(float).eps - np.abs(dist2line)
        dist2line[dist2line < 0] = 0.0

        h = np.rot90(dist2line, 2)
        nr, nc = dist2line.shape
        hr, hc = h.shape
        out = np.zeros((hr + nr - 1, hc + nc - 1), dtype=float)
        out[:hr, :hc] = h
        out[hr - 1 :, hc - 1 :] = np.maximum(out[hr - 1 :, hc - 1 :], dist2line)

        h = out / (np.sum(out) + np.finfo(float).eps * length * length)
        if cosphi > 0:
            h = np.flipud(h)
        return h

    if t == "prewitt":
        return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=float)

    if t == "sobel":
        return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    if t == "unsharp":
        alpha = float(p2)
        return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float) - fspecial(
            "laplacian", alpha
        )

    raise ValueError("Unsupported filter type.")
