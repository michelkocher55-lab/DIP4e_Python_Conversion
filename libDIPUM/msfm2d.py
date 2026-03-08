import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt


def _as_source_points_2d(source_points):
    sp = np.asarray(source_points, dtype=float)
    if sp.ndim == 1:
        sp = sp.reshape(2, 1)
    if sp.shape[0] != 2 and sp.shape[1] == 2:
        sp = sp.T
    if sp.shape[0] != 2:
        raise ValueError('SourcePoints must be shape (2, N) or (N, 2).')
    # MATLAB code uses int32(floor(SourcePoints)) in 1-based coordinates.
    return np.floor(sp).astype(int)


def _roots_quadratic(coeff):
    a, b, c = float(coeff[0]), float(coeff[1]), float(coeff[2])
    d = max(b * b - 4.0 * a * c, 0.0)
    sd = np.sqrt(d)
    if a != 0:
        z1 = (-b - sd) / (2.0 * a)
        z2 = (-b + sd) / (2.0 * a)
    else:
        # same algebraic form used in the provided MATLAB helper.
        den1 = (-b - sd)
        den2 = (-b + sd)
        z1 = (2.0 * c) / den1 if den1 != 0 else np.inf
        z2 = (2.0 * c) / den2 if den2 != 0 else np.inf
    return z1, z2


def _calculate_distance(T, Fij, i, j, usesecond, usecross, frozen, eps=1e-12):
    nrows, ncols = T.shape

    # Build local 5x5 patch with frozen-only values.
    Tpatch = np.full((5, 5), np.inf, dtype=float)
    for nx in range(-2, 3):
        for ny in range(-2, 3):
            ii = i + nx
            jj = j + ny
            if 0 <= ii < nrows and 0 <= jj < ncols and frozen[ii, jj]:
                Tpatch[nx + 2, ny + 2] = T[ii, jj]

    Tm = np.full(4, np.inf, dtype=float)
    Tm2 = np.zeros(4, dtype=float)
    order = np.zeros(4, dtype=int)

    # First-order in x,y.
    Tm[0] = min(Tpatch[1, 2], Tpatch[3, 2])
    if np.isfinite(Tm[0]):
        order[0] = 1

    Tm[1] = min(Tpatch[2, 1], Tpatch[2, 3])
    if np.isfinite(Tm[1]):
        order[1] = 1

    # First-order cross.
    if usecross:
        Tm[2] = min(Tpatch[1, 1], Tpatch[3, 3])
        if np.isfinite(Tm[2]):
            order[2] = 1

        Tm[3] = min(Tpatch[1, 3], Tpatch[3, 1])
        if np.isfinite(Tm[3]):
            order[3] = 1

    # Second-order derivatives.
    if usesecond:
        ch1 = (Tpatch[0, 2] < Tpatch[1, 2]) and np.isfinite(Tpatch[1, 2])
        ch2 = (Tpatch[4, 2] < Tpatch[3, 2]) and np.isfinite(Tpatch[3, 2])
        if ch1 and ch2:
            Tm2[0] = min((4.0 * Tpatch[1, 2] - Tpatch[0, 2]) / 3.0,
                         (4.0 * Tpatch[3, 2] - Tpatch[4, 2]) / 3.0)
            order[0] = 2
        elif ch1:
            Tm2[0] = (4.0 * Tpatch[1, 2] - Tpatch[0, 2]) / 3.0
            order[0] = 2
        elif ch2:
            Tm2[0] = (4.0 * Tpatch[3, 2] - Tpatch[4, 2]) / 3.0
            order[0] = 2

        ch1 = (Tpatch[2, 0] < Tpatch[2, 1]) and np.isfinite(Tpatch[2, 1])
        ch2 = (Tpatch[2, 4] < Tpatch[2, 3]) and np.isfinite(Tpatch[2, 3])
        if ch1 and ch2:
            Tm2[1] = min((4.0 * Tpatch[2, 1] - Tpatch[2, 0]) / 3.0,
                         (4.0 * Tpatch[2, 3] - Tpatch[2, 4]) / 3.0)
            order[1] = 2
        elif ch1:
            Tm2[1] = (4.0 * Tpatch[2, 1] - Tpatch[2, 0]) / 3.0
            order[1] = 2
        elif ch2:
            Tm2[1] = (4.0 * Tpatch[2, 3] - Tpatch[2, 4]) / 3.0
            order[1] = 2

        if usecross:
            ch1 = (Tpatch[0, 0] < Tpatch[1, 1]) and np.isfinite(Tpatch[1, 1])
            ch2 = (Tpatch[4, 4] < Tpatch[3, 3]) and np.isfinite(Tpatch[3, 3])
            if ch1 and ch2:
                Tm2[2] = min((4.0 * Tpatch[1, 1] - Tpatch[0, 0]) / 3.0,
                             (4.0 * Tpatch[3, 3] - Tpatch[4, 4]) / 3.0)
                order[2] = 2
            elif ch1:
                Tm2[2] = (4.0 * Tpatch[1, 1] - Tpatch[0, 0]) / 3.0
                order[2] = 2
            elif ch2:
                Tm2[2] = (4.0 * Tpatch[3, 3] - Tpatch[4, 4]) / 3.0
                order[2] = 2

            ch1 = (Tpatch[0, 4] < Tpatch[1, 3]) and np.isfinite(Tpatch[1, 3])
            ch2 = (Tpatch[4, 0] < Tpatch[3, 1]) and np.isfinite(Tpatch[3, 1])
            if ch1 and ch2:
                Tm2[3] = min((4.0 * Tpatch[1, 3] - Tpatch[0, 4]) / 3.0,
                             (4.0 * Tpatch[3, 1] - Tpatch[4, 0]) / 3.0)
                order[3] = 2
            elif ch1:
                Tm2[3] = (4.0 * Tpatch[1, 3] - Tpatch[0, 4]) / 3.0
                order[3] = 2
            elif ch2:
                Tm2[3] = (4.0 * Tpatch[3, 1] - Tpatch[4, 0]) / 3.0
                order[3] = 2

    coeff = np.array([0.0, 0.0, -1.0 / max(Fij * Fij, eps)], dtype=float)

    for t in range(2):
        if order[t] == 1:
            coeff += np.array([1.0, -2.0 * Tm[t], Tm[t] * Tm[t]])
        elif order[t] == 2:
            coeff += np.array([1.0, -2.0 * Tm2[t], Tm2[t] * Tm2[t]]) * 2.25

    z1, z2 = _roots_quadratic(coeff)
    Tt = max(z1, z2)

    if usecross:
        coeff2 = coeff + np.array([0.0, 0.0, -1.0 / max(Fij * Fij, eps)], dtype=float)
        for t in range(2, 4):
            if order[t] == 1:
                coeff2 += 0.5 * np.array([1.0, -2.0 * Tm[t], Tm[t] * Tm[t]])
            elif order[t] == 2:
                coeff2 += 0.5 * np.array([1.0, -2.0 * Tm2[t], Tm2[t] * Tm2[t]]) * 2.25
        z1, z2 = _roots_quadratic(coeff2)
        Tt2 = max(z1, z2)
        if np.isfinite(Tt2):
            Tt = min(Tt, Tt2)

    direct = Tm[np.isfinite(Tm)]
    if direct.size > 0 and np.any(direct >= Tt):
        Tt = np.min(direct) + (1.0 / max(Fij, eps))

    return float(Tt)


def msfm2d(F, SourcePoints, usesecond=False, usecross=False):
    """Python transcription of msfm2d.m (2D only)."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError('Speed image must be 2D.')
    if np.any(F <= 0):
        raise ValueError('Speed image values must be > 0.')

    sources = _as_source_points_2d(SourcePoints)

    nrows, ncols = F.shape
    T = -np.ones_like(F, dtype=float)
    frozen = np.zeros_like(F, dtype=bool)

    # 4-neighborhood.
    ne = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Freeze sources.
    for z in range(sources.shape[1]):
        x = int(sources[0, z]) - 1
        y = int(sources[1, z]) - 1
        if 0 <= x < nrows and 0 <= y < ncols:
            frozen[x, y] = True
            T[x, y] = 0.0

    # Narrow band as heap with lazy updates.
    heap = []

    # Initialize with neighbors of sources.
    for z in range(sources.shape[1]):
        x = int(sources[0, z]) - 1
        y = int(sources[1, z]) - 1
        if not (0 <= x < nrows and 0 <= y < ncols):
            continue
        for dx, dy in ne:
            i = x + dx
            j = y + dy
            if 0 <= i < nrows and 0 <= j < ncols and not frozen[i, j]:
                Tt = 1.0 / max(F[i, j], np.finfo(float).eps)
                if T[i, j] > 0:
                    if Tt < T[i, j]:
                        T[i, j] = Tt
                        heapq.heappush(heap, (Tt, i, j))
                else:
                    T[i, j] = Tt
                    heapq.heappush(heap, (Tt, i, j))

    # Marching.
    for _ in range(F.size):
        if not heap:
            break

        t, x, y = heapq.heappop(heap)

        if frozen[x, y]:
            continue
        # stale heap entry
        if abs(t - T[x, y]) > 1e-12:
            continue

        frozen[x, y] = True

        for dx, dy in ne:
            i = x + dx
            j = y + dy
            if 0 <= i < nrows and 0 <= j < ncols and not frozen[i, j]:
                Tt = _calculate_distance(T, F[i, j], i, j, usesecond, usecross, frozen)

                if T[i, j] > 0:
                    if Tt < T[i, j]:
                        T[i, j] = Tt
                        heapq.heappush(heap, (Tt, i, j))
                else:
                    T[i, j] = Tt
                    heapq.heappush(heap, (Tt, i, j))

    # Augmented distance map used by skeletonization:
    # Euclidean distance to nearest source point.
    src_mask = np.ones((nrows, ncols), dtype=bool)
    for z in range(sources.shape[1]):
        x = int(sources[0, z]) - 1
        y = int(sources[1, z]) - 1
        if 0 <= x < nrows and 0 <= y < ncols:
            src_mask[x, y] = False
    Y = distance_transform_edt(src_mask)
    # Keep only points actually reached by fast marching so the caller
    # does not pick an unreachable start point.
    Y[T < 0] = 0.0

    return T, Y
