from typing import Any
import numpy as np
from scipy.ndimage import binary_dilation

from helpers.libdipum.msfm import msfm
from helpers.libdipum.shortestpath import shortestpath


def skeleton(I: Any, verbose: Any = True):
    """
    Python transcription of MATLAB skeleton.m.

    Parameters
    ----------
    I : ndarray
        2D or 3D binary image/volume.
    verbose : bool
        Print progress messages.

    Returns
    -------
    S : list of ndarray
        Skeleton branches; each branch is Nx2 (2D) or Nx3 (3D), in MATLAB-style
        1-based coordinates.
    """
    I = np.asarray(I).astype(bool)
    if I.ndim not in (2, 3):
        raise ValueError("I must be 2D or 3D binary array.")

    IS3D = I.ndim == 3

    BoundaryDistance = getBoundaryDistance(I, IS3D)
    if verbose:
        print("Distance Map Constructed")

    SourcePoint, maxD = maxDistancePoint(BoundaryDistance, I, IS3D)

    SpeedImage = (BoundaryDistance / maxD) ** 4
    SpeedImage[SpeedImage == 0] = 1e-10

    SkeletonSegments = [None] * 1000
    itt = 0

    while True:
        if verbose:
            print(f"Find Branches Iterations : {itt}")

        T, Y = msfm(SpeedImage, SourcePoint, False, False)

        StartPoint, _ = maxDistancePoint(Y, I, IS3D)

        ShortestLine = shortestpath(T, StartPoint, SourcePoint, 1, "simple")
        linelength = GetLineLength(ShortestLine, IS3D)

        if linelength < maxD * 1.2:
            break

        itt += 1
        SkeletonSegments[itt - 1] = ShortestLine

        # Add branch points to source points for next iteration.
        SourcePoint = np.concatenate((SourcePoint, ShortestLine.T), axis=1)

    SkeletonSegments = SkeletonSegments[:itt]
    S = OrganizeSkeleton(SkeletonSegments, IS3D)

    if verbose:
        print(f"Skeleton Branches Found : {len(S)}")

    return S


def GetLineLength(L: Any, IS3D: Any):
    """GetLineLength."""
    L = np.asarray(L, dtype=float)
    if L.shape[0] < 2:
        return 0.0
    d = np.diff(L, axis=0)
    if IS3D:
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)
    else:
        dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    return float(np.sum(dist))


def OrganizeSkeleton(SkeletonSegments: Any, IS3D: Any):
    """OrganizeSkeleton."""
    n = len(SkeletonSegments)
    if n == 0:
        return []

    dim = 3 if IS3D else 2
    Endpoints = np.zeros((n * 2, dim), dtype=float)
    lmax = 1

    for w, ss in enumerate(SkeletonSegments):
        ss = np.asarray(ss, dtype=float)
        lmax = max(lmax, len(ss))
        Endpoints[w * 2, :] = ss[0, :]
        Endpoints[w * 2 + 1, :] = ss[-1, :]

    CutSkel = np.zeros((n, lmax), dtype=bool)
    ConnectDistance = 2**2

    for w, ss in enumerate(SkeletonSegments):
        ss = np.asarray(ss, dtype=float)

        ex = Endpoints[:, 0:1]
        sx = ss[:, 0][None, :]
        ey = Endpoints[:, 1:2]
        sy = ss[:, 1][None, :]

        if IS3D:
            ez = Endpoints[:, 2:3]
            sz = ss[:, 2][None, :]
            D = (ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2
        else:
            D = (ex - sx) ** 2 + (ey - sy) ** 2

        check = np.min(D, axis=1) < ConnectDistance
        check[w * 2] = False
        check[w * 2 + 1] = False

        idx = np.where(check)[0]
        for j in idx:
            line = D[j, :]
            k = int(np.argmin(line))
            if 2 < k < (len(line) - 2):
                CutSkel[w, k] = True

    S = []
    for w, ss in enumerate(SkeletonSegments):
        ss = np.asarray(ss, dtype=float)
        cuts = np.where(CutSkel[w, : len(ss)])[0].tolist()
        r = [0] + cuts + [len(ss) - 1]

        for i in range(len(r) - 1):
            a, b = r[i], r[i + 1]
            if b >= a:
                S.append(ss[a : b + 1, :])

    return S


def getBoundaryDistance(I: Any, IS3D: Any):
    """getBoundaryDistance."""
    # Boundary pixels: xor(I, imdilate(I,ones(...)))
    S = np.ones((3, 3, 3), dtype=bool) if IS3D else np.ones((3, 3), dtype=bool)
    B = np.logical_xor(I, binary_dilation(I, structure=S))

    ind = np.flatnonzero(B)
    if ind.size == 0:
        return np.zeros_like(I, dtype=float)

    coords = np.array(np.unravel_index(ind, B.shape)).astype(float) + 1.0

    SpeedImage = np.ones(I.shape, dtype=float)
    BoundaryDistance, _ = msfm(SpeedImage, coords, False, True)

    BoundaryDistance = np.asarray(BoundaryDistance, dtype=float)
    BoundaryDistance[~I] = 0.0
    return BoundaryDistance


def maxDistancePoint(BoundaryDistance: Any, I: Any, IS3D: Any):
    """maxDistancePoint."""
    D = np.array(BoundaryDistance, dtype=float, copy=True)
    D[~I] = 0.0

    ind = int(np.argmax(D))
    maxD = float(np.max(D))

    if not np.isfinite(maxD):
        raise ValueError("Skeleton:Maximum: Maximum from MSFM is infinite!")

    if IS3D:
        x, y, z = np.unravel_index(ind, I.shape)
        posD = np.array([[x + 1.0], [y + 1.0], [z + 1.0]])
    else:
        x, y = np.unravel_index(ind, I.shape)
        posD = np.array([[x + 1.0], [y + 1.0]])

    return posD, maxD
