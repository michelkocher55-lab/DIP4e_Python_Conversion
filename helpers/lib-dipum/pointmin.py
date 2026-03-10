from typing import Any
import numpy as np


def pointmin(I: Any):
    """
    Python transcription of MATLAB pointmin.m.

    Returns
    -------
    2D: Fy, Fx
    3D: Fy, Fx, Fz
    """
    I = np.asarray(I)

    Fx = np.zeros_like(I)
    Fy = np.zeros_like(I)
    Fz = np.zeros_like(I)

    J = np.zeros(tuple(np.array(I.shape) + 2), dtype=I.dtype)
    J[...] = np.max(I)

    if I.ndim == 2:
        Iwork = I.copy()
        J[1:-1, 1:-1] = Iwork
        Ne = np.array(
            [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ],
            dtype=int,
        )

        for k in range(Ne.shape[0]):
            dx, dy = Ne[k]
            In = J[1 + dx : J.shape[0] - 1 + dx, 1 + dy : J.shape[1] - 1 + dy]
            check = In < Iwork
            Iwork[check] = In[check]
            D = Ne[k].astype(float)
            D = D / np.sqrt(np.sum(D**2))
            Fx[check] = D[0]
            Fy[check] = D[1]

        return Fy, Fx

    if I.ndim == 3:
        Iwork = I.copy()
        J[1:-1, 1:-1, 1:-1] = Iwork
        Ne = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 0],
                [-1, -1, 1],
                [-1, 0, -1],
                [-1, 0, 0],
                [-1, 0, 1],
                [-1, 1, -1],
                [-1, 1, 0],
                [-1, 1, 1],
                [0, -1, -1],
                [0, -1, 0],
                [0, -1, 1],
                [0, 0, -1],
                [0, 0, 1],
                [0, 1, -1],
                [0, 1, 0],
                [0, 1, 1],
                [1, -1, -1],
                [1, -1, 0],
                [1, -1, 1],
                [1, 0, -1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, -1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=int,
        )

        for k in range(Ne.shape[0]):
            dx, dy, dz = Ne[k]
            In = J[
                1 + dx : J.shape[0] - 1 + dx,
                1 + dy : J.shape[1] - 1 + dy,
                1 + dz : J.shape[2] - 1 + dz,
            ]
            check = In < Iwork
            Iwork[check] = In[check]
            D = Ne[k].astype(float)
            D = D / np.sqrt(np.sum(D**2))
            Fx[check] = D[0]
            Fy[check] = D[1]
            Fz[check] = D[2]

        return Fy, Fx, Fz

    raise ValueError("I must be a 2D or 3D array.")
