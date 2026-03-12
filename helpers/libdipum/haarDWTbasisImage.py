from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from helpers.libdipum.wavedec2 import wavedec2
from helpers.libdipum.waverec2 import waverec2


def _mat2gray(a: Any):
    """_mat2gray."""
    a = np.asarray(a, dtype=float)
    amin = np.min(a)
    amax = np.max(a)
    if amax == amin:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)


def haarDWTbasisImage(scale: Any):
    """haarDWTbasisImage."""
    N = 8
    P = 2
    fill = 1
    line = 0

    f = np.zeros((8, 8))
    c, s = wavedec2(f, 3, "haar")
    c = c.reshape(-1)

    if scale == 1:
        coef = np.array(
            [
                [1, 5, 9, 13, 17, 21, 25, 29],
                [2, 6, 10, 14, 18, 22, 26, 30],
                [3, 7, 11, 15, 19, 23, 27, 31],
                [4, 8, 12, 16, 20, 24, 28, 32],
                [33, 37, 41, 45, 49, 53, 57, 61],
                [34, 38, 42, 46, 50, 54, 58, 62],
                [35, 39, 43, 47, 51, 55, 59, 63],
                [36, 40, 44, 48, 52, 56, 60, 64],
            ]
        )
    elif scale == 2:
        coef = np.array(
            [
                [1, 3, 5, 7, 17, 21, 25, 29],
                [2, 4, 6, 8, 18, 22, 26, 30],
                [9, 11, 13, 15, 19, 23, 27, 31],
                [10, 12, 14, 16, 20, 24, 28, 32],
                [33, 37, 41, 45, 49, 53, 57, 61],
                [34, 38, 42, 46, 50, 54, 58, 62],
                [35, 39, 43, 47, 51, 55, 59, 63],
                [36, 40, 44, 48, 52, 56, 60, 64],
            ]
        )
    else:
        coef = np.array(
            [
                [1, 2, 5, 7, 17, 21, 25, 29],
                [3, 4, 6, 8, 18, 22, 26, 30],
                [9, 11, 13, 15, 19, 23, 27, 31],
                [10, 12, 14, 16, 20, 24, 28, 32],
                [33, 37, 41, 45, 49, 53, 57, 61],
                [34, 38, 42, 46, 50, 54, 58, 62],
                [35, 39, 43, 47, 51, 55, 59, 63],
                [36, 40, 44, 48, 52, 56, 60, 64],
            ]
        )

    B = np.zeros((64, 64))
    for i in range(8):
        for j in range(8):
            cc = np.zeros_like(c)
            cc[coef[i, j] - 1] = 1
            b = waverec2(cc, s, "haar")
            sI = 8 * i
            sJ = 8 * j
            B[sI : sI + 8, sJ : sJ + 8] = b

    SC = _mat2gray(B)

    w = N * N + 2 * N
    S_LINE = np.full((w, w), line, dtype=float)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            x = N * i - N
            y = N * j - N
            xd = (N + 2) * i - N
            yd = (N + 2) * j - N
            S_LINE[xd : xd + N, yd : yd + N] = SC[x : x + N, y : y + N]

    w = N * (N + 2) + P * (N - 1)
    haar_img = np.full((w, w), fill, dtype=float)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            x = (N + 2) * i - N - 1
            y = (N + 2) * j - N - 1
            xd = (N + P + 2) * i - N - 1 - P
            yd = (N + P + 2) * j - N - 1 - P
            haar_img[xd : xd + N + 2, yd : yd + N + 2] = S_LINE[
                x : x + N + 2, y : y + N + 2
            ]

    plt.imshow(haar_img, cmap="gray", vmin=0, vmax=1)
    return haar_img
