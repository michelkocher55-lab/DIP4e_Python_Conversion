from typing import Any
import numpy as np
from helpers.libdip.tmat4e import tmat4e
from helpers.libdipum.huff2mat import huff2mat


def jpeg2im(y: Any):
    """
    Decode an IM2JPEG compressed image (MATLAB-style).
    """
    m = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=float,
    )

    order = (
        np.array(
            [
                1,
                9,
                2,
                3,
                10,
                17,
                25,
                18,
                11,
                4,
                5,
                12,
                19,
                26,
                33,
                41,
                34,
                27,
                20,
                13,
                6,
                7,
                14,
                21,
                28,
                35,
                42,
                49,
                57,
                50,
                43,
                36,
                29,
                22,
                15,
                8,
                16,
                23,
                30,
                37,
                44,
                51,
                58,
                59,
                52,
                45,
                38,
                31,
                24,
                32,
                39,
                46,
                53,
                60,
                61,
                54,
                47,
                40,
                48,
                55,
                62,
                63,
                56,
                64,
            ]
        )
        - 1
    )
    rev = np.argsort(order)

    m = (float(y["quality"]) / 100.0) * m
    xb = int(y["numblocks"])
    xm, xn = map(int, y["size"])

    x = huff2mat(y["huffman"])
    x = np.asarray(x).ravel()
    eob = np.max(x)

    z = np.zeros((64, xb))
    k = 0
    for j in range(xb):
        for i in range(64):
            if x[k] == eob:
                k += 1
                break
            z[i, j] = x[k]
            k += 1

    z = z[rev, :]

    # Determine padded size
    pad_h = (8 - xm % 8) % 8
    pad_w = (8 - xn % 8) % 8
    xm_pad = xm + pad_h
    xn_pad = xn + pad_w
    n_blocks_row = xm_pad // 8
    n_blocks_col = xn_pad // 8

    # Rebuild blocks and inverse DCT
    T = tmat4e("DCT", 8)
    blocks = np.zeros((xm_pad, xn_pad))
    idx = 0
    for br in range(n_blocks_row):
        for bc in range(n_blocks_col):
            block = z[:, idx].reshape((8, 8), order="F")
            block = block * m
            rec = T.T @ block @ T
            blocks[br * 8 : (br + 1) * 8, bc * 8 : (bc + 1) * 8] = rec
            idx += 1

    # Level shift and crop
    blocks = blocks + 2 ** (int(y["bits"]) - 1)
    blocks = blocks[:xm, :xn]

    if int(y["bits"]) <= 8:
        return blocks.astype(np.uint8)
    return blocks.astype(np.uint16)
