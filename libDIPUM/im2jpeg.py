import numpy as np
from libDIP.tmat4e import tmat4e
from libDIPUM.mat2huff import mat2huff


def im2jpeg(x, quality=1, bits=8):
    """
    Compress an image using a JPEG approximation (MATLAB-style).
    """
    if x.ndim != 2 or not np.isrealobj(x):
        raise ValueError("The input image must be a real 2-D array.")
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError("The input image must be unsigned integer.")
    if bits < 1 or bits > 16:
        raise ValueError("The input image must have 1 to 16 bits/pixel.")
    if quality <= 0:
        raise ValueError("Input parameter QUALITY must be greater than zero.")

    m = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=float) * quality

    order = np.array([
        1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33,
        41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50,
        43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52,
        45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55,
        62, 63, 56, 64
    ]) - 1

    xm, xn = x.shape
    x = x.astype(float) - 2 ** (round(bits) - 1)

    T = tmat4e('DCT', 8)

    # Pad to multiples of 8 (blockproc PadPartialBlocks = true)
    pad_h = (8 - xm % 8) % 8
    pad_w = (8 - xn % 8) % 8
    if pad_h or pad_w:
        x = np.pad(x, ((0, pad_h), (0, pad_w)), mode='constant')
    xm_pad, xn_pad = x.shape

    n_blocks_row = xm_pad // 8
    n_blocks_col = xn_pad // 8
    xb = n_blocks_row * n_blocks_col

    # DCT and quantize block-by-block
    blocks = np.zeros((64, xb))
    idx = 0
    for br in range(n_blocks_row):
        for bc in range(n_blocks_col):
            block = x[br*8:(br+1)*8, bc*8:(bc+1)*8]
            dctb = T @ block @ T.T
            q = np.round(dctb / m)
            blocks[:, idx] = q.reshape(64, order='F')
            idx += 1

    # Zig-zag reorder
    y = blocks[order, :]

    # Run-length encode with EOB
    eob = np.max(y) + 1
    r = np.zeros(y.size + y.shape[1], dtype=float)
    count = 0
    for j in range(xb):
        col = y[:, j]
        nz = np.nonzero(col)[0]
        i = nz[-1] + 1 if nz.size else 0
        p = count
        q = p + i
        r[p:q] = col[:i]
        r[q] = eob
        count = q + 1
    r = r[:count]

    out = {
        'size': np.array([xm, xn], dtype=np.uint16),
        'bits': np.uint16(bits),
        'numblocks': np.uint16(xb),
        'quality': np.uint16(quality * 100),
        'huffman': mat2huff(r)
    }

    return out
