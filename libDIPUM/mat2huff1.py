import numpy as np

from libDIPUM.huffman1 import huffman1


def mat2huff1(x):
    """
    Faster alternative to mat2huff using huffman1.
    Output structure format matches mat2huff.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.ndim != 2 or not np.isrealobj(x) or (not np.issubdtype(x.dtype, np.number) and x.dtype != np.bool_):
        raise ValueError("X must be a 2-D real numeric or logical matrix.")

    y = {}
    y['size'] = np.asarray(x.shape, dtype=np.uint32)

    x = np.round(x.astype(float))
    xmin = int(np.min(x))
    xmax = int(np.max(x))

    pmin = np.int16(xmin)
    y['min'] = np.uint16(int(pmin) + 32768)

    edges = np.arange(xmin, xmax + 2)
    h, _ = np.histogram(x.reshape(-1), bins=edges)
    if np.max(h) > 65535:
        h = 65535 * h / np.max(h)
    h = h.astype(np.uint16)
    y['hist'] = h

    code_map = huffman1(h.astype(float))
    idx = (x.reshape(-1, order='F').astype(int) - xmin)
    bits = ''.join(code_map[i] for i in idx)

    ysize = int(np.ceil(len(bits) / 16.0))
    bits16 = ('0' * (ysize * 16))
    bits16 = bits + bits16[len(bits):]

    words = []
    for i in range(0, len(bits16), 16):
        w = bits16[i:i + 16]
        words.append(int(w, 2))

    y['code'] = np.asarray(words, dtype=np.uint16)
    return y
