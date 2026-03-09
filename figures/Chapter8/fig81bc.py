from typing import Any
import numpy as np


def fig81bc(part: Any, s: Any = 256, n: Any = None, p: Any = None):
    """
    Figure 8.1(b,c) generator.
    (b) s x s image with s gray-level lines (uniform distribution)
    (c) s x s image with random gray levels n at p points on medium gray field
    """
    if n is None:
        n = np.array([125, 126, 127, 129, 130, 131], dtype=np.uint8)
        p = np.array([1935, 5123, 9997, 7652, 4755, 1877], dtype=int)
        jend = 6
    else:
        n = np.asarray(n, dtype=np.uint8)
        p = np.asarray(p, dtype=int)
        if n.shape != p.shape:
            raise ValueError("n and p must be the same size!")
        jend = n.shape[0] if n.ndim == 1 else n.shape[1]

    img = np.zeros((s, s), dtype=np.uint8)

    if part == "b":
        gl = np.arange(s, dtype=np.uint8)
        for k in range(s):
            r = int(np.ceil(len(gl) * np.random.rand())) - 1
            img[k, :] = gl[r]
            gl = np.delete(gl, r)
    else:
        img[:, :] = 128
        for j in range(jend):
            for _ in range(int(p[j])):
                img[
                    int(np.ceil(s * np.random.rand())) - 1,
                    int(np.ceil(s * np.random.rand())) - 1,
                ] = n[j]

        for k in range(50, 81):
            img[:, k] = 128

        for j in range(50, 81):
            img[j, :] = 128

    return img
