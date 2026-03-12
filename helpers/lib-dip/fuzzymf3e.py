import numpy as np


def fuzzymf3e(type_str, np_points, low, high, param, z0=None):
    param = np.asarray(param, dtype=float).ravel()
    if param.size < 2:
        raise ValueError("param must contain at least two values")

    a = param[0]
    b = param[1]

    c = None
    d = None
    if param.size >= 3:
        c = param[2]
    if param.size >= 4:
        d = param[3]

    z = np.linspace(low, high, int(np_points), endpoint=True)
    eps = np.finfo(float).eps
    u = np.zeros(int(np_points), dtype=float)

    if type_str == 'triang':
        idx1 = (z >= a - b) & (z < a)
        idx2 = (z >= a) & (z <= a + c)
        u[idx1] = 1.0 - (a - z[idx1]) / (b + eps)
        u[idx2] = 1.0 - (z[idx2] - a) / (c + eps)

    elif type_str == 'trapez':
        idx1 = (z >= a - c) & (z < a)
        idx2 = (z >= a) & (z < b)
        idx3 = (z >= b) & (z <= b + d)
        u[idx1] = 1.0 - (a - z[idx1]) / (c + eps)
        u[idx2] = 1.0
        u[idx3] = 1.0 - (z[idx3] - b) / (d + eps)

    elif type_str == 'sigma':
        idx1 = (z >= a - b) & (z <= a)
        idx2 = z > a
        u[idx1] = 1.0 - (a - z[idx1]) / (b + eps)
        u[idx2] = 1.0

    elif type_str == 's-shape':
        c = param[2]
        b_mid = (a + c) / 2.0
        idxa = z < a
        idxab = (z >= a) & (z <= b_mid)
        idxbc = (z > b_mid) & (z <= c)
        idxc = z > c
        u[idxa] = 0.0
        u[idxab] = 2.0 * ((z[idxab] - a) / (c - a + eps)) ** 2
        u[idxbc] = 1.0 - 2.0 * ((z[idxbc] - c) / (c - a + eps)) ** 2
        u[idxc] = 1.0

    elif type_str == 'bell':
        idxc = z <= c
        z1 = z[idxc]
        a1 = c - b
        c1 = c
        b1 = (a1 + c1) / 2.0
        u1 = np.zeros_like(z1, dtype=float)
        idxa1 = z1 < a1
        idxa1b1 = (z1 >= a1) & (z1 <= b1)
        idxb1c1 = (z1 > b1) & (z1 <= c1)
        idxc1 = z1 > c1
        u1[idxa1] = 0.0
        u1[idxa1b1] = 2.0 * ((z1[idxa1b1] - a1) / (c1 - a1 + eps)) ** 2
        u1[idxb1c1] = 1.0 - 2.0 * ((z1[idxb1c1] - c1) / (c1 - a1 + eps)) ** 2
        u1[idxc1] = 1.0

        idxc = z > c
        z2 = z[idxc]
        a2 = c
        c2 = c + b
        b2 = (a2 + c2) / 2.0
        u2 = np.zeros_like(z2, dtype=float)
        idxa2 = z2 < a2
        idxa2b2 = (z2 >= a2) & (z2 <= b2)
        idxb2c2 = (z2 > b2) & (z2 <= c2)
        idxc2 = z2 > c2
        u2[idxa2] = 0.0
        u2[idxa2b2] = 2.0 * ((z2[idxa2b2] - a2) / (c2 - a2 + eps)) ** 2
        u2[idxb2c2] = 1.0 - 2.0 * ((z2[idxb2c2] - c2) / (c2 - a2 + eps)) ** 2
        u2[idxc2] = 1.0
        u2 = 1.0 - u2

        u = np.concatenate([u1, u2])

    elif type_str == 'gaussian':
        idx = (z >= a - c) & (z <= a + c)
        u[idx] = np.exp(-((z[idx] - a) ** 2) / (2.0 * b ** 2 + eps))

    else:
        raise ValueError("Unknown type.")

    if z0 is None:
        uz0 = 0
    else:
        z0_arr = np.asarray(z0, dtype=float)
        if np.any((z0_arr < low) | (z0_arr > high)):
            raise ValueError("All values of z0 must be in the interval [low, high].")

        flat = z0_arr.ravel()
        idx = np.empty_like(flat, dtype=int)
        for i, val in enumerate(flat):
            d = np.abs(z - val)
            idx0 = np.where(d == np.min(d))[0]
            idx[i] = int(idx0[0])
        uz0 = u[idx].reshape(z0_arr.shape)

    return u, uz0


fuzzymf = fuzzymf3e
