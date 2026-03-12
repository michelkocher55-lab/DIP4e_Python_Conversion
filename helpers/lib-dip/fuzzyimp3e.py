import numpy as np


def fuzzyimp3e(rule_strength, outmf):
    rs = np.asarray(rule_strength, dtype=float).ravel()
    outmf = np.asarray(outmf, dtype=float)

    ns = rs.size
    q = np.zeros_like(outmf, dtype=float)

    for i in range(ns):
        clip_level = rs[i]
        h = outmf[i, :].copy()
        idx = h > clip_level
        h[idx] = clip_level
        q[i, :] = h

    return q


fuzzyimp = fuzzyimp3e
