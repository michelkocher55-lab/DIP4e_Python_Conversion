import numpy as np


def fuzzyad3e(q):
    q = np.asarray(q, dtype=float)
    np_points = q.shape[1]
    v = np.linspace(0.0, 1.0, np_points, endpoint=True)
    Q = np.max(q, axis=0)
    den = np.sum(Q)
    defuzz = np.sum(v * Q) / (den + np.finfo(float).eps)
    return float(defuzz)


fuzzyad = fuzzyad3e
