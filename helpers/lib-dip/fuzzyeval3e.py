import numpy as np


def fuzzyeval3e(R, inmf, z, OP, ELSE=0):
    R = np.asarray(R)
    inmf = np.asarray(inmf, dtype=float)
    z = np.asarray(z, dtype=float).ravel()

    if R.ndim != 2:
        raise ValueError("R must be a 2-D array")
    if inmf.ndim != 2:
        raise ValueError("inmf must be a 2-D array")

    NR, NZ = R.shape
    NF, NP = inmf.shape

    if z.size != NZ:
        raise ValueError("The number of inputs must equal the number of cols in R")

    z_idx = np.floor((NP - 1) * z).astype(int) + 1
    z_idx = np.clip(z_idx, 1, NP)

    L = 1.0
    if OP == 2:
        L = 0.0

    Emu = np.zeros((NF, NZ), dtype=float)
    for r in range(NF):
        mf = inmf[r, :]
        for c in range(NZ):
            Emu[r, c] = mf[z_idx[c] - 1]

    Erule = np.zeros((NR, NZ), dtype=float)
    for i in range(NR):
        row = R[i, :]
        idxnot0 = np.where(row != 0)[0]
        Erule[i, :] = L
        for j in idxnot0:
            mf_idx = int(row[j])
            Erule[i, j] = Emu[mf_idx - 1, j]

    rule_strength = np.zeros(NR, dtype=float)
    if OP == 1:
        for i in range(NR):
            rule_strength[i] = np.min(Erule[i, :])
    else:
        for i in range(NR):
            rule_strength[i] = np.max(Erule[i, :])

    if ELSE:
        else_strength = np.min(1.0 - rule_strength)
        rule_strength = np.concatenate([rule_strength, np.array([else_strength])])

    return rule_strength, Emu


fuzzyeval = fuzzyeval3e
