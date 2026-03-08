import numpy as np


def wavework(opcode, kind, c, s, level=None, x=None):
    """
    Edit/access wavelet decomposition structures.
    MATLAB-faithful translation of DIPUM wavework.m.

    Returns:
      opcode 'copy'  -> y
      opcode 'cut'   -> (nc, y)
      opcode 'paste' -> nc
    """
    c = np.asarray(c)
    # MATLAB expects a row vector for C.
    if c.ndim == 2 and c.shape[0] == 1:
        c = c.reshape(-1)
    elif c.ndim != 1:
        raise ValueError("C must be a row vector.")

    s = np.asarray(s)
    if s.ndim != 2 or not np.isrealobj(s) or not np.issubdtype(s.dtype, np.number) or s.shape[1] not in (2, 3):
        raise ValueError("S must be a real, numeric two- or three-column array.")

    s_int = s.astype(int)
    elements = np.prod(s_int, axis=1)
    if (c.size < elements[-1]) or not (elements[0] + 3 * np.sum(elements[1:-1]) >= elements[-1]):
        raise ValueError("[C S] must form a standard wavelet decomposition structure.")

    opcode_l = str(opcode).lower()
    if opcode_l.startswith('pas') and x is None:
        raise ValueError("Not enough input arguments.")

    if level is None:
        level = 1
    n = int(level)
    nmax = s_int.shape[0] - 2

    kind_l = str(kind).lower()
    if len(kind_l) == 0:
        raise ValueError("TYPE must begin with 'a', 'h', 'v', or 'd'.")

    aflag = (kind_l[0] == 'a')
    if (not aflag) and (n > nmax):
        raise ValueError("N exceeds the decompositions in [C, S].")

    # Make pointers into C (convert MATLAB 1-based to Python 0-based at end).
    if kind_l[0] == 'a':
        nindex = 1
        start = 1
        stop = int(elements[0])
        ntst = nmax
    elif kind_l[0] in ('h', 'v', 'd'):
        if kind_l[0] == 'h':
            offset = 0
        elif kind_l[0] == 'v':
            offset = 1
        else:
            offset = 2

        nindex = s_int.shape[0] - n
        start = int(elements[0] + 3 * np.sum(elements[1:nmax - n + 1]) + offset * elements[nindex - 1] + 1)
        stop = int(start + elements[nindex - 1] - 1)
        ntst = n
    else:
        raise ValueError("TYPE must begin with 'a', 'h', 'v', or 'd'.")

    # Convert to Python slice bounds.
    start0 = start - 1
    stop0 = stop  # inclusive MATLAB stop -> exclusive Python stop

    # Do requested action.
    if opcode_l in ('copy', 'cut'):
        y = c[start0:stop0]
        # MATLAB reshape is column-major.
        y = y.reshape(tuple(s_int[nindex - 1, :]), order='F')
        nc = c.copy()
        if opcode_l.startswith('cut'):
            nc[start0:stop0] = 0
            return nc, y
        return y

    if opcode_l == 'paste':
        x_arr = np.asarray(x)
        if x_arr.size != int(elements[-1 - ntst]):
            raise ValueError("X is not sized for the requested paste.")
        nc = c.copy()
        nc[start0:stop0] = x_arr.reshape(-1, order='F')
        return nc

    raise ValueError("Unrecognized OPCODE.")
