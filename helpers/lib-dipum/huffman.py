from typing import Any
import sys
import numpy as np


def huffman(p: Any, return_tree: Any = False):
    """
    Build a variable-length Huffman code for source probability vector p.
    MATLAB-faithful translation of DIPUM huffman.m.

    Parameters
    ----------
    p : array_like
        Real numeric vector of probabilities (need not be normalized).
    return_tree : bool, optional
        If True, also return the final reduction tree 's'.

    Returns
    -------
    code : list[str]
        Huffman codewords, code[i] corresponds to symbol i+1 in MATLAB terms.
    s : object
        Reduction tree (only if return_tree=True).
    """
    p = np.asarray(p)
    if p.ndim == 2 and min(p.shape) == 1:
        p = p.reshape(-1)
    elif p.ndim != 1:
        raise ValueError("P must be a real numeric vector.")

    if not np.isrealobj(p) or not np.issubdtype(p.dtype, np.number):
        raise ValueError("P must be a real numeric vector.")

    n = int(p.size)
    if n == 0:
        return ([], []) if return_tree else []

    # Shared CODE container (MATLAB nested-function semantics).
    code = [""] * n

    if n > 1:
        ps = p.astype(float)
        ssum = np.sum(ps)
        if ssum <= 0:
            raise ValueError("Sum of probabilities must be positive.")
        ps = ps / ssum

        s = _reduce(ps)

        sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))

        def makecode(sc: Any, codeword: Any):
            """makecode."""
            # sc is either a list node {left,right} or a numeric leaf symbol index (1-based).
            if isinstance(sc, list):
                makecode(sc[0], codeword + [0])
                makecode(sc[1], codeword + [1])
            else:
                idx = int(sc) - 1  # MATLAB leaf indices are 1-based.
                code[idx] = "".join("1" if b else "0" for b in codeword)

        makecode(s, [])
    else:
        code = ["1"]
        s = [1]

    if return_tree:
        return code, s
    return code


def _reduce(p: Any):
    """
    Create Huffman source reduction tree (MATLAB reduce function).
    Leaves are 1-based symbol indices.
    """
    p = np.asarray(p, dtype=float).reshape(-1)

    # Starting tree with symbol nodes 1..N.
    s = [i + 1 for i in range(len(p))]

    while len(s) > 2:
        # MATLAB: [p, i] = sort(p)
        i = np.argsort(p, kind="mergesort")
        p = p[i]

        # Merge and prune two lowest probabilities.
        p[1] = p[0] + p[1]
        p = p[1:]

        # Reorder tree for new probabilities.
        s = [s[int(k)] for k in i]

        # Merge and prune nodes analogously.
        s[1] = [s[0], s[1]]
        s = s[1:]

    return s
