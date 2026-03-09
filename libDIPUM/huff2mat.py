from typing import Any
import numpy as np

from libDIPUM.huffman import huffman


def _build_search_table(map_codes: Any):
    """
    MATLAB-faithful construction of code/link tables from huff2mat.m.
    Uses 1-based node indexing semantics internally.
    """
    code = ["", "0", "1"]
    link = [2, 0, 0]
    left = [2, 3]
    found = 0
    tofind = len(map_codes)

    while left and (found < tofind):
        node = left[0]
        token = code[node - 1]

        # MATLAB: look = find(strcmp(map, code{left(1)}))
        look = [i + 1 for i, s in enumerate(map_codes) if s == token]

        if len(look) > 0:
            # Negative index points to symbol entry in map.
            link[node - 1] = -look[0]
            left = left[1:]
            found += 1
        else:
            ln = len(code)
            link[node - 1] = ln + 1

            link.extend([0, 0])
            code.append(code[node - 1] + "0")
            code.append(code[node - 1] + "1")

            left = left[1:]
            left.extend([ln + 1, ln + 2])

    return np.asarray(link, dtype=int)


def _bits_from_codewords(code_field: Any, pad: Any = 0):
    """
    Yield bits MSB->LSB from MATLAB-style uint16 code words.
    Also supports legacy byte-packed code fields.
    """
    if isinstance(code_field, (bytes, bytearray)):
        bits = []
        for bv in code_field:
            bits.extend([(bv >> i) & 1 for i in range(7, -1, -1)])
        if pad > 0:
            bits = bits[:-pad]
        return bits

    words = np.asarray(code_field).reshape(-1)
    bits = []
    for w in words:
        w = int(w)
        bits.extend([(w >> i) & 1 for i in range(15, -1, -1)])
    return bits


def _unravel(code_field: Any, link: Any, nvals: Any, pad: Any = 0):
    """
    Decode code words using UNRAVEL state-machine semantics.
    Returns 1-based symbol indices into the Huffman map.
    """
    bits = _bits_from_codewords(code_field, pad=pad)
    out = np.zeros(nvals, dtype=float)

    k = 0
    # UNRAVEL uses state n = 0 as the entry point. We keep 0-based state.
    state = 0
    nstates = len(link)

    for b in bits:
        v = int(link[state])
        if v <= 0:
            # Defensive reset; valid streams should not consume bits in a leaf state.
            state = 0
            v = int(link[state])

        # Transition:
        # bit=0 -> state [LINK(n)-1]
        # bit=1 -> state [LINK(n)]
        state = (v - 1) if int(b) == 0 else v
        if state < 0 or state >= nstates:
            raise ValueError("Invalid Huffman state transition.")

        # If terminal state, output |LINK(state)| and reset state to 0.
        lv = int(link[state])
        if lv < 0:
            out[k] = -lv
            k += 1
            if k >= nvals:
                break
            state = 0

    if k < nvals:
        raise ValueError("Decoded symbol count does not match target matrix size.")

    return out


def huff2mat(y: Any):
    """
    Decode a Huffman encoded matrix structure returned by mat2huff.
    MATLAB-faithful translation of DIPUM huff2mat.m.
    """
    if (
        not isinstance(y, dict)
        or "min" not in y
        or "size" not in y
        or "hist" not in y
        or "code" not in y
    ):
        raise ValueError("The input must be a structure as returned by MAT2HUFF.")

    sz = np.asarray(y["size"]).reshape(-1).astype(float)
    m = int(sz[0])
    n = int(sz[1])

    xmin = float(np.asarray(y["min"]).reshape(-1)[0]) - 32768.0
    map_codes = huffman(np.asarray(y["hist"]).reshape(-1).astype(float))

    link = _build_search_table(map_codes)

    x = _unravel(y["code"], link, m * n, pad=int(y.get("pad", 0)))
    x = x + xmin - 1.0
    x = np.asarray(x, dtype=float).reshape((m, n), order="F")
    return x
